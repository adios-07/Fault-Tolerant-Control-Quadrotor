#!/usr/bin/python3
"""Fault-tolerant flight with the learned policy (i025a), handed over when our detector declares a failure.

Before the handover the node runs in shadow mode: it feeds the policy wrapper (pftc.py) every 10 ms so its
filters are warm, using PX4's actual motor commands as the policy's thrust memory, and publishes nothing.
On the first /rotor_failure/detected with failed_motor != 0 it takes over: offboard direct-actuator mode,
policy thrusts sent to PX4 as /fmu/in/actuator_motors at 100 Hz, holding the position it had at handover.

State sources (param `state`):
  gyro      default, fully onboard: EKF2 until the handover; then attitude = the EKF2 attitude from 150 ms before
            detection integrated with the raw gyro (EKF2 drifts during the spin, the gyro does not; magnetometer
            heading aiding exists but is off, see MAG_AID), and position/velocity from pos_kf.py (accelerometer
            rotated by the gyro attitude + GPS), started from EKF2's
  ekf       PX4 EKF2 throughout (for comparison: its attitude drifts 15-20 deg in the first second of the spin)
  gt        Gazebo ground truth (diagnosis)

Frames: PX4 gives NED / FRD, the policy wants world ENU / body FLU (see policies/README in the training repo).
"""
import collections
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from actuator_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from px4_msgs.msg import (ActuatorMotors, OffboardControlMode, SensorCombined, SensorGps, VehicleCommand,
                          VehicleMagnetometer, VehicleOdometry)
from rotor_fail.msg import RotorFailure

from pftc import PFTC, MOTOR_CONSTANT
from pos_kf import PosVelKF

NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])     # also FRD -> FLU
OMEGA_MIN, OMEGA_MAX = 150.0, 1000.0        # PX4 SIM_GZ_EC_MIN / MAX [rad/s]
HANDOVER_MARKER = 31012                     # .ulg marker: param1 motor, param5 sim time of the handover [s]
DESCENT_MARKER, DOWN_MARKER = 31013, 31014  # .ulg markers: descent started, motors cut after touchdown
# Landing after the recovery. The policy flies toward its target, so the target slides
# down at V_DESC: p - target stays at ~V_DESC (inside the trained range), a jump to the ground would not.
HOLD_S = 10.0                               # [s] hover at the hold point once settled, then descend
V_DESC = 0.4                                # [m/s] descent rate; landing gear takes this without a flare
CONTACT_ACC = 20.0                          # [m/s^2] touchdown: flight never exceeds ~10.4, ground contact 80-160
SEED_BACK = 15                              # odometry messages (100 Hz): gyro attitude starts from EKF2 150 ms before detection
# Magnetometer heading aiding: off. Live over DDS the mag/IMU sample times disagree by tens of ms (PX4 shifts
# timestamps onto the agent clock and publishes mag up to 60 ms late), which at 20 rad/s is 20-70 deg of heading.
# Needs UXRCE_DDS_SYNCT=0 and our own message stamps first. Without it a real gyro's z scale error drifts heading.
MAG_AID = False
MAG_TAU = 0.5                               # [s] heading correction time constant (a gyro z scale error drifts ~11 deg/s per %)
MAG_MAX_TILT = np.radians(45.0)             # no mag correction while tumbling: its level projection is poor
MIN_SEVERITY = 0.0                          # hand over on any detection: i025a is trained for partial loss 30-80 %
                                            # (0.6 kept i022c away from partial losses PX4 flew better)


def rotation(v):
	"""Rotation matrix of the rotation vector v (Rodrigues; scipy per 250 Hz sample was too slow for the callback)."""
	th = np.linalg.norm(v)
	k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
	if th < 1e-9:
		return np.eye(3) + k
	k /= th
	return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)


class PftcNode(Node):
	def __init__(self):
		super().__init__('pftc')
		default = os.path.join(get_package_share_directory('rotor_fail'), 'policies', 'x500v2_i025a', 'policy.onnx')
		self.pftc = PFTC(self.declare_parameter('policy', default).value)
		source = self.declare_parameter('state', 'gyro').value
		assert source in ('gyro', 'ekf', 'gt'), source
		self.att_src = self.pos_src = source   # attitude and position sources (the code keeps them apart)
		self.att = None             # (R body FLU -> world ENU, w FLU) from the attitude source
		self.trans = None           # (p, v) ENU from the position source
		# Topics are lined up by arrival order, never by timestamp: PX4 shifts each message's timestamps onto the
		# agent clock with an offset that keeps changing (fast in a lockstep sim), so cross-topic times disagree.
		self.n_imu = 0              # IMU messages received so far
		self.ekf_hist = collections.deque(maxlen=100)   # (n_imu at arrival, R NED<-FRD)
		self.imu_hist = collections.deque(maxlen=250)   # (PX4 time [s], gyro FRD), ~1 s
		self.r_gyro = None          # R NED<-FRD integrated from the gyro after the handover
		self.gaps = []              # IMU gaps > 6 ms after the handover [s], logged
		self.integrated = 0.0       # IMU time integrated since the seed [s]; vs sim time it shows dropped samples
		self.mag_hist = collections.deque(maxlen=100)   # (PX4 sample time [s], field FRD [gauss])
		self.psi_ref = None         # heading of the horizontal field in NED, learned at the seed
		self.t_mag = None
		self.imu_new, self.mag_new = [], []   # samples not integrated yet
		self.ekf_pv = None          # EKF2 (p, v) NED, starts the KF
		self.kf = None              # PosVelKF after the handover (state 'gyro')
		self.gps_ref = None         # (lat, lon, alt, KF position NED) at the first GPS fix after the handover
		self.v_prev = None
		# sim diagnosis: CSV of the onboard estimator's error vs Gazebo truth after the handover (time, tilt, signed heading)
		self.debug = self.declare_parameter('debug_csv', '').value
		self.gt_r = None
		self.debug_rows = []
		self.gt_prev = None
		self.acc_b = None           # specific force, FLU [m/s^2]
		self.cmd_thrust = None      # PX4's commanded thrust per rotor [N]
		self.active = False
		self.target = None
		self.yaw_fix = None         # world rotation that makes the starting heading 0, as in training

		qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
		                 history=HistoryPolicy.KEEP_LAST, depth=1)
		if 'gt' in (self.att_src, self.pos_src):
			self.create_subscription(Odometry, '/gt_odom', self.on_gt, qos_profile_sensor_data)
		elif self.debug:
			self.create_subscription(Odometry, '/gt_odom', self.on_gt_truth, qos_profile_sensor_data)
		if source != 'gt':
			self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.on_odom, qos_profile_sensor_data)
		# every mode: the lander's touchdown detection uses the accelerometer (with state gt, tick() still feeds the
		# policy the ground-truth specific force). Deep queue: a dropped sample in a 20 rad/s spin is lost yaw.
		imu_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100)
		self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.on_imu, imu_qos)
		if self.pos_src == 'gyro':
			self.create_subscription(SensorGps, '/fmu/out/vehicle_gps_position', self.on_gps, qos_profile_sensor_data)
		if self.att_src == 'gyro':
			if MAG_AID:
				self.create_subscription(VehicleMagnetometer, '/fmu/out/vehicle_magnetometer', self.on_mag, qos_profile_sensor_data)
		self.create_subscription(Actuators, '/rotor_cmd', self.on_cmd, qos_profile_sensor_data)
		self.create_subscription(RotorFailure, '/rotor_failure/detected', self.on_detected, 10)
		self.mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
		self.motor_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos)
		self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
		self.create_timer(0.01, self.tick)

	def on_odom(self, msg):
		if not (np.all(np.isfinite(msg.q)) and np.all(np.isfinite(msg.position))):
			return
		r_ned_frd = Rotation.from_quat([msg.q[1], msg.q[2], msg.q[3], msg.q[0]]).as_matrix()  # PX4 q is w x y z
		if self.yaw_fix is None:
			self.set_yaw_fix(self.flu_enu(r_ned_frd))
		self.ekf_hist.append((self.n_imu, r_ned_frd))
		if self.att_src == 'ekf' or (self.att_src == 'gyro' and self.r_gyro is None):
			self.att = (self.yaw_fix @ self.flu_enu(r_ned_frd), FLU_TO_FRD @ np.array(msg.angular_velocity, float))
		self.ekf_pv = (np.array(msg.position, float), np.array(msg.velocity, float))
		if self.pos_src == 'ekf' or (self.pos_src == 'gyro' and self.kf is None):
			self.set_trans(*self.ekf_pv)

	def set_trans(self, p_ned, v_ned):
		self.trans = (self.yaw_fix @ NED_TO_ENU @ p_ned, self.yaw_fix @ NED_TO_ENU @ v_ned)

	@staticmethod
	def flu_enu(r_ned_frd):
		return NED_TO_ENU @ r_ned_frd @ FLU_TO_FRD                                          # body FLU -> ENU

	def set_yaw_fix(self, r):
		# Training always started at heading 0. Dynamics don't depend on heading, so rotate the world once
		self.yaw_fix = Rotation.from_euler('z', -np.arctan2(r[1, 0], r[0, 0])).as_matrix()

	def on_gt(self, msg):
		"""Ground-truth pose; velocity and body rates are differenced from it (the plugin's own twist is not body rates)."""
		o, pos = msg.pose.pose.orientation, msg.pose.pose.position
		t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		r = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()      # body FLU -> world ENU
		if self.yaw_fix is None and self.att_src == 'gt':
			self.set_yaw_fix(r)
		if self.yaw_fix is None:      # mixed sources: the world frame follows the attitude source's heading
			return
		r = self.yaw_fix @ r
		p = self.yaw_fix @ np.array([pos.x, pos.y, pos.z])
		if self.gt_prev is not None and t > self.gt_prev[0]:
			t0, p0, r0 = self.gt_prev
			if self.att_src == 'gt':
				self.att = (r, Rotation.from_matrix(r0.T @ r).as_rotvec() / (t - t0))
			if self.pos_src == 'gt':
				self.trans = (p, (p - p0) / (t - t0))
		self.gt_prev = (t, p, r)

	def on_gt_truth(self, msg):
		o = msg.pose.pose.orientation
		self.gt_r = self.flu_enu(Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())   # same map back: R NED<-FRD

	def log_error(self):
		r_est, r_gt = self.r_gyro, self.gt_r
		tilt = np.degrees(np.arccos(np.clip(r_est[:, 2] @ r_gt[:, 2], -1.0, 1.0)))
		e = r_est @ r_gt.T                                   # world-frame error rotation
		self.debug_rows.append([self.get_clock().now().nanoseconds * 1e-9, tilt, np.degrees(np.arctan2(e[1, 0], e[0, 0]))])
		if len(self.debug_rows) % 100 == 0:
			np.savetxt(self.debug, np.array(self.debug_rows), delimiter=',', fmt='%.3f', header='t,tilt_err_deg,heading_err_deg')

	def gt_specific_force(self):
		"""As in training: R^T((v_k - v_{k-1}) / dt + g), once per 10 ms tick."""
		p, v, q, w = self.state()
		dv = np.zeros(3) if self.v_prev is None else (v - self.v_prev) / 0.01
		self.v_prev = v
		return Rotation.from_quat(q).as_matrix().T @ (dv + np.array([0.0, 0.0, 9.81]))

	def state(self):
		"""(p, v, q_xyzw, w) in ENU / FLU, or None until both sources have reported."""
		if self.att is None or self.trans is None:
			return None
		return self.trans[0], self.trans[1], Rotation.from_matrix(self.att[0]).as_quat(), self.att[1]

	def on_imu(self, msg):
		# only queue: tick() integrates. A slow callback here makes DDS drop samples, i.e. lost rotation in the spin
		sample = (msg.timestamp * 1e-6, np.array(msg.gyro_rad, float),
		          np.array(msg.accelerometer_m_s2, float), msg.gyro_integral_dt * 1e-6)
		self.imu_hist.append(sample)
		self.n_imu += 1
		if self.r_gyro is not None:
			self.imu_new.append(sample)
		self.acc_b = FLU_TO_FRD @ sample[2]
		self.acc_peak = max(getattr(self, 'acc_peak', 0.0), np.linalg.norm(sample[2]))   # peak since the last tick

	def on_mag(self, msg):
		if not self.imu_hist:
			return
		# sample time on the IMU clock: the latest IMU sample's time minus how much older this mag sample is.
		# Both stamps of one message share the same clock offset, so (timestamp - timestamp_sample) is clean.
		self.mag_hist.append((self.imu_hist[-1][0] - (msg.timestamp - msg.timestamp_sample) * 1e-6,
		                      np.array(msg.magnetometer_ga, float)))
		if self.t_mag is not None:
			self.mag_new = self.mag_new[-50:] + [self.mag_hist[-1]]

	def correct_heading(self, t, gyro):
		"""Pull the gyro attitude's heading toward the magnetometer's, using the mag samples taken up to time t.
		The attitude is rotated back to each sample's own time first: 4 ms at 20 rad/s is 5 deg of heading."""
		while self.mag_new and self.mag_new[0][0] <= t:
			tm, m = self.mag_new.pop(0)
			if t - tm > 0.05:                 # stale, or the timesync offset moved between the two topics: skip it
				continue
			r = self.r_gyro @ rotation(-gyro * (t - tm))
			m_ned = r @ m
			psi = np.arctan2(m_ned[1], m_ned[0])
			if self.psi_ref is None:          # first sample after the seed: field direction, attitude still exact
				self.psi_ref = psi
			elif r[2, 2] > np.cos(MAG_MAX_TILT):
				e = np.angle(np.exp(1j * (self.psi_ref - psi)))
				self.r_gyro = rotation(np.array([0.0, 0.0, e * min(tm - self.t_mag, 0.1) / MAG_TAU])) @ self.r_gyro
			self.t_mag = tm

	def on_gps(self, msg):
		if self.kf is None or not msg.fix_type >= 3:
			return
		lat, lon, alt = np.radians(msg.latitude_deg), np.radians(msg.longitude_deg), msg.altitude_msl_m
		if self.gps_ref is None:
			self.gps_ref = (lat, lon, alt, self.kf.p)      # GPS deltas from here on, so no EKF origin is needed
		lat0, lon0, alt0, p0 = self.gps_ref
		p = p0 + np.array([(lat - lat0) * 6371000.0, (lon - lon0) * 6371000.0 * np.cos(lat0), -(alt - alt0)])
		self.kf.update_gps(p, [msg.vel_n_m_s, msg.vel_e_m_s, msg.vel_d_m_s])

	def step(self, t, gyro, acc, dt):
		"""One IMU sample: rotate by gyro x its own integration interval, heading aiding, KF prediction.
		Never timestamp differences: PX4 shifts outgoing timestamps onto the agent's clock and keeps adjusting the
		offset, so their steps are clock corrections, not motion."""
		self.r_gyro = self.r_gyro @ rotation(gyro * dt)
		self.integrated += dt
		self.t_gyro = t
		if self.t_mag is not None:
			self.correct_heading(t, gyro)
		if self.kf is not None:
			self.kf.predict(self.r_gyro, acc, dt)

	def advance(self):
		"""Integrate the IMU samples received since the last call, each exactly once. Called every tick."""
		for t, gyro, acc, dt in self.imu_new:
			if t - self.t_gyro > 1.5 * dt:    # dropped samples or a timesync step: logged, not integrated
				self.gaps.append(t - self.t_gyro)
			self.step(t, gyro, acc, dt)
		self.imu_new = []
		self.att = (self.yaw_fix @ self.flu_enu(self.r_gyro), FLU_TO_FRD @ self.imu_hist[-1][1])
		if self.kf is not None:
			self.set_trans(self.kf.p, self.kf.v)

	def seed_gyro(self):
		"""Start from the EKF2 attitude SEED_BACK before now (it is still good then), catch up on the buffered gyro."""
		n, self.r_gyro = self.ekf_hist[max(len(self.ekf_hist) - 1 - SEED_BACK, 0)]
		self.imu_new = list(self.imu_hist)[max(len(self.imu_hist) - (self.n_imu - n), 0):]   # IMU samples that arrived after it
		self.t_gyro = self.imu_new[0][0] - self.imu_new[0][3]
		back = sum(s[3] for s in self.imu_new)
		if self.mag_hist:                     # mag aiding on: the field direction is taken from the first sample after the seed
			self.t_mag = self.t_gyro
			self.mag_new = [s for s in self.mag_hist if s[0] > self.t_gyro]
		self.advance()
		# frame self-check: right after detection EKF2 is still within ~1 deg, so a large gap means a frame bug
		gap = np.degrees(Rotation.from_matrix(self.ekf_hist[-1][1].T @ self.r_gyro).magnitude())
		self.get_logger().warn(f'gyro attitude seeded {back:.3f} s back; vs EKF2 now: {gap:.2f} deg; '
		                       f'mag heading aiding {"on" if self.t_mag is not None else "OFF (no magnetometer data)"}')

	def on_cmd(self, msg):
		self.cmd_thrust = MOTOR_CONSTANT * np.square(np.array(msg.velocity[:4], float))

	def on_detected(self, msg):
		if msg.failed_motor and (msg.cause == 'motor_stop' or msg.severity >= MIN_SEVERITY):
			self.handover(msg.failed_motor, msg.cause)

	def handover(self, motor, why):
		if self.active or self.state() is None:
			return
		if self.att_src == 'gyro':
			self.seed_gyro()
		if self.pos_src == 'gyro':
			self.kf = PosVelKF(*self.ekf_pv)
		self.active, self.target = True, self.state()[0].copy()
		self.phase, self.t_settled, self.t_slow = 'hold', None, None
		# PX4 re-reads the offboard flags only at 2 Hz or on a status change, so on its own it would keep its
		# attitude controller up to 500 ms after we switch to direct actuator. Announce the new flags, then re-request
		# Offboard: an accepted mode request forces the control-mode update on this tick.
		hb = OffboardControlMode()
		hb.direct_actuator = True
		self.mode_pub.publish(hb)
		self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
		self.marker(motor)
		self.get_logger().warn(f'handover to policy: motor {motor} ({why}), holding {self.target.round(2)}')

	def tick(self):
		if self.r_gyro is not None:
			self.advance()
			now = self.get_clock().now().nanoseconds * 1e-9
			self.t_seed_sim = getattr(self, 't_seed_sim', now - self.integrated)
			if int(now - self.t_seed_sim) // 5 != int(now - self.t_seed_sim - 0.01) // 5:
				self.get_logger().info(f'IMU integrated {self.integrated:.3f} s in {now - self.t_seed_sim:.3f} s of sim time '
				                       f'(gaps > 6 ms: {len(self.gaps)})')
		if self.att_src == 'gt' and self.state() is not None:
			self.acc_b = self.gt_specific_force()
		if self.state() is None or self.acc_b is None or self.cmd_thrust is None:
			return
		p, v, q, w = self.state()
		target = self.target if self.active else p
		if not hasattr(self, 'logged'):  # frame check: at hover this should read q ~ [1,0,0,0], acc_b ~ [0,0,9.81]
			self.logged = True
			obs = self.pftc.observation(p, v, q, w, self.acc_b, target)
			self.get_logger().info(f'first observation: q(wxyz) {obs[6:10].round(3)}, acc_b {self.acc_b.round(2)}')
		if not self.active:
			# Shadow mode from take-off: keep the policy's filters current with the thrust PX4 applies.
			# The same PFTC object then takes over; a cold start 80 ms into a failure crashes.
			self.pftc.track(p, v, q, w, target, self.cmd_thrust)
			return

		if self.debug and self.r_gyro is not None and self.gt_r is not None:
			self.log_error()
		self.land(p, v)
		if self.phase == 'down':
			omega = np.zeros(4)                   # all motors stopped
		else:
			omega = PFTC.rotor_speed(self.pftc.step(p, v, q, w, self.acc_b, target))
		hb = OffboardControlMode()
		hb.direct_actuator = True
		self.mode_pub.publish(hb)
		motors = ActuatorMotors()
		control = np.full(12, np.nan, dtype=np.float32)
		# PX4 maps 0..1 to 150..1000 rad/s; below half the floor, stopping the motor is closer to what was asked
		control[:4] = np.where(omega < OMEGA_MIN / 2, np.nan,
		                       np.clip((omega - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN), 0.0, 1.0))
		motors.control = control.tolist()
		self.motor_pub.publish(motors)

	def land(self, p, v):
		"""hold -> descend -> down. Hold until settled (< 0.5 m/s, within 0.5 m) for 1 s plus HOLD_S, slide the
		target down at V_DESC with x, y frozen, and cut the motors once the vertical speed has collapsed while the
		target keeps descending (the ground is holding the drone up); a spinning drone left on its motors skids."""
		now = self.get_clock().now().nanoseconds * 1e-9
		peak, self.acc_peak = getattr(self, 'acc_peak', 0.0), 0.0
		if self.phase == 'hold':
			settled = np.linalg.norm(v) < 0.5 and np.linalg.norm((p - self.target)[:2]) < 0.5   # the hold orbit itself runs ~0.3 m/s
			self.t_settled = (self.t_settled or now) if settled else None
			if self.t_settled is not None and now - self.t_settled >= 1.0 + HOLD_S:
				self.phase, self.target[2] = 'descend', p[2]     # start from where it is: the hold sinks slowly
				self.command(DESCENT_MARKER, p5=now)
				self.get_logger().warn(f'descending from {p[2]:.2f} m at {V_DESC} m/s')
		elif self.phase == 'descend':
			self.target[2] -= V_DESC * 0.01
			stopped = abs(v[2]) < 0.1 and p[2] < 1.0   # sinking at V_DESC until the ground stops it; 1 m: loose sanity gate
			self.t_slow = (self.t_slow or now) if stopped else None
			impact = peak > CONTACT_ACC and p[2] < 1.0
			if impact or (self.t_slow is not None and now - self.t_slow >= 0.2):
				self.phase = 'down'
				self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0)   # force disarm
				self.command(DOWN_MARKER, p5=now)
				self.get_logger().warn(f'touchdown at {p[2]:.2f} m ({"impact" if impact else "stopped"}, '
				                       f'peak {peak:.0f} m/s^2): motors cut')

	def marker(self, motor):
		self.command(HANDOVER_MARKER, float(motor), p5=self.get_clock().now().nanoseconds * 1e-9)

	def command(self, command, p1=0.0, p2=0.0, p5=0.0):
		msg = VehicleCommand()
		msg.timestamp = 0  # PX4's DDS bridge stamps it on arrival with PX4 time
		msg.command = command
		msg.param1, msg.param2, msg.param5 = float(p1), float(p2), float(p5)
		msg.target_system = msg.target_component = 1
		msg.source_system = msg.source_component = 1
		msg.from_external = True
		self.cmd_pub.publish(msg)


def main():
	rclpy.init()
	try:
		rclpy.spin(PftcNode())
	except (KeyboardInterrupt, ExternalShutdownException):
		pass


if __name__ == '__main__':
	main()
