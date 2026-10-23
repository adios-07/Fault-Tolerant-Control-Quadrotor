#!/usr/bin/python3
"""Rotor failure detector. Publishes /rotor_failure/detected at 50 Hz.

Two independent checks:
  ESC: each rotor's measured speed vs the speed PX4 commanded.
       A rotor much slower than commanded is stopped or weak.
  IMU: the torque the body actually felt vs the torque the commanded speeds should make.
       If rotor i stops pushing, exactly rotor i's torque goes missing.
       This also catches a lost propeller, where the motor still spins at full speed.

The maths lives in Detector (no ROS), so bench/replay.py can run it offline on recorded data.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from actuator_msgs.msg import Actuators
from sensor_msgs.msg import JointState
from px4_msgs.msg import VehicleOdometry, VehicleCommand
from rotor_fail.msg import RotorFailure

# Vehicle, from models/custom_x500/model.sdf
KF = 1.18e-05           # thrust = KF * speed^2 [N]
KM = 0.016              # yaw torque = KM * thrust [Nm]
SLOWDOWN = 10.0         # Gazebo joint speed * 10 = real rotor speed
INERTIA = np.array([0.0152, 0.0152, 0.028])
# Motors 1-4 (rotor_0..3): position in body FRD [m] and spin (+1 = CCW from above)
ROTOR_X = np.array([0.17678, -0.17678, 0.17678, -0.17678])
ROTOR_Y = np.array([0.17678, -0.17678, -0.17678, 0.17678])
ROTOR_SPIN = np.array([1, 1, -1, -1])

# Tuning
MIN_SPEED = 100.0       # [rad/s] commanded speed below this = motors off, skip checks
WEAK_RATIO = 0.9        # measured / commanded speed below this = rotor not keeping up
STOP_RATIO = 0.15       # below this = rotor stopped
LOSS_THRESH = 0.3       # IMU: fraction of one rotor's torque that must be missing
ESC_HOLD = 5            # ticks (0.1 s) the ESC check must hold before declaring
IMU_HOLD = 3            # ticks (0.06 s) the IMU check must hold; tuned on the benchmark (was 8, see bench/)
SMOOTH = 0.2            # low-pass weight on the torque residual per odometry sample; tuned (was 0.3)

DETECT_MARKER = 31011   # MAV_CMD_USER_2: param1 motor, param2 cause (1 stop, 2 weak, 3 prop), param3 severity,
                        # param5 sim time of the decision [s]
CAUSES = ('motor_stop', 'weak_motor', 'prop_loss')


def rotor_torques(speed):
	"""Body torque (roll, pitch, yaw) [Nm] each rotor makes at the given speed, one row per rotor."""
	thrust = KF * speed ** 2
	return np.column_stack([-ROTOR_Y * thrust, ROTOR_X * thrust, ROTOR_SPIN * KM * thrust])


class Detector:
	"""Detection maths without ROS. Feed it with on_cmd / on_joints / on_odom and call check() every 20 ms."""

	def __init__(self, imu_hold=IMU_HOLD, smooth=SMOOTH):
		self.imu_hold = imu_hold
		self.smooth = smooth
		self.cmd = None                 # commanded rotor speeds [rad/s]
		self.speed = None               # measured rotor speeds [rad/s]
		self.residual = np.zeros(3)     # felt - expected body torque [Nm]
		self.last_w = None
		self.last_t = None
		self.esc_ticks = 0
		self.imu_ticks = 0
		self.armed = False
		self.raw_detections = 0         # every time a check fires, held verdict or not (false-alarm counting)
		self.failed = 0
		self.cause = 'none'
		self.severity = 0.0
		self.esc_motor = 0
		self.imu_motor = 0
		self.ratio = np.ones(4)

	def on_cmd(self, cmd):
		self.cmd = cmd

	def on_joints(self, speed):
		self.speed = speed

	def on_odom(self, w, t):
		"""w: body rates FRD [rad/s], t: sample time [s]."""
		if not np.all(np.isfinite(w)):
			return
		if self.cmd is not None and self.last_t is not None and t > self.last_t:
			accel = (w - self.last_w) / (t - self.last_t)
			felt = INERTIA * accel + np.cross(w, INERTIA * w)   # Euler's rotation equation
			expected = rotor_torques(self.cmd).sum(axis=0)
			self.residual += self.smooth * (felt - expected - self.residual)
		self.last_w, self.last_t = w, t

	def check(self):
		"""One 20 ms tick. Returns True on the tick a new failure is declared."""
		if self.cmd is None or self.speed is None:
			return False
		running = self.cmd > MIN_SPEED

		# A new flight (motors off, then all running) clears the previous verdict
		if not running.any():
			self.armed = False
		elif running.all() and not self.armed:
			self.armed = True
			self.failed, self.cause, self.severity = 0, 'none', 0.0

		# ESC check: the rotor furthest below its commanded speed
		self.ratio = np.where(running, self.speed / np.maximum(self.cmd, 1.0), 1.0)
		self.esc_motor = int(np.argmin(self.ratio)) + 1 if self.ratio.min() < WEAK_RATIO else 0

		# IMU check: how much of each rotor's torque is missing (1 = all of it, ~0 for the others)
		loss = np.zeros(4)
		self.imu_motor = 0
		if running.all():
			torques = rotor_torques(self.cmd)
			loss = -(torques @ self.residual) / np.sum(torques ** 2, axis=1)
			if loss.max() > LOSS_THRESH:
				self.imu_motor = int(np.argmax(loss)) + 1

		self.esc_ticks = self.esc_ticks + 1 if self.esc_motor else 0
		self.imu_ticks = self.imu_ticks + 1 if self.imu_motor else 0
		if self.esc_ticks == ESC_HOLD or self.imu_ticks == self.imu_hold:
			self.raw_detections += 1

		# Declare once and hold it for the rest of the flight, so the tumble and crash can't change it.
		if self.failed:
			return False
		if self.esc_ticks >= ESC_HOLD:
			motor = self.esc_motor
		elif self.imu_ticks >= self.imu_hold:
			motor = self.imu_motor
		else:
			return False
		# Whichever check fired first only names the motor. The cause comes from that motor's speed now:
		# speed gone = motor stopped, speed low = weak motor, speed normal but torque missing = prop loss.
		r = self.ratio[motor - 1]
		self.failed = motor
		if r < WEAK_RATIO:
			self.cause = 'motor_stop' if r < STOP_RATIO else 'weak_motor'
			self.severity = float(1.0 - r ** 2)        # thrust ~ speed^2
		else:
			self.cause = 'prop_loss'
			self.severity = float(min(loss[motor - 1], 1.0))
		return True


class DetectFailure(Node):
	def __init__(self):
		super().__init__('failure_detection')
		self.det = Detector()
		self.create_subscription(Actuators, '/rotor_cmd', self.on_cmd, qos_profile_sensor_data)
		self.create_subscription(JointState, '/rotor_joint_states', self.on_joints, qos_profile_sensor_data)
		self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.on_odom, qos_profile_sensor_data)
		self.pub = self.create_publisher(RotorFailure, '/rotor_failure/detected', 10)
		self.marker_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
		self.create_timer(0.02, self.check)

	def on_cmd(self, msg):
		self.det.on_cmd(np.abs(np.array(msg.velocity[:4])))

	def on_joints(self, msg):
		speeds = dict(zip(msg.name, msg.velocity))
		self.det.on_joints(np.abs([speeds[f'rotor_{i}_joint'] for i in range(4)]) * SLOWDOWN)

	def on_odom(self, msg):
		self.det.on_odom(np.array(msg.angular_velocity), msg.timestamp_sample * 1e-6)

	def check(self):
		d = self.det
		declared = d.check()
		if d.cmd is None or d.speed is None:
			return

		msg = RotorFailure()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.failed_motor = d.failed
		msg.cause = d.cause
		msg.severity = d.severity
		msg.confirmed = d.esc_motor != 0 and d.esc_motor == d.imu_motor
		msg.esc_motor = d.esc_motor
		msg.imu_motor = d.imu_motor
		msg.speed_ratio = d.ratio.tolist()
		msg.torque_residual = d.residual.tolist()
		self.pub.publish(msg)
		if declared:  # after publishing: logging and the first PX4 marker publish take 20-40 ms
			self.log_marker(d.failed, CAUSES.index(d.cause) + 1, d.severity, self.get_clock().now().nanoseconds * 1e-9)
			self.get_logger().warn(f'motor {d.failed} failed: {d.cause}, severity {d.severity:.2f}')

	def log_marker(self, p1, p2, p3, t_event):
		"""Dummy vehicle_command so the detection lands in PX4's .ulg log. PX4 rejects the unknown command,
		but the logger records it. The message arrives some ms after the decision, so param5 carries the exact
		sim time of the decision [s] (PX4 lockstep time = Gazebo sim time)."""
		msg = VehicleCommand()
		msg.timestamp = 0  # PX4's DDS bridge stamps it on arrival with PX4 time
		msg.command = DETECT_MARKER
		msg.param1, msg.param2, msg.param3, msg.param5 = float(p1), float(p2), float(p3), float(t_event)
		msg.target_system = msg.target_component = 1
		msg.source_system = msg.source_component = 1
		msg.from_external = True
		self.marker_pub.publish(msg)


def main():
	rclpy.init()
	try:
		rclpy.spin(DetectFailure())
	except (KeyboardInterrupt, ExternalShutdownException):
		pass


if __name__ == '__main__':
	main()
