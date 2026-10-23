#!/usr/bin/python3
"""Scripted benchmark flight: arm, take off, fly one flight phase, inject a fault, exit.

Offboard control pattern (heartbeat before setpoint, set Offboard then arm) follows
px4_teleop/px4_teleop/control/mode_manager.py.

Fault run:   --ros-args -p phase:=yaw -p fault:="3 prop 0.7" -p offset:=4.2
Healthy run: --ros-args -p phase:=healthy -p duration:=780

Phases (10 s each, labels published on /bench/phase):
  hover    hold position at 3 m
  forward  2 m/s north at constant height (healthy mode alternates north and south)
  climb    1 m/s up
  descent  1 m/s down (takes off to 13 m first)
  yaw      hold position, spin at 1.5 rad/s
The fault goes on /motor_fault and is marked in the .ulg as vehicle_command 31010, like fault_injector.py.
The flyer exits `tail` seconds after the fault, or after `duration` in healthy mode.
With ftc:=true it stops sending setpoints when the detector declares a failure, so pftc_node can take over.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from rotor_fail.msg import RotorFailure
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition,
                          VehicleStatus)

RATE = 50.0             # [Hz] setpoint rate
PHASE_LEN = 10.0        # [s]
SPEED = 2.0             # [m/s] forward
CLIMB = 1.0             # [m/s] climb / descent
YAW_RATE = 1.5          # [rad/s]
INJECT_MARKER = 31010
MODES = ('stop', 'weak', 'prop')
HEALTHY_CYCLE = ('hover', 'forward', 'yaw', 'back', 'climb', 'descent')
NAN = float('nan')


class PhaseFlyer(Node):
	def __init__(self):
		super().__init__('phase_flyer')
		self.phase = self.declare_parameter('phase', 'hover').value
		self.fault = self.declare_parameter('fault', '').value           # "<motor> <mode> <intensity>"
		self.offset = self.declare_parameter('offset', 5.0).value        # [s] into the phase
		self.duration = self.declare_parameter('duration', 780.0).value  # [s] healthy mode only
		self.tail = self.declare_parameter('tail', 3.0).value            # [s] keep flying / recording after the fault
		self.ftc = self.declare_parameter('ftc', False).value            # hand over to pftc_node on detection
		self.handed_over = False

		qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
		                 history=HistoryPolicy.KEEP_LAST, depth=1)
		self.mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
		self.sp_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
		self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
		self.phase_pub = self.create_publisher(String, '/bench/phase', 10)
		self.fault_pub = self.create_publisher(String, '/motor_fault', 10)
		self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v4', self.on_status, qos)
		self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.on_pos, qos)
		self.create_subscription(RotorFailure, '/rotor_failure/detected', self.on_detected, 10)

		self.armed = False
		self.pos = None
		self.home = None                # (x, y) where we took off
		self.hold = None                # (x, y) held by the current phase: where it started
		self.label = None
		self.alt = 13.0 if self.phase == 'descent' else 3.0
		self.state = 'arming'           # arming -> takeoff -> phase -> done
		self.t_state = 0.0
		self.t_fault = None
		self.tick_n = 0
		self.create_timer(1.0 / RATE, self.tick)

	def on_status(self, msg):
		self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

	def on_pos(self, msg):
		self.pos = (msg.x, msg.y, msg.z)

	def on_detected(self, msg):
		# same rule as pftc_node (MIN_SEVERITY): below it PX4 keeps flying a partial loss
		if (self.ftc and msg.failed_motor and not self.handed_over
		        and (msg.cause == 'motor_stop' or msg.severity >= 0.0)):
			self.handed_over = True  # pftc_node takes the motors; stop our offboard stream so it is the only one
			self.get_logger().warn(f'failure detected (motor {msg.failed_motor}): handing over to the policy')

	def now(self):
		return self.get_clock().now().nanoseconds * 1e-9

	def command(self, command, p1=0.0, p2=0.0, p3=0.0, p5=0.0):
		msg = VehicleCommand()
		msg.timestamp = 0  # PX4's DDS bridge stamps it on arrival with PX4 time
		msg.command = command
		msg.param1, msg.param2, msg.param3, msg.param5 = float(p1), float(p2), float(p3), float(p5)
		msg.target_system = msg.target_component = 1
		msg.source_system = msg.source_component = 1
		msg.from_external = True
		self.cmd_pub.publish(msg)

	def send(self, position, velocity=(NAN, NAN, NAN), yawspeed=NAN):
		hb = OffboardControlMode()
		hb.position = not math.isnan(position[0]) or not math.isnan(position[2])
		hb.velocity = not all(math.isnan(v) for v in velocity)
		self.mode_pub.publish(hb)                       # heartbeat first, so the mode matches the setpoint
		sp = TrajectorySetpoint()
		sp.position = [float(p) for p in position]
		sp.velocity = [float(v) for v in velocity]
		sp.yaw = NAN                                    # heading is rate-controlled: 0 = keep the current heading
		sp.yawspeed = 0.0 if math.isnan(yawspeed) else float(yawspeed)
		self.sp_pub.publish(sp)

	def phase_setpoint(self, phase):
		if phase != self.label:                         # new phase: hold where it starts, not where we took off
			self.label, self.hold = phase, self.pos[:2]
		x, y = self.hold
		z = -self.alt
		if phase == 'forward':
			self.send((NAN, NAN, z), (SPEED, 0.0, NAN))
		elif phase == 'back':
			self.send((NAN, NAN, z), (-SPEED, 0.0, NAN))
		elif phase == 'climb':
			self.send((x, y, NAN), (NAN, NAN, -CLIMB))
		elif phase == 'descent':
			self.send((x, y, NAN), (NAN, NAN, CLIMB))
		elif phase == 'yaw':
			self.send((x, y, z), yawspeed=YAW_RATE)
		else:
			self.send((x, y, z))

	def tick(self):
		self.tick_n += 1
		t = self.now()
		if self.pos is None:
			return
		if self.home is None:
			self.home = self.pos[:2]

		if self.state == 'arming':
			self.send((self.home[0], self.home[1], -self.alt))
			if self.tick_n % 25 == 0 and not self.armed:  # retry every 0.5 s: Offboard first, then arm
				self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
				self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
			if self.armed:
				self.state, self.t_state = 'takeoff', t
			return

		if self.state == 'takeoff':
			self.send((self.home[0], self.home[1], -self.alt))
			if abs(self.pos[2] + self.alt) < 0.3 and t - self.t_state > 3.0:
				self.state, self.t_state = 'phase', t
			return

		if self.state == 'phase':
			elapsed = t - self.t_state
			if self.phase == 'healthy':
				label = HEALTHY_CYCLE[int(elapsed // PHASE_LEN) % len(HEALTHY_CYCLE)]
				if label in ('climb', 'descent'):
					self.alt = -self.pos[2]           # keep whatever height the vertical phase reached
				self.phase_setpoint(label)
				self.phase_pub.publish(String(data='forward' if label == 'back' else label))
				if elapsed > self.duration:
					self.finish('healthy session done')
				return

			if not self.handed_over:
				self.phase_setpoint(self.phase)
			self.phase_pub.publish(String(data=self.phase))
			if self.fault and self.t_fault is None and elapsed >= self.offset:
				self.inject(t)
			if self.t_fault is not None and t - self.t_fault > self.tail:
				self.finish('fault run done')

	def inject(self, t):
		motor, mode, intensity = self.fault.split()
		self.fault_pub.publish(String(data=self.fault))
		self.command(INJECT_MARKER, int(motor), MODES.index(mode) + 1, float(intensity), t)  # param5: exact sim time
		self.t_fault = t
		self.get_logger().warn(f'injected: motor {motor} {mode}, intensity {intensity}, phase {self.phase}')

	def finish(self, why):
		self.get_logger().info(why)
		self.state = 'done'
		raise SystemExit


def main():
	rclpy.init()
	try:
		rclpy.spin(PhaseFlyer())
	except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
		pass


if __name__ == '__main__':
	main()
