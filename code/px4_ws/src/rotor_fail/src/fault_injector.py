#!/usr/bin/python3
"""Random rotor fault injector.

After a 5 s warm-up, injects one fault at a random moment. The wait is exponential with
rate LAM (mean 1/LAM s): the fault is equally likely at every instant, no matter how long
the drone has already flown.

Fault size: at least 50 % of the motor's thrust is lost, and half of all faults are total (100 %).
Publishes "<motor> <mode> <intensity>" on /motor_fault (bridged to RotorFaultSystem in Gazebo)
and marks the injection in PX4's .ulg as vehicle_command 31010.
"""
import random

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from px4_msgs.msg import VehicleCommand

WARMUP = 15.0            # [s]
LAM = 1.0 / 6.0         # [1/s] fault rate after warm-up, mean wait 6 s
MODES = ('stop', 'weak', 'prop')
INJECT_MARKER = 31010   # MAV_CMD_USER_1: param1 motor, param2 mode (1 stop, 2 weak, 3 prop), param3 intensity,
                        # param5 time of the injection [s]


def log_marker(pub, command, p1, p2, p3, t_event):
	"""Send a dummy vehicle_command so the event lands in PX4's .ulg log. PX4 rejects the unknown command,
	but the logger records it. param5 carries the exact time of the event [s] (sim time with use_sim_time)."""
	msg = VehicleCommand()
	msg.timestamp = 0  # PX4's DDS bridge stamps it on arrival with PX4 time
	msg.command = command
	msg.param1, msg.param2, msg.param3, msg.param5 = float(p1), float(p2), float(p3), float(t_event)
	msg.target_system = msg.target_component = 1
	msg.source_system = msg.source_component = 1
	msg.from_external = True
	pub.publish(msg)


class FaultInjector(Node):
	def __init__(self):
		super().__init__('fault_injector')
		self.mode = self.declare_parameter('mode', 'random').value   # stop | weak | prop | random
		self.pub = self.create_publisher(String, '/motor_fault', 10)
		self.marker_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

		delay = WARMUP + random.expovariate(LAM)
		self.timer = self.create_timer(delay, self.inject)
		self.get_logger().info(f'fault scheduled in {delay:.1f} s')

	def inject(self):
		self.timer.cancel()
		motor = random.randint(1, 4)
		mode = self.mode if self.mode in MODES else random.choice(MODES)
		if mode == 'stop' or random.random() < 0.5:
			intensity = 1.0
		else:
			intensity = round(random.uniform(0.5, 1.0), 2)

		self.pub.publish(String(data=f'{motor} {mode} {intensity}'))
		log_marker(self.marker_pub, INJECT_MARKER, motor, MODES.index(mode) + 1, intensity,
		           self.get_clock().now().nanoseconds * 1e-9)
		self.get_logger().warn(f'injected: motor {motor} {mode}, intensity {intensity}')


def main():
	rclpy.init()
	try:
		rclpy.spin(FaultInjector())
	except (KeyboardInterrupt, ExternalShutdownException):
		pass


if __name__ == '__main__':
	main()
