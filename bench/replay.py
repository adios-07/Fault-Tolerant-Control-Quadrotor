#!/usr/bin/python3
"""Replay recorded flights through the detector offline.

Feeds a run's bag (sim-time stamps) into failure_detection.Detector in time order and calls check()
every 20 ms of sim time, like the live 50 Hz timer. /motor_fault is never shown to the detector; it is
only the ground truth for scoring.

  python3 replay.py runs/tune_03 [runs/... ...]     print the result of each run
"""
import json
import os
import pickle
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

HERE = os.path.dirname(os.path.abspath(__file__))
# The ROS package sits next to bench/ in the live workspace, and under code/px4_ws/src/ in the
# submission tree. Try both so this runs unchanged in either.
for _cand in (os.path.join(HERE, '..', 'src'),
              os.path.join(HERE, '..', 'code', 'px4_ws', 'src', 'rotor_fail', 'src')):
    if os.path.isfile(os.path.join(_cand, 'failure_detection.py')):
        sys.path.insert(0, _cand)
        break
else:
    raise SystemExit('cannot locate failure_detection.py; looked in ../src and '
                     '../code/px4_ws/src/rotor_fail/src')
from failure_detection import Detector, SLOWDOWN  # noqa: E402

TICK = 0.02  # [s]


def read_bag(path):
	"""All messages of a bag as (time [s], topic, msg), in time order."""
	reader = rosbag2_py.SequentialReader()
	reader.open(rosbag2_py.StorageOptions(uri=path), rosbag2_py.ConverterOptions('', ''))
	types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
	out = []
	while reader.has_next():
		topic, data, t = reader.read_next()
		out.append((t * 1e-9, topic, deserialize_message(data, types[topic])))
	return out


def stamp(msg):
	return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def compact(inputs, grid):
	"""Drop rotor-speed samples the detector never sees: it only reads the newest one at each 20 ms tick.
	Commands and odometry are all kept, since the torque residual uses the command at every odometry sample.
	Replay results are unchanged (keeping a few extra samples near tick edges is harmless)."""
	keep, next_t = [], None
	for t, kind, value in reversed(inputs):
		if kind == 'joints':
			first_tick_after = grid + (np.floor((t - grid) / TICK) + 1) * TICK
			if next_t is not None and next_t < first_tick_after - 1e-6:
				continue
			next_t = t
		keep.append((t, kind, value))
	return keep[::-1]


def load(run_dir):
	"""A run's detector inputs and ground truth, decoded once from the bag and cached as inputs.pkl."""
	cache = os.path.join(run_dir, 'inputs.pkl')
	if os.path.exists(cache):
		return pickle.load(open(cache, 'rb'))
	msgs = read_bag(os.path.join(run_dir, 'bag'))
	inputs, phases, t_fault, live, grid = [], [], None, None, None
	for t, topic, msg in msgs:
		if topic == '/rotor_cmd':
			inputs.append((t, 'cmd', np.abs(np.array(msg.velocity[:4]))))
		elif topic == '/rotor_joint_states':
			speeds = dict(zip(msg.name, msg.velocity))
			inputs.append((t, 'joints', np.abs([speeds[f'rotor_{i}_joint'] for i in range(4)]) * SLOWDOWN))
		elif topic == '/fmu/out/vehicle_odometry':
			inputs.append((t, 'odom', (np.array(msg.angular_velocity), msg.timestamp_sample * 1e-6)))
		elif topic == '/motor_fault' and t_fault is None:
			t_fault = t
		elif topic == '/bench/phase':
			phases.append(t)
		elif topic == '/rotor_failure/detected':
			grid = grid if grid is not None else stamp(msg)
			if msg.failed_motor and t_fault is not None and live is None:
				live = stamp(msg)  # sim time of the live decision
	t0 = msgs[0][0]
	grid = t0 if grid is None else grid - np.ceil((grid - t0) / TICK) * TICK
	run = {'inputs': compact(inputs, grid), 't_fault': t_fault, 'live': live, 'grid': grid,
	       'phase_start': phases[0] if phases else None, 'phase_end': phases[-1] if phases else None}
	pickle.dump(run, open(cache, 'wb'))
	return run


def replay(run_dir, **detector_kwargs):
	"""Score one run. Fault runs: first detection after the fault. Healthy runs: every raw detection."""
	info = json.load(open(os.path.join(run_dir, 'run.json')))
	run = load(run_dir)
	t_fault, live, start = run['t_fault'], run['live'], run['phase_start']
	det = Detector(**detector_kwargs)
	detect = early = None           # early = a verdict declared before the fault was injected
	alarms = []                     # raw detection times inside the flown phases (healthy runs)
	next_tick = run['grid']         # same 20 ms grid as the live node, whose messages are stamped on its ticks

	for t, kind, value in run['inputs']:
		while next_tick <= t:
			raw_before = det.raw_detections
			declared = det.check()
			if det.raw_detections > raw_before and start is not None and start <= next_tick <= run['phase_end']:
				alarms.append(next_tick - start)
			if declared:
				if t_fault is None or next_tick < t_fault:
					early = early or (next_tick, det.failed, det.cause)
				elif detect is None:
					detect = (next_tick, det.failed, det.cause, det.severity)
			next_tick += TICK
		if kind == 'cmd':
			det.on_cmd(value)
		elif kind == 'joints':
			det.on_joints(value)
		else:
			det.on_odom(*value)

	res = dict(info)
	if info['split'] == 'healthy':
		res.update(minutes=(run['phase_end'] - start) / 60, alarms=alarms)
		return res

	truth_cause = {'stop': 'motor_stop', 'weak': 'weak_motor', 'prop': 'prop_loss'}[info['mode']]
	if info['mode'] == 'weak' and info['intensity'] >= 0.98:
		truth_cause = 'motor_stop'  # a 100 % weak motor has zero speed: physically a stopped motor
	res.update(truth_cause=truth_cause, detected=detect is not None, early=early is not None,
	           live_latency_ms=None if live is None or t_fault is None else (live - t_fault) * 1000)
	if detect:
		td, motor, cause, severity = detect
		res.update(latency_ms=(td - t_fault) * 1000, det_motor=motor, det_cause=cause, severity=severity,
		           motor_ok=motor == info['motor'], cause_ok=cause == truth_cause)
	return res


def cost(res, miss=1000.0):
	"""Per-run cost for tuning and the arbiter: latency if the right motor AND cause were found, else a miss penalty."""
	ok = res.get('detected') and res.get('motor_ok') and res.get('cause_ok') and not res.get('early')
	return res['latency_ms'] if ok else miss


if __name__ == '__main__':
	for d in sys.argv[1:]:
		r = replay(d)
		keep = ('id', 'phase', 'motor', 'mode', 'intensity', 'detected', 'det_motor', 'det_cause', 'severity',
		        'latency_ms', 'live_latency_ms', 'early', 'minutes', 'alarms')
		print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in keep})
