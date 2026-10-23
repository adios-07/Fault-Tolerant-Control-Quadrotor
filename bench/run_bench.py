#!/usr/bin/python3
"""Run benchmark flights headless: one sim launch per flight, raw detector inputs bagged, .ulg copied.

  python3 run_bench.py schedule            write the seeded fault schedule to schedule.json
  python3 run_bench.py faults [ID ...]     fly the tune/test fault runs (or only the given ids)
  python3 run_bench.py ftc [ID ...]        fly the ftc runs: fault, detection, handover to the learned policy
  python3 run_bench.py data [ID ...]       fly the phase set: complete loss in every flight phase, policy on detection
  python3 run_bench.py partial [ID ...]    thrust left 30-80 % on every motor and phase, policy on detection
  python3 run_bench.py healthy SECONDS     fly one healthy session cycling all phases

Each flight lands in runs/<id>/: inputs.pkl (detector inputs for replay.py, from a temporary sim-time
rosbag), run.json (what was flown), <id>.ulg.
Needs the workspace sourced (source install/setup.bash).
"""
import glob
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time

import replay

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
PX4_LOGS = os.path.expanduser(os.environ.get('PX4_DIR', '~/PX4-Autopilot') + '/build/px4_sitl_default/log')
BOOT_WAIT = 10.0        # [s] Gazebo + PX4 + bridges up; the flyer retries arming until PX4 accepts
RTF = os.environ.get('RTF', '1.0')                    # real-time factor of the sim (> 1 = faster)
SEED = 2026
STATE = os.environ.get('FTC_STATE', 'gyro')           # onboard estimator; 'ekf', 'gt' (diagnosis)
TOPICS = ['/rotor_cmd', '/rotor_joint_states', '/fmu/out/vehicle_odometry', '/motor_fault',
          '/rotor_failure/detected', '/bench/phase', '/clock']
LEFTOVERS = ['bin/px4 -s', 'gz sim -r -s', 'MicroXRCEAgent udp4', 'parameter_bridge', 'failure_detection.py',
             'phase_flyer.py', 'ros2 bag record']


def make_schedule():
	"""4 phase groups x 3 modes x 2 reps per split; motors rotate so each appears equally. Plus extra and ftc runs."""
	rng = random.Random(SEED)
	runs = []
	for split in ('tune', 'test'):
		k = 0
		for group in ('hover', 'forward', 'vertical', 'yaw'):
			for mode in ('stop', 'weak', 'prop'):
				for rep in range(2):
					phase = ('climb', 'descent')[rep] if group == 'vertical' else group
					intensity = 1.0 if mode == 'stop' or rep == 0 else round(rng.uniform(0.5, 1.0), 2)
					runs.append({'id': f'{split}_{k:02d}', 'split': split, 'phase': phase, 'motor': k % 4 + 1,
					             'mode': mode, 'intensity': intensity, 'offset': round(rng.uniform(2.0, 7.0), 3)})
					k += 1
	# Extra: 50 % faults the random draw never reached, flown with the final detector (also the submission .ulg)
	extra = [('hover', 'prop'), ('forward', 'prop'), ('climb', 'prop'), ('yaw', 'prop'), ('hover', 'weak'), ('hover', 'stop')]
	for k, (phase, mode) in enumerate(extra):
		runs.append({'id': f'extra_{k:02d}', 'split': 'extra', 'phase': phase, 'motor': k % 4 + 1, 'mode': mode,
		             'intensity': 1.0 if mode == 'stop' else 0.5, 'offset': round(rng.uniform(2.0, 7.0), 3)})
	# FTC: complete rotor loss in hover, handed to the learned policy on detection (it was trained for full loss only)
	for k, (motor, mode) in enumerate((m, md) for m in range(1, 5) for md in ('stop', 'prop')):
		runs.append({'id': f'ftc_{k:02d}', 'split': 'ftc', 'phase': 'hover', 'motor': motor, 'mode': mode,
		             'intensity': 1.0, 'offset': round(rng.uniform(2.0, 7.0), 3)})
	# Data: the phase set. Complete loss (motors 1-4 x stop / prop) in every flight phase, policy on detection.
	for k, (phase, motor, mode) in enumerate((p, m, md) for p in ('hover', 'forward', 'climb', 'descent', 'yaw')
	                                         for m in range(1, 5) for md in ('stop', 'prop')):
		runs.append({'id': f'data_{k:02d}', 'split': 'data', 'phase': phase, 'motor': motor, 'mode': mode,
		             'intensity': 1.0, 'offset': round(rng.uniform(2.0, 7.0), 3)})
	# Sweep (retired experiment: policy vs PX4 alone on partial loss). Kept so the seeded offsets of later runs stay the same.
	for k, (motor, mode, intensity) in enumerate((m, md, i) for m in (1, 3) for md in ('weak', 'prop')
	                                            for i in (0.5, 0.7, 0.85, 1.0)):
		runs.append({'id': f'sweep_{k:02d}', 'split': 'sweep', 'phase': 'hover', 'motor': motor, 'mode': mode,
		             'intensity': intensity, 'offset': round(rng.uniform(2.0, 7.0), 3)})
	# Partial: rotor thrust scaled to 30 / 50 / 65 / 80 % (weak mode: speed, so thrust and drag torque scale), every
	# motor and phase, policy on detection
	for k, (phase, motor, left) in enumerate((p, m, f) for p in ('hover', 'forward', 'climb', 'descent', 'yaw')
	                                         for m in range(1, 5) for f in (0.3, 0.5, 0.65, 0.8)):
		runs.append({'id': f'partial_{k:02d}', 'split': 'partial', 'phase': phase, 'motor': motor, 'mode': 'weak',
		             'intensity': round(1 - left, 2), 'offset': round(rng.uniform(2.0, 7.0), 3)})
	return runs


def kill_leftovers():
	for pattern in LEFTOVERS:
		subprocess.run(['pkill', '-9', '-f', pattern], stderr=subprocess.DEVNULL)
	time.sleep(2)


def stop(proc):
	"""SIGINT the whole process group, then SIGKILL whatever is left."""
	try:
		os.killpg(proc.pid, signal.SIGINT)
		proc.wait(timeout=10)
	except (ProcessLookupError, subprocess.TimeoutExpired):
		try:
			os.killpg(proc.pid, signal.SIGKILL)
		except ProcessLookupError:
			pass


def fly(run_id, flyer_args, info):
	out = os.path.join(RUNS, run_id)
	shutil.rmtree(out, ignore_errors=True)  # a rerun replaces the old run
	os.makedirs(out)
	log = open(os.path.join(out, 'console.log'), 'w')
	kill_leftovers()
	t_start = time.time()

	sim = subprocess.Popen(['ros2', 'launch', 'rotor_fail', 'bringup.launch.py', 'headless:=true', 'inject:=false', f'rtf:={RTF}',
	                        f"ftc:={'true' if info.get('ftc') else 'false'}", f"state:={STATE}"]
	                       + ([f"debug_csv:={os.path.join(out, 'estimate_error.csv')}"] if os.environ.get('FTC_DEBUG_CSV') else []),
	                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
	time.sleep(BOOT_WAIT)
	record = info.get('split') in ('tune', 'test', 'extra', 'healthy')   # detector replay needs the bag; FTC runs use the .ulg
	if record:
		bag = subprocess.Popen(['ros2', 'bag', 'record', '--use-sim-time', '-o', os.path.join(out, 'bag')] + TOPICS,
		                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		time.sleep(2)
	try:
		flyer = subprocess.run(['python3', os.path.join(HERE, 'phase_flyer.py'), '--ros-args', '-p', 'use_sim_time:=true']
		                       + flyer_args, stdout=log, stderr=subprocess.STDOUT, timeout=info.get('timeout', 150))
		info['flyer_exit'] = flyer.returncode
	except subprocess.TimeoutExpired:
		info['flyer_exit'] = 'timeout'
	if record:
		stop(bag)
	stop(sim)
	kill_leftovers()

	new = [f for f in glob.glob(os.path.join(PX4_LOGS, '*', '*.ulg')) if os.path.getmtime(f) > t_start]
	if new:
		os.replace(max(new, key=os.path.getmtime), os.path.join(out, f'{run_id}.ulg'))
	info['ulg'] = bool(new)
	json.dump(info, open(os.path.join(out, 'run.json'), 'w'), indent=1)
	if record:
		replay.load(out)                               # keep only the compact detector inputs (inputs.pkl)
		shutil.rmtree(os.path.join(out, 'bag'), ignore_errors=True)
	print(f"{run_id}: flyer exit {info['flyer_exit']}, ulg {bool(new)}", flush=True)


def main():
	cmd = sys.argv[1] if len(sys.argv) > 1 else ''
	sched_path = os.path.join(HERE, 'schedule.json')
	if cmd == 'schedule':
		json.dump(make_schedule(), open(sched_path, 'w'), indent=1)
		print(f'wrote {sched_path}')
	elif cmd in ('faults', 'ftc', 'data', 'partial'):
		only = set(sys.argv[2:])
		for r in json.load(open(sched_path)):
			if only and r['id'] not in only:
				continue
			if not only and r['split'] not in {'faults': ('tune', 'test'), 'ftc': ('ftc',), 'data': ('data',), 'partial': ('partial',)}[cmd]:
				continue
			fault = f"{r['motor']} {r['mode']} {r['intensity']}"
			args = ['-p', f"phase:={r['phase']}", '-p', f'fault:={fault}', '-p', f"offset:={r['offset']}"]
			run_id = r['id']
			ftc = r['split'] in ('ftc', 'data', 'partial')
			if ftc:
				args += ['-p', f'ftc:={str(ftc).lower()}', '-p', f"tail:={float(os.environ.get('FTC_TAIL', 15))}"]
				run_id = r['id'] if r['split'] in ('data', 'partial') else f"{r['id']}_{STATE}"
				run_id += f"_{os.environ['RUN_TAG']}" if os.environ.get('RUN_TAG') else ''
			fly(run_id, args, dict(r, state=STATE if ftc else 'px4', ftc=ftc))
	elif cmd == 'healthy':
		seconds = float(sys.argv[2])
		fly('healthy', ['-p', 'phase:=healthy', '-p', f'duration:={seconds}'],
		    {'id': 'healthy', 'split': 'healthy', 'duration': seconds, 'timeout': seconds + 120})
	else:
		print(__doc__)


if __name__ == '__main__':
	main()
