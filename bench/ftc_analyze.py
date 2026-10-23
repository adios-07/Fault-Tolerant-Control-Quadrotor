#!/usr/bin/python3
"""Fault-tolerant flight results: the learned policy taking over on detection (run_bench.py ftc).

  python3 ftc_analyze.py [TAG]       TAG = RUN_TAG of the batch (e.g. i022c); none = the untagged runs

Reads runs/ftc_NN_<state>[_TAG] and runs/data_NN[_TAG], scores every run against Gazebo ground truth (logged by PX4
SITL) from the handover to the end of the flight (fault + 15 s). For runs on EKF2 state also the EKF2 attitude error
vs truth after the fault. Writes results/ftc_metrics[_TAG].md and results/ftc_trajectories[_TAG].png.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyulog import ULog  # noqa: E402

from estimation import ekf_error, load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
CRASH_ALT = 0.3          # [m] ground-truth height above ground below which the drone has crashed
SCORE_WINDOW = 5.0       # [s] position error is scored over the last 5 s of the flight
EKF_TIMES = (0.5, 1.0, 1.5, 2.5)   # [s] after the fault: EKF2 attitude error reported
MOTOR_COLOR = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a', 4: '#eda100'}   # validated categorical palette
INK, INK2, SURFACE, GRIDC = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'


def flown(run_dir):
	"""A finished flight with its .ulg (a timeout means it never took off or never ended normally: fly it again)."""
	info = os.path.join(run_dir, 'run.json')
	return os.path.exists(info) and json.load(open(info)).get('flyer_exit') == 0 and bool(glob.glob(os.path.join(run_dir, '*.ulg')))


def score(run_dir):
	info = json.load(open(os.path.join(run_dir, 'run.json')))
	u = ULog(glob.glob(os.path.join(run_dir, '*.ulg'))[0], ['vehicle_command', 'vehicle_local_position_groundtruth',
	                                                        'vehicle_attitude_groundtruth', 'vehicle_angular_velocity'])
	d = {x.name: x.data for x in u.data_list}
	vc = d['vehicle_command']
	ev = {int(c): p5 for c, p5 in zip(vc['command'], vc['param5']) if c in (31010, 31011, 31012)}
	t_fault, t_hand = ev[31010], ev.get(31012)
	lp, att, av = d['vehicle_local_position_groundtruth'], d['vehicle_attitude_groundtruth'], d['vehicle_angular_velocity']
	tl, ta, tw = lp['timestamp'] * 1e-6, att['timestamp'] * 1e-6, av['timestamp'] * 1e-6
	t_end = t_fault + 15.0
	ground = np.median(-lp['z'][tl < tl[0] + 2.0])          # height of the drone standing on the ground

	i0 = np.searchsorted(tl, t_hand if t_hand else t_fault)
	m = (tl >= tl[i0]) & (tl <= t_end)
	t = tl[m] - t_fault
	alt = -lp['z'][m] - ground
	err = np.hypot(lp['x'][m] - lp['x'][i0], lp['y'][m] - lp['y'][i0])
	ma = (ta >= tl[i0]) & (ta <= t_end)
	tilt = np.degrees(np.arccos(np.clip(1 - 2 * (att['q[1]'][ma] ** 2 + att['q[2]'][ma] ** 2), -1, 1)))
	# ground contact only: the policy can pass 90 deg tilt for a moment and recover (climb / forward / yaw faults)
	crash_t = t[np.argmax(alt < CRASH_ALT)] if (alt < CRASH_ALT).any() else None
	last = t >= t[-1] - SCORE_WINDOW
	mw = (tw >= tl[i0]) & (tw <= t_end)
	xy_last = np.column_stack([lp['x'][m][last], lp['y'][m][last]])
	ekf = None
	if info.get('state') == 'ekf':
		dd, _, _ = load(run_dir, ['vehicle_attitude', 'vehicle_attitude_groundtruth'])
		tilt_e, yaw_e = ekf_error(dd, t_fault + np.array(EKF_TIMES))
		ekf = [(a, b) for a, b in zip(tilt_e, yaw_e)]
	est = None
	if os.path.exists(os.path.join(run_dir, 'estimate_error.csv')):   # onboard estimator vs Gazebo truth (debug_csv)
		e = np.loadtxt(os.path.join(run_dir, 'estimate_error.csv'), delimiter=',', ndmin=2)
		te, h = e[:, 0] - e[0, 0], np.degrees(np.unwrap(np.radians(e[:, 2])))
		early, late = h[(te >= 3) & (te < 5)], h[te >= te[-1] - 2]
		est = (float(e[:, 1].max()), float(early.mean()), float(late.mean() - early.mean())) if len(early) else None
	return {**info, 'est': est,
	        'tilt_peak_t': float((ta[ma] - t_fault)[np.argmax(tilt)]), 'ekf_err': ekf,
	        'orbit': float(np.sqrt(np.mean(np.sum((xy_last - xy_last.mean(0)) ** 2, axis=1)))) if crash_t is None else None, 'detect_ms': (ev[31011] - t_fault) * 1000 if 31011 in ev else None,
	        'handover_ms': (t_hand - t_fault) * 1000 if t_hand else None,
	        'survived': crash_t is None, 'crash_s': crash_t,
	        'err_last5': float(np.sqrt(np.mean(err[last] ** 2))) if crash_t is None else None,
	        'err_max': float(err.max()), 'alt_loss': float(alt[0] - alt.min()), 'tilt_max': float(tilt.max()),
	        'spin': float(np.mean(np.abs(av['xyz[2]'][mw]))), 't': t, 'err': err, 'alt': alt}


def landing(run_dir):
	"""Touchdown metrics after the descent marker (31013), from ground truth. None if the run never descended."""
	u = ULog(glob.glob(os.path.join(run_dir, '*.ulg'))[0], ['vehicle_command', 'vehicle_local_position_groundtruth',
	                                                        'vehicle_attitude_groundtruth', 'vehicle_angular_velocity'])
	d = {x.name: x.data for x in u.data_list}
	vc = d['vehicle_command']
	ev = {int(c): p5 for c, p5 in zip(vc['command'], vc['param5']) if c in (31012, 31013, 31014)}
	if 31013 not in ev:
		return None
	lp, att, av = d['vehicle_local_position_groundtruth'], d['vehicle_attitude_groundtruth'], d['vehicle_angular_velocity']
	tl = lp['timestamp'] * 1e-6
	ground = np.median(lp['z'][tl < tl[0] + 2.0])              # NED z of the drone resting on the ground
	after = np.flatnonzero((tl > ev[31013]) & (lp['z'] > ground - 0.03))
	if not len(after):
		return {'touchdown': False, 'cut': 31014 in ev}
	k = after[0]
	td = tl[k]
	pre = (tl >= td - 0.2) & (tl < td)
	ta = att['timestamp'] * 1e-6
	tilt = np.degrees(np.arccos(np.clip(1 - 2 * (att['q[1]'] ** 2 + att['q[2]'] ** 2), -1, 1)))
	tw = av['timestamp'] * 1e-6
	spin = np.abs(av['xyz[2]'])
	slow = tw[(tw > td) & (spin < 1.0)]
	ih = np.searchsorted(tl, ev[31012])
	return {'touchdown': True, 'v_td': float(np.mean(lp['vz'][pre])), 'v_lat': float(np.mean(np.hypot(lp['vx'][pre], lp['vy'][pre]))),
	        'tilt_td': float(np.interp(td, ta, tilt)), 'tilt_after': float(tilt[ta > td].max()),
	        'tilt_final': float(np.mean(tilt[ta > ta[-1] - 1.0])), 't_rest': float(slow[0] - td) if len(slow) else None,
	        'cut_delay': float(ev[31014] - td) if 31014 in ev else None,
	        'miss': float(np.hypot(lp['x'][-1] - lp['x'][ih], lp['y'][-1] - lp['y'][ih])), 'hold_s': float(ev[31013] - ev[31012])}


def landing_table(dirs):
	head = ['run', 'motor', 'fault', 'descent at [s after handover]', 'touchdown v down [m/s]', 'lateral v [m/s]',
	        'tilt at contact [deg]', 'max tilt after [deg]', 'final tilt [deg]', 'spin < 1 rad/s after [s]',
	        'motors cut after contact [s]', 'distance from hold point [m]', 'pass']
	lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
	passed = 0
	for dr in dirs:
		info, r = json.load(open(os.path.join(dr, 'run.json'))), landing(dr)
		if r is None or not r['touchdown']:
			lines.append(f"| {info['id']} | {info['motor']} | {info['mode']} | " + ('no descent' if r is None else 'no touchdown') + ' |' * (len(head) - 3))
			continue
		ok = (r['v_td'] <= 0.5 and r['tilt_after'] < 60 and r['tilt_final'] < 20 and r['cut_delay'] is not None
		      and abs(r['cut_delay']) <= 1.0 and r['miss'] <= 0.5)
		passed += ok
		f = lambda v, s='{:.2f}': '-' if v is None else s.format(v)  # noqa: E731
		lines.append(f"| {info['id']} | {info['motor']} | {info['mode']} | {r['hold_s']:.1f} | {r['v_td']:.2f} | {r['v_lat']:.2f} | "
		             f"{r['tilt_td']:.0f} | {r['tilt_after']:.0f} | {r['tilt_final']:.0f} | {f(r['t_rest'])} | {f(r['cut_delay'])} | "
		             f"{r['miss']:.2f} | {'yes' if ok else '**no**'} |")
	return f'Landed (all gates) {passed}/{len(dirs)}', '\n'.join(lines)


def table(rows):
	ekf = any(r['ekf_err'] for r in rows)
	est = any(r['est'] for r in rows)
	head = ['run', 'phase', 'motor', 'fault', 'detect [ms]', 'handover [ms]', 'survived', 'crash at [s]',
	        'peak tilt [deg] at [s]', 'altitude loss [m]', 'xy RMS last 5 s [m]', 'orbit radius [m]', 'max xy error [m]',
	        'mean spin [rad/s]'] + ([f'EKF tilt / yaw err +{t:g} s [deg]' for t in EKF_TIMES] if ekf else []) \
	       + (['estimator max tilt err [deg]', 'estimator heading err 3-5 s [deg]', 'heading drift to end [deg]'] if est else [])
	fmt = lambda v, f='{:.2f}': '-' if v is None else f.format(v)  # noqa: E731
	lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
	for r in rows:
		lines.append('| ' + ' | '.join([r['id'], r['phase'], str(r['motor']), r['mode'], fmt(r['detect_ms'], '{:.0f}'),
		                                fmt(r['handover_ms'], '{:.0f}'), 'yes' if r['survived'] else '**no**',
		                                fmt(r['crash_s'], '{:.1f}'), f"{r['tilt_max']:.0f} at {r['tilt_peak_t']:.2f}",
		                                fmt(r['alt_loss']), fmt(r['err_last5']), fmt(r['orbit']), fmt(r['err_max']),
		                                fmt(r['spin'], '{:.1f}')]
		                               + ([f'{a:.1f} / {b:.1f}' for a, b in r['ekf_err']] if ekf else [])
		                               + ([f"{r['est'][0]:.1f}", f"{r['est'][1]:.1f}", f"{r['est'][2]:+.1f}"] if est and r['est'] else
		                                  ['-', '-', '-'] if est else [])) + ' |')
	ok = [r for r in rows if r['survived']]
	summary = (f"Survived {len(ok)}/{len(rows)}; peak tilt max {max(r['tilt_max'] for r in rows):.0f} deg, "
	           f"{sum(r['tilt_max'] > 90 for r in rows)} runs above 90 deg"
	           + (f"; xy RMS last 5 s mean {np.mean([r['err_last5'] for r in ok]):.2f} m, orbit radius mean "
	              f"{np.mean([r['orbit'] for r in ok]):.2f} m" if ok else ''))
	return summary, '\n'.join(lines)


def plot(groups, path):
	fig, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 6.4), sharex=True, squeeze=False,
	                         facecolor=SURFACE)
	for col, (name, rows) in enumerate(groups.items()):
		for row, (key, label) in enumerate((('alt', 'height above ground [m]'), ('err', 'xy distance from hold point [m]'))):
			ax = axes[row, col]
			ax.set_facecolor(SURFACE)
			ax.grid(color=GRIDC, linewidth=0.8)
			for s in ('top', 'right'):
				ax.spines[s].set_visible(False)
			for r in rows:
				ax.plot(r['t'], r[key], color=MOTOR_COLOR[r['motor']], linewidth=2,
				        linestyle='-' if r['mode'] == 'stop' else '--',
				        label=f"motor {r['motor']} {r['mode']}" if row == 0 else None)
			ax.set_ylabel(label, color=INK)
			ax.tick_params(colors=INK2)
			if row == 0:
				ax.set_title(f'policy state from {name}', color=INK, fontsize=11, loc='left')
				ax.axhline(CRASH_ALT, color=INK2, linewidth=1, linestyle=':')
			else:
				ax.set_xlabel('time after the fault [s]', color=INK)
	axes[0, 0].legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK)
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(fig)


def main():
	os.makedirs(OUT, exist_ok=True)
	tag = sys.argv[1] if len(sys.argv) > 1 else ''
	sfx = f'_{tag}' if tag else ''
	groups = {}
	md = [f'# Fault-tolerant flight: learned policy after detection{f" ({tag})" if tag else ""}', '',
	      'Complete loss of one rotor (`stop`: motor dead; `prop`: propeller gone, motor spinning) in hover. PX4 flies until '
	      'the detector declares the failure, then the policy takes the motors (offboard direct actuator). Scored against '
	      f'Gazebo ground truth from the handover to fault + 15 s. Crash = height above ground < {CRASH_ALT} m.', '']
	if not any(glob.glob(os.path.join(HERE, 'runs', '*', '*.ulg'))):
		md += ['> **No .ulg logs found in `runs/`.** Per-run logs were pruned for the submission (3.1 GB); the '
		       'curated logs are not shipped, and the committed tables in `bench/results/` are the record. '
		       'Re-fly the batch to regenerate this file:', '',
		       '```bash',
		       'export RTF=2 FTC_DEBUG_CSV=1',
		       'RUN_TAG=i025a python3 run_bench.py ftc',
		       'RUN_TAG=i025a_gyro python3 run_bench.py data',
		       'RUN_TAG=i025a python3 run_bench.py partial',
		       'FTC_TAIL=35 RUN_TAG=land python3 run_bench.py ftc',
		       'python3 ftc_analyze.py i025a; python3 ftc_analyze.py i025a_gyro',
		       '```', '']
	for name, label in (('gt', 'Gazebo ground truth (ideal state)'), ('ekf', 'PX4 EKF2 (onboard estimate)'),
	                    ('ekf_gt', 'EKF2 attitude + true position/velocity (IDEA-022 control)'),
	                    ('gyro_gt', 'gyro attitude + true position/velocity (IDEA-022 A)'),
	                    ('gyro', 'gyro attitude + own KF, fully onboard (IDEA-022 B)')):
		dirs = [d for d in sorted(glob.glob(os.path.join(HERE, 'runs', f'ftc_[0-9][0-9]_{name}{sfx}'))) if flown(d)]
		if not dirs:
			continue
		rows = [score(d) for d in dirs]
		groups[label] = rows
		summary, tab = table(rows)
		md += [f'## Policy state: {label}', '', summary + '.', '', tab, '']
	data = [d for d in sorted(glob.glob(os.path.join(HERE, 'runs', f'data_[0-9][0-9]{sfx}'))) if flown(d)]
	if data:
		rows = [score(d) for d in data]
		md += [f"## Phase set (data runs): policy state {rows[0]['state']}, fault in every flight phase", '']
		for phase in ('hover', 'forward', 'climb', 'descent', 'yaw'):
			if any(r['phase'] == phase for r in rows):
				md.append(f"- {phase}: " + table([r for r in rows if r['phase'] == phase])[0])
		md += ['', table(rows)[1], '']
	part = [score(d) for d in sorted(glob.glob(os.path.join(HERE, 'runs', f'partial_[0-9][0-9]{sfx}'))) if flown(d)]
	if part:
		md += ['## Partial loss (thrust left 30-80 %, weak mode), onboard estimator, per motor', '',
		       'Cell: xy RMS last 5 s [m] / peak tilt [deg]; CRASH t = ground contact at t s after the fault; '
		       '"no det" = the detector never fired, PX4 kept flying.', '']
		for left in (0.3, 0.5, 0.65, 0.8):
			rows = [r for r in part if abs(1 - r['intensity'] - left) < 1e-6]
			if not rows:
				continue
			md += [f'### thrust left {left * 100:.0f} %: ' + table(rows)[0], '',
			       '| phase | motor 1 | motor 2 | motor 3 | motor 4 |', '|---|---|---|---|---|']
			for phase in ('hover', 'forward', 'climb', 'descent', 'yaw'):
				cells = []
				for m in (1, 2, 3, 4):
					r = next((r for r in rows if r['phase'] == phase and r['motor'] == m), None)
					cells.append('-' if r is None else
					             f"CRASH {r['crash_s']:.1f}" if not r['survived'] else
					             f"{r['err_last5']:.2f} / {r['tilt_max']:.0f}" + ('' if r['detect_ms'] else ' (no det)'))
				md.append(f'| {phase} | ' + ' | '.join(cells) + ' |')
			md.append('')
	landed = [d for name in ('gt', 'gyro') for d in sorted(glob.glob(os.path.join(HERE, 'runs', f'ftc_[0-9][0-9]_{name}{sfx}')))
	          if flown(d)]
	if any(landing(d) for d in landed):
		summary, tab = landing_table(landed)
		md += ['## Landing after the hold (hold 10 s once settled, descend at 0.4 m/s, motors cut on touchdown)', '',
		       'Pass = touchdown <= 0.5 m/s, never past 60 deg tilt, final tilt < 20 deg, motors cut within 1 s of '
		       'contact, within 0.5 m of the hold point.', '', summary + '.', '', tab, '']
	if groups:
		plot(groups, os.path.join(OUT, f'ftc_trajectories{sfx}.png'))
	open(os.path.join(OUT, f'ftc_metrics{sfx}.md'), 'w').write('\n'.join(md) + '\n')
	print('\n'.join(md))


if __name__ == '__main__':
	main()
