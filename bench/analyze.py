#!/usr/bin/python3
"""Benchmark analysis: metrics, plots and the one pre-registered tuning (IMU_HOLD x SMOOTH).

  python3 analyze.py

Reads runs/ (written by run_bench.py), replays every run offline (replay.py) and writes results/:
  metrics.md               all tables (test split, baseline vs tuned) + replay validation
  latency_vs_phase.png     detection latency per flight phase, per fault mode
  confusion_motor.png      isolation confusion matrix (true motor vs detected motor / missed)
  confusion_cause.png      root-cause confusion matrix
  base.json, tuned.json    per-run cost on the test split, scored by the council arbiter

Tuning is chosen on the TUNE split only and reported once on the TEST split.
"""
import glob
import itertools
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402

from replay import replay, cost  # noqa: E402  (replay.py also puts ../src on the path)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
GRID = {'imu_hold': [3, 4, 5, 8], 'smooth': [0.2, 0.3, 0.5, 0.7]}
BASE = {'imu_hold': 8, 'smooth': 0.3}   # detector values before tuning (the 'before' in before/after)
PHASES = ['hover', 'forward', 'climb/descent', 'yaw']
MODES = ['stop', 'weak', 'prop']
CAUSES = ['motor_stop', 'weak_motor', 'prop_loss']
# Validated categorical palette (dataviz reference, light): blue, orange, aqua; shapes as second encoding
MODE_STYLE = {'stop': ('#2a78d6', 'o'), 'weak': ('#eb6834', 's'), 'prop': ('#1baf7a', '^')}
INK, INK2, SURFACE, GRIDC = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
BLUES = ListedColormap(['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def group(phase):
	return 'climb/descent' if phase in ('climb', 'descent') else phase


def score(dirs, params):
	return [replay(d, **params) for d in dirs]


def false_alarms(healthy, half):
	"""Raw detections in the first ('tune') or second ('report') half of the healthy session, and its minutes."""
	mid = healthy['minutes'] * 60 / 2
	n = sum(1 for a in healthy['alarms'] if (a < mid) == (half == 'tune'))
	return n, healthy['minutes'] / 2


def metrics(res):
	inj = len(res)
	det = [r for r in res if r['detected'] and not r['early']]
	lat = [r['latency_ms'] for r in det]
	pct = lambda a, b: f'{100 * a / b:.0f} % ({a}/{b})' if b else 'n/a'  # noqa: E731
	return {
		'detection rate': pct(len(det), inj),
		'false verdict before the fault': sum(r['early'] for r in res),
		'latency mean / worst [ms]': f'{np.mean(lat):.0f} / {np.max(lat):.0f}' if lat else 'n/a',
		'isolation accuracy': pct(sum(r['motor_ok'] for r in det), len(det)),
		'root-cause accuracy': pct(sum(r['cause_ok'] for r in det), len(det)),
	}


def ulg_markers(path):
	"""Injection (31010) and detection (31011) markers in a .ulg: {command: (arrival time [s], exact event time [s])}."""
	from pyulog import ULog
	vc = ULog(path, ['vehicle_command']).data_list[0].data
	return {int(c): (t * 1e-6, p5) for t, c, p5 in zip(vc['timestamp'], vc['command'], vc['param5']) if c in (31010, 31011)}


def table(rows, header):
	lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
	lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
	return '\n'.join(lines)


def style(ax):
	ax.set_facecolor(SURFACE)
	for s in ('top', 'right'):
		ax.spines[s].set_visible(False)
	for s in ('left', 'bottom'):
		ax.spines[s].set_color(INK2)
	ax.tick_params(colors=INK2)


def plot_latency(results, path):
	fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2), sharey=True, facecolor=SURFACE)
	rng = np.random.default_rng(0)
	for ax, (title, res) in zip(np.atleast_1d(axes), results.items()):
		style(ax)
		ax.grid(axis='y', color=GRIDC, linewidth=0.8)
		ax.set_axisbelow(True)
		for i, phase in enumerate(PHASES):
			for j, mode in enumerate(MODES):
				pts = [r['latency_ms'] for r in res if group(r['phase']) == phase and r['mode'] == mode
				       and r['detected'] and not r['early'] and r['motor_ok']]
				color, marker = MODE_STYLE[mode]
				x = i + (j - 1) * 0.22 + rng.uniform(-0.05, 0.05, len(pts))
				ax.scatter(x, pts, s=46, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5,
				           label=mode if i == 0 else None, zorder=3)
			missed = sum(1 for r in res if group(r['phase']) == phase and not (r['detected'] and r['motor_ok'] and not r['early']))
			if missed:
				ax.text(i, -0.08, f'{missed} missed / wrong', transform=ax.get_xaxis_transform(), ha='center',
				        va='top', color=INK2, fontsize=8)
		ax.set_xticks(range(len(PHASES)), PHASES, color=INK)
		ax.set_title(title, color=INK, fontsize=11, loc='left')
	np.atleast_1d(axes)[0].set_ylabel('detection latency [ms]', color=INK)
	np.atleast_1d(axes)[0].legend(title='fault mode', frameon=False, labelcolor=INK)
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(fig)


def plot_confusion(results, path, rows, cols, row_key, col_key, xlabel, ylabel):
	fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.8), facecolor=SURFACE)
	for ax, (title, res) in zip(np.atleast_1d(axes), results.items()):
		m = np.zeros((len(rows), len(cols)), int)
		for r in res:
			ok = r['detected'] and not r['early']
			m[rows.index(row_key(r)), cols.index(col_key(r) if ok else 'missed')] += 1
		ax.imshow(m, cmap=BLUES, vmin=0, vmax=max(m.max(), 1))
		for (i, j), v in np.ndenumerate(m):
			ax.text(j, i, v, ha='center', va='center', color=SURFACE if v > m.max() / 2 else INK, fontsize=10)
		ax.set_xticks(range(len(cols)), cols, color=INK, fontsize=8)
		ax.set_yticks(range(len(rows)), rows, color=INK, fontsize=8)
		ax.set_xlabel(xlabel, color=INK)
		ax.set_ylabel(ylabel, color=INK)
		ax.set_title(title, color=INK, fontsize=11, loc='left')
		for s in ax.spines.values():
			s.set_visible(False)
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor=SURFACE)
	plt.close(fig)


def main():
	os.makedirs(OUT, exist_ok=True)
	runs = lambda split: sorted(glob.glob(os.path.join(HERE, 'runs', f'{split}_*')))  # noqa: E731
	tune_dirs, test_dirs = runs('tune'), runs('test')
	healthy_dir = os.path.join(HERE, 'runs', 'healthy')

	# Tuning on the TUNE split: minimum mean cost, healthy tune-half false alarms must not exceed the baseline's
	base_fa, _ = false_alarms(replay(healthy_dir, **BASE), 'tune')
	grid_rows, best = [], (None, BASE)
	for hold, smooth in itertools.product(GRID['imu_hold'], GRID['smooth']):
		p = {'imu_hold': hold, 'smooth': smooth}
		c = np.mean([cost(r) for r in score(tune_dirs, p)])
		fa, _ = false_alarms(replay(healthy_dir, **p), 'tune')
		ok = fa <= base_fa
		grid_rows.append([hold, smooth, f'{c:.1f}', fa, 'yes' if ok else 'no'])
		if ok and (best[0] is None or c < best[0]):
			best = (c, p)
	tuned = best[1]

	# Report once on the TEST split
	res = {'baseline': score(test_dirs, BASE), 'tuned': score(test_dirs, tuned)}
	healthy = {k: replay(healthy_dir, **p) for k, p in (('baseline', BASE), ('tuned', tuned))}
	for name, rr in res.items():
		json.dump({r['id']: cost(r) for r in rr}, open(os.path.join(OUT, f"{'base' if name == 'baseline' else 'tuned'}.json"), 'w'), indent=1)
	arbiter = subprocess.run(['python3', os.path.expanduser('~/.claude/skills/council/arbiter.py'),
	                          os.path.join(OUT, 'base.json'), os.path.join(OUT, 'tuned.json'), '--lower-is-better'],
	                         capture_output=True, text=True).stdout.strip()

	# Replay validation: offline decision time vs the live node's decision time, same runs
	pairs = [(r['id'], r['latency_ms'], r['live_latency_ms']) for d in tune_dirs + test_dirs
	         for r in [replay(d, **BASE)] if r.get('latency_ms') is not None and r.get('live_latency_ms') is not None]
	diffs = np.array([live - off for _, off, live in pairs]) if pairs else np.array([0.0])

	plot_latency({f'test split, baseline (IMU_HOLD {BASE["imu_hold"]}, SMOOTH {BASE["smooth"]})': res['baseline'],
	              f'test split, tuned (IMU_HOLD {tuned["imu_hold"]}, SMOOTH {tuned["smooth"]})': res['tuned']},
	             os.path.join(OUT, 'latency_vs_phase.png'))
	plot_confusion(res, os.path.join(OUT, 'confusion_motor.png'), [1, 2, 3, 4], [1, 2, 3, 4, 'missed'],
	               lambda r: r['motor'], lambda r: r['det_motor'], 'detected motor', 'injected motor')
	plot_confusion(res, os.path.join(OUT, 'confusion_cause.png'), CAUSES, CAUSES + ['missed'],
	               lambda r: r['truth_cause'], lambda r: r['det_cause'], 'detected cause', 'true cause')

	md = ['# Task 2c benchmark results', '',
	      f'Runs: {len(tune_dirs)} tune + {len(test_dirs)} test fault runs, one healthy session of '
	      f'{healthy["baseline"]["minutes"]:.1f} min. Latency = sim time from fault injection to the detector\'s decision.', '']
	md += ['## Test split: baseline vs tuned', '']
	rows = []
	for name in ('baseline', 'tuned'):
		fa, mins = false_alarms(healthy[name], 'report')
		fa_txt = f'{fa / mins:.2f} /min ({fa} in {mins:.1f} min)' if fa else f'0 (< {3 / mins:.2f} /min, rule of three)'
		rows.append([name] + list(metrics(res[name]).values()) + [fa_txt])
	md.append(table(rows, ['detector'] + list(metrics(res['baseline']).keys()) + ['false alarms (healthy, report half)']))
	md += ['', f'Arbiter (per-run cost, test split, lower is better): `{arbiter}`', '']
	for key, title, vals in (('mode', 'Per fault mode (test, baseline)', MODES), ('phase', 'Per flight phase (test, baseline)', PHASES)):
		md += [f'## {title}', '']
		sel = lambda r, v: (group(r['phase']) if key == 'phase' else r['mode']) == v  # noqa: E731
		md.append(table([[v] + list(metrics([r for r in res['baseline'] if sel(r, v)]).values()) for v in vals],
		                [key] + list(metrics(res['baseline']).keys())))
		md.append('')
	md += ['## Tuning grid (tune split)', '', f'Baseline healthy tune-half false alarms: {base_fa}. '
	       f'Chosen: IMU_HOLD {tuned["imu_hold"]}, SMOOTH {tuned["smooth"]}.', '',
	       table(grid_rows, ['IMU_HOLD', 'SMOOTH', 'mean cost [ms]', 'false alarms (tune half)', 'allowed']), '']
	extra_rows = []
	for d in runs('extra'):
		r = replay(d, **tuned)
		# Per-run .ulg files were pruned for the submission (3.1 GB); the curated logs live in
		# submission/. Without one, the two marker columns are reported as '-' rather than crashing.
		ulg = glob.glob(os.path.join(d, '*.ulg'))
		m = ulg_markers(ulg[0]) if ulg else {}
		exact = (m[31011][1] - m[31010][1]) * 1000 if len(m) == 2 else None
		arrival = (m[31011][0] - m[31010][0]) * 1000 if len(m) == 2 else None
		fmt = lambda v: f'{v:.0f}' if v is not None else '-'  # noqa: E731
		extra_rows.append([r['id'], r['phase'], r['mode'], r['intensity'], r['motor'], r.get('det_motor', '-'),
		                   r.get('det_cause', '-'), fmt(r.get('latency_ms')), fmt(r.get('live_latency_ms')), fmt(exact), fmt(arrival)])
	if extra_rows:
		md += ['## Extra runs: 50 % faults, final detector, live', '',
		       'Flown after tuning with the final detector (IMU_HOLD 3, SMOOTH 0.2). ".ulg exact" is param5 of the '
		       'markers (sim time of the event); ".ulg arrival" is when the marker reached PX4. Those two columns '
		       'need the run\'s .ulg; per-run logs were pruned, logs not shipped.', '',
		       table(extra_rows, ['run', 'phase', 'mode', 'intensity', 'motor', 'detected motor', 'detected cause',
		                          'replay [ms]', 'live [ms]', '.ulg exact [ms]', '.ulg arrival [ms]']), '']
	md += ['## Replay validation', '', f'Offline vs live decision time on {len(pairs)} runs: mean bias '
	       f'{diffs.mean():+.1f} ms, worst {np.abs(diffs).max():.1f} ms (accept if mean |bias| < 10 ms).', '']
	md += ['## All test runs (baseline)', '',
	       table([[r['id'], r['phase'], r['mode'], r['intensity'], r['motor'], r.get('det_motor', '-'), r.get('det_cause', '-'),
	               f"{r['latency_ms']:.0f}" if r.get('latency_ms') is not None else '-', 'yes' if r['early'] else ''] for r in res['baseline']],
	             ['run', 'phase', 'mode', 'intensity', 'motor', 'detected motor', 'detected cause', 'latency [ms]', 'early'])]
	open(os.path.join(OUT, 'metrics.md'), 'w').write('\n'.join(md) + '\n')
	print('\n'.join(md[:14]))


if __name__ == '__main__':
	main()
