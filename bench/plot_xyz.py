#!/usr/bin/python3
"""Position error vs time after a complete rotor loss, many runs overlaid (paper-style figure).

  python3 plot_xyz.py [TAG]      default TAG i025a -> results/xyz_<TAG>.png

Blue: policy on Gazebo ground-truth state (8 hover runs + the 40-run phase set).
Black: policy on our onboard estimator (state gyro): 8 hover runs + the phase set (TAG_gyro).
Ground-truth position minus the hold point (the position at handover), from the fault to fault + 15 s.
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

HERE = os.path.dirname(os.path.abspath(__file__))


def track(run_dir):
	u = ULog(glob.glob(os.path.join(run_dir, '*.ulg'))[0], ['vehicle_command', 'vehicle_local_position_groundtruth'])
	d = {x.name: x.data for x in u.data_list}
	vc = d['vehicle_command']
	ev = {int(c): p5 for c, p5 in zip(vc['command'], vc['param5']) if c in (31010, 31012)}
	lp = d['vehicle_local_position_groundtruth']
	t = lp['timestamp'] * 1e-6 - ev[31010]
	i0 = np.searchsorted(t, ev[31012] - ev[31010])
	m = (t >= 0.0) & (t <= 15.0)
	# NED -> x north, y east, z up; relative to the hold point
	return t[m], lp['x'][m] - lp['x'][i0], lp['y'][m] - lp['y'][i0], -(lp['z'][m] - lp['z'][i0])


def main():
	tag = sys.argv[1] if len(sys.argv) > 1 else 'i025a'
	runs = lambda pat: [d for d in sorted(glob.glob(os.path.join(HERE, 'runs', pat))) if glob.glob(os.path.join(d, '*.ulg'))]  # noqa: E731
	gt = runs(f'ftc_[0-9][0-9]_gt_{tag}') + runs(f'data_[0-9][0-9]_{tag}')
	own = runs(f'ftc_[0-9][0-9]_gyro_{tag}') + runs(f'data_[0-9][0-9]_{tag}_gyro')
	fig, axes = plt.subplots(3, 1, figsize=(4.2, 7.6), sharex=True)
	for dirs, color, alpha, lw in ((gt, '#1f77b4', 0.25, 1.0), (own, 'black', 0.6, 1.0)):
		for d in dirs:
			t, *xyz = track(d)
			for ax, v in zip(axes, xyz):
				ax.plot(t, v, color=color, alpha=alpha, lw=lw)
	for ax, label, lim in zip(axes, ('x (m)', 'y (m)', 'z (m)'), (3.2, 3.2, 1.4)):
		ax.axhline(0.0, color='red', ls='--', lw=2)
		ax.set_ylabel(label)
		ax.set_ylim(-lim, lim)
		ax.grid(True)
	axes[0].set_title(f'policy {tag}, complete rotor loss')
	axes[0].plot([], [], color='#1f77b4', label=f'ground-truth state ({len(gt)})')
	axes[0].plot([], [], color='black', label=f'onboard estimator ({len(own)})')
	axes[0].legend(fontsize=7, loc='upper right')
	axes[-1].set_xlabel('time after the fault (s)')
	fig.tight_layout()
	out = os.path.join(HERE, 'results', f'xyz_{tag}.png')
	fig.savefig(out, dpi=150)
	print(out, json.dumps({'gt': len(gt), 'onboard': len(own)}))


if __name__ == '__main__':
	main()
