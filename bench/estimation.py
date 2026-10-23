#!/usr/bin/python3
"""State-estimation analysis on recorded runs (council round 4, IDEA-020 / IDEA-023).

  python3 estimation.py aid      which EKF2 aid sources fuse after the fault vs the EKF tilt error (IDEA-020)
  python3 estimation.py gyro     raw-gyro dead reckoning seeded before detection vs EKF2 vs truth (IDEA-023)
  python3 estimation.py kf       IDEA-022 step B offline: gyro attitude + pos_kf.py (IMU + GPS) velocity vs EKF2
  python3 estimation.py heading  heading error with gyro scale error: gyro only, + magnetometer aiding, PX4 EKF-GSF yaw
  python3 estimation.py ekf      EKF2 configurations (runs ftc_0*_ekf[_TAG]): attitude error 0-5 s after the fault + survival (IDEA-021)

Errors are against Gazebo ground truth (vehicle_attitude_groundtruth, slerp-interpolated to the estimate's
timestamps). Tilt error = angle between the true and estimated body z axes; yaw error = the rest of the rotation.
"""
import glob
import json
import os
import sys

import numpy as np
from pyulog import ULog
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
TIMES = (0.5, 1.0, 2.0)                 # [s] after the fault


def load(run_dir, topics):
	u = ULog(glob.glob(os.path.join(run_dir, '*.ulg'))[0], topics + ['vehicle_command'])
	d = {x.name: x.data for x in u.data_list}
	vc = d['vehicle_command']
	ev = {int(c): p5 for c, p5 in zip(vc['command'], vc['param5']) if c in (31010, 31011, 31012)}
	return d, ev, json.load(open(os.path.join(run_dir, 'run.json')))


def quat(data):
	"""PX4 w x y z -> scipy Rotation (body FRD -> world NED)."""
	return Rotation.from_quat(np.column_stack([data['q[1]'], data['q[2]'], data['q[3]'], data['q[0]']]))


def truth_at(d, t):
	g = d['vehicle_attitude_groundtruth']
	tg = g['timestamp'] * 1e-6
	keep = np.concatenate([[True], np.diff(tg) > 0])
	t = np.clip(t, tg[keep][0], tg[keep][-1])
	return Slerp(tg[keep], quat(g)[keep])(t)


def errors(est, true):
	"""Tilt error (angle between body z axes) and yaw error (remaining rotation angle) [deg]."""
	z_e = est.apply([0, 0, 1])
	z_t = true.apply([0, 0, 1])
	tilt = np.degrees(np.arccos(np.clip(np.sum(z_e * z_t, axis=1), -1, 1)))
	total = np.degrees((true.inv() * est).magnitude())
	return tilt, np.sqrt(np.maximum(total ** 2 - tilt ** 2, 0))


def ekf_error(d, times):
	a = d['vehicle_attitude']
	ta = a['timestamp'] * 1e-6
	idx = np.searchsorted(ta, times).clip(0, len(ta) - 1)
	return errors(quat(a)[idx], truth_at(d, ta[idx]))


def frac(d, topic, t0, t1, key='fused'):
	a = d[topic]
	t = a['timestamp'] * 1e-6
	m = (t >= t0) & (t < t1)
	return float(np.mean(a[key][m])) if m.any() else float('nan')


def aid_table():
	"""IDEA-020: per EKF run, fraction of samples each aid source fused in each window after the fault."""
	topics = ['vehicle_attitude', 'vehicle_attitude_groundtruth', 'estimator_aid_src_mag', 'estimator_aid_src_gravity',
	          'estimator_aid_src_gnss_vel', 'estimator_aid_src_gnss_pos', 'estimator_aid_src_baro_hgt', 'estimator_status_flags']
	windows = ((-1.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0))
	lines = ['| run | fault | window [s] | mag fused | gravity fused | GNSS vel fused | GNSS pos fused | baro fused | '
	         'mag 3d on | mag hdg on | EKF tilt err end [deg] | EKF yaw err end [deg] |', '|' + '---|' * 12]
	for run in sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0*_ekf'))):
		d, ev, info = load(run, topics)
		tf = ev[31010]
		sf = d['estimator_status_flags']
		ts = sf['timestamp'] * 1e-6
		for w0, w1 in windows:
			m = (ts >= tf + w0) & (ts < tf + w1)
			tilt, yaw = ekf_error(d, np.array([tf + w1]))
			lines.append('| ' + ' | '.join([
				os.path.basename(run), f"m{info['motor']} {info['mode']}", f'{w0:+.1f}..{w1:+.1f}',
				*[f"{frac(d, f'estimator_aid_src_{s}', tf + w0, tf + w1):.2f}" for s in ('mag', 'gravity', 'gnss_vel', 'gnss_pos', 'baro_hgt')],
				f"{np.mean(sf['cs_mag_3d'][m]):.2f}" if m.any() else '-', f"{np.mean(sf['cs_mag_hdg'][m]):.2f}" if m.any() else '-',
				f'{tilt[0]:.1f}', f'{yaw[0]:.1f}']) + ' |')
	return '\n'.join(lines)


def dead_reckon(d, t_seed, t_end, scale=0.0, bias=0.0):
	"""Seed with the EKF attitude at t_seed, then integrate raw gyro (body FRD) with the exact rotation per sample.
	scale / bias inject a gyro scale-factor error (fraction) and a constant bias [rad/s] on every axis."""
	a = d['vehicle_attitude']
	r = quat(a)[int(np.searchsorted(a['timestamp'] * 1e-6, t_seed))]
	sc = d['sensor_combined']
	ts = sc['timestamp'] * 1e-6
	m = (ts > t_seed) & (ts <= t_end)
	w = np.column_stack([sc[f'gyro_rad[{i}]'][m] for i in range(3)]) * (1 + scale) + bias
	steps = Rotation.from_rotvec(w * (sc['gyro_integral_dt'][m] * 1e-6)[:, None])
	out = []
	for step in steps:
		r = r * step
		out.append(r.as_quat())
	return ts[m], Rotation.from_quat(np.array(out))


def gyro_table():
	"""IDEA-023: gyro dead reckoning seeded 150 ms before detection, vs EKF2, errors vs truth after the fault."""
	topics = ['vehicle_attitude', 'vehicle_attitude_groundtruth', 'sensor_combined']
	evals = (0.5, 1.0, 2.0, 5.0, 10.0)
	cases = [('ideal gyro', 0.0, 0.0), ('scale +1 %', 0.01, 0.0), ('scale +3 %', 0.03, 0.0),
	         ('bias 0.5 deg/s', 0.0, np.radians(0.5)), ('bias 2 deg/s', 0.0, np.radians(2.0))]
	rows = {}
	seed_err = []
	for run in sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0*'))) + sorted(glob.glob(os.path.join(HERE, 'runs', 'extra_*'))):
		d, ev, info = load(run, topics)
		if 31011 not in ev:
			continue
		tf, t_seed = ev[31010], ev[31011] - 0.15
		g = d['vehicle_attitude_groundtruth']['timestamp'] * 1e-6
		t_end = min(tf + max(evals), g[-1])
		at = np.array([tf + e for e in evals if tf + e <= t_end])
		group = 'policy runs (15 s)' if 'ftc' in run else '50 % fault runs (PX4 flies, ~3 s)'
		seed_err.append(errors(quat(d['vehicle_attitude'])[[int(np.searchsorted(d['vehicle_attitude']['timestamp'] * 1e-6, t_seed))]],
		                       truth_at(d, np.array([t_seed])))[0][0])
		tilt_e, yaw_e = ekf_error(d, at)
		rows.setdefault((group, 'EKF2'), []).append((at - tf, tilt_e, yaw_e))
		for name, sc, b in cases:
			ts, est = dead_reckon(d, t_seed, t_end, sc, b)
			idx = np.searchsorted(ts, at).clip(0, len(ts) - 1)
			tilt, yaw = errors(est[idx], truth_at(d, ts[idx]))
			rows.setdefault((group, name), []).append((at - tf, tilt, yaw))
	lines = [f'Seed (EKF attitude 150 ms before detection) tilt error vs truth: max {max(seed_err):.2f} deg over {len(seed_err)} runs.', '',
	         '| runs | estimator | ' + ' | '.join(f'tilt / yaw at +{e:g} s' for e in evals) + ' |', '|' + '---|' * (2 + len(evals))]
	for (group, name), vals in rows.items():
		cells = []
		for e in evals:
			pts = [(t, y) for at_, tilt, yaw in vals for a, t, y in zip(at_, tilt, yaw) if abs(a - e) < 1e-6]
			cells.append(f'{np.mean([p[0] for p in pts]):.1f} / {np.mean([p[1] for p in pts]):.1f}' if pts else '-')
		lines.append(f'| {group} | {name} | ' + ' | '.join(cells) + ' |')
	return '\n'.join(lines)


def ekf_variants():
	"""IDEA-021: per EKF2 configuration (run tag), RMS tilt / yaw error over 0-5 s after the fault (until a crash)."""
	from ftc_analyze import score
	arms = {'': 'default (MAG_TYPE 0, auto)', 'maghdg': 'MAG_TYPE 1, heading only', 'maginit': 'MAG_TYPE 6, init only'}
	lines = ['| EKF2 config | runs | survived | tilt RMS 0-5 s [deg] | yaw RMS 0-5 s [deg] | tilt at +1 s [deg] | crash at [s] |',
	         '|---|---|---|---|---|---|---|']
	for tag, label in arms.items():
		runs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(HERE, 'runs', 'ftc_0?_ekf' + (f'_{tag}' if tag else ''), '*.ulg')))
		if not runs:
			continue
		tilt_rms, yaw_rms, tilt1, alive, crash = [], [], [], 0, []
		for run in runs:
			d, ev, _ = load(run, ['vehicle_attitude', 'vehicle_attitude_groundtruth'])
			s = score(run)
			tf = ev[31010]
			t_end = tf + min(5.0, s['crash_s'] or 5.0)
			ta = d['vehicle_attitude']['timestamp'] * 1e-6
			t = ta[(ta >= tf) & (ta < t_end)]
			tilt, yaw = ekf_error(d, t)
			tilt_rms.append(np.sqrt(np.mean(tilt ** 2)))
			yaw_rms.append(np.sqrt(np.mean(yaw ** 2)))
			tilt1.append(ekf_error(d, np.array([tf + 1.0]))[0][0])
			alive += s['survived']
			crash += [s['crash_s']] if s['crash_s'] else []
		lines.append(f'| {label} | {len(runs)} | {alive}/{len(runs)} | {np.mean(tilt_rms):.1f} ({min(tilt_rms):.1f}-{max(tilt_rms):.1f}) | '
		             f'{np.mean(yaw_rms):.1f} | {np.mean(tilt1):.1f} | {f"{min(crash):.1f}-{max(crash):.1f}" if crash else "-"} |')
	return '\n'.join(lines)


def kf_table():
	"""IDEA-022 step B offline: from detection on, gyro attitude + PosVelKF on IMU and GPS; velocity error vs truth."""
	for _cand in (os.path.join(HERE, '..', 'src'),
	              os.path.join(HERE, '..', 'code', 'px4_ws', 'src', 'rotor_fail', 'src')):
		if os.path.isfile(os.path.join(_cand, 'pos_kf.py')):
			sys.path.insert(0, _cand)
			break
	from pos_kf import PosVelKF
	topics = ['vehicle_attitude', 'vehicle_attitude_groundtruth', 'sensor_combined', 'vehicle_gps_position',
	          'vehicle_local_position', 'vehicle_local_position_groundtruth']
	conds = {'SITL GPS (~30 Hz)': (1, 0.0), 'GPS 10 Hz + 100 ms': (3, 0.1)}
	res = {}
	for run in sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0[0-7]_gt'))) + sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0[0-7]_ekf'))):
		d, ev, _ = load(run, topics)
		t0, tf = ev[31011], ev[31010]
		gt = d['vehicle_local_position_groundtruth']
		tg = gt['timestamp'] * 1e-6
		vt = lambda t: np.array([np.interp(t, tg, gt[k]) for k in ('vx', 'vy', 'vz')])  # noqa: E731
		crash = tg[(tg > tf) & (-gt['z'] + gt['z'][0] < 0.3)]
		t_end = min(tf + 5.0, crash[0] if len(crash) else np.inf)
		lp = d['vehicle_local_position']
		i0 = int(np.searchsorted(lp['timestamp'] * 1e-6, t0))
		lat0, lon0, alt0 = np.radians(lp['ref_lat'][i0]), np.radians(lp['ref_lon'][i0]), lp['ref_alt'][i0]
		gps = d['vehicle_gps_position']
		tgps = gps['timestamp'] * 1e-6
		gp = np.column_stack([(np.radians(gps['latitude_deg']) - lat0) * 6371000.0,
		                      (np.radians(gps['longitude_deg']) - lon0) * 6371000.0 * np.cos(lat0), -(gps['altitude_msl_m'] - alt0)])
		gv = np.column_stack([gps['vel_n_m_s'], gps['vel_e_m_s'], gps['vel_d_m_s']])
		ts_att, att = dead_reckon(d, t0 - 0.15, t_end)
		sc = d['sensor_combined']
		ts = sc['timestamp'] * 1e-6
		acc = np.column_stack([sc[f'accelerometer_m_s2[{i}]'] for i in range(3)])
		ekf_v = lambda t: np.array([np.interp(t, lp['timestamp'] * 1e-6, lp[k]) for k in ('vx', 'vy', 'vz')])  # noqa: E731
		group = 'policy on GT (15 s)' if run.endswith('_gt') else 'policy on EKF (crash ~2 s)'
		for cond, (every, delay) in conds.items():
			fixes = np.arange(0, len(tgps), every)
			kf = PosVelKF([lp['x'][i0], lp['y'][i0], lp['z'][i0]], [lp['vx'][i0], lp['vy'][i0], lp['vz'][i0]])
			j = int(np.searchsorted(tgps[fixes] + delay, t0))
			err_kf, err_ekf = [], []
			for k in np.flatnonzero((ts > t0) & (ts <= t_end)):
				kf.predict(att[min(np.searchsorted(ts_att, ts[k]), len(att) - 1)].as_matrix(), acc[k], sc['gyro_integral_dt'][k] * 1e-6)
				while j < len(fixes) and tgps[fixes[j]] + delay <= ts[k]:
					kf.update_gps(gp[fixes[j]], gv[fixes[j]])
					j += 1
				if ts[k] >= tf:
					err_kf.append(np.linalg.norm(kf.v - vt(ts[k])))
					err_ekf.append(np.linalg.norm(ekf_v(ts[k]) - vt(ts[k])))
			res.setdefault((group, cond), []).append((np.sqrt(np.mean(np.square(err_kf))), np.sqrt(np.mean(np.square(err_ekf)))))
	lines = ['| flights | GPS | runs | velocity RMS 0-5 s, gyro att + KF [m/s] | EKF2 [m/s] |', '|---|---|---|---|---|']
	for (group, cond), v in res.items():
		v = np.array(v)
		lines.append(f'| {group} | {cond} | {len(v)} | {v[:, 0].mean():.2f} (max {v[:, 0].max():.2f}) | {v[:, 1].mean():.2f} |')
	return '\n'.join(lines)


def heading_table():
	"""Heading error in the spin with an injected gyro scale error (all axes): gyro dead reckoning alone, with
	magnetometer heading aiding (complementary, time constant TAU, field direction learned at the seed), and PX4's
	EKF-GSF yaw (GPS velocity + IMU, logged as yaw_estimator_status; it saw the true gyro, so only its
	no-scale-error row is meaningful)."""
	TAU, MAX_TILT = 0.5, 45.0
	topics = ['vehicle_attitude', 'vehicle_attitude_groundtruth', 'sensor_combined', 'vehicle_magnetometer', 'yaw_estimator_status']
	evals = (2.0, 5.0, 10.0, 15.0)
	res = {}
	for run in sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0[0-7]_gt_i022c'))) + sorted(glob.glob(os.path.join(HERE, 'runs', 'ftc_0[0-7]_gyro_i022c'))):
		d, ev, _ = load(run, topics)
		tf, t0 = ev[31010], ev[31011] - 0.15
		g = d['vehicle_attitude_groundtruth']['timestamp'] * 1e-6
		t_end = min(tf + max(evals), g[-1])
		a = d['vehicle_attitude']
		r_seed = quat(a)[int(np.searchsorted(a['timestamp'] * 1e-6, t0))]
		sc, mg = d['sensor_combined'], d['vehicle_magnetometer']
		ts, tm = sc['timestamp'] * 1e-6, mg['timestamp_sample'] * 1e-6   # sample time: at 25 rad/s a 20 ms lag is 30 deg
		gyro = np.column_stack([sc[f'gyro_rad[{i}]'] for i in range(3)])
		mag = np.column_stack([mg[f'magnetometer_ga[{i}]'] for i in range(3)])
		im0 = int(np.searchsorted(tm, t0))
		m_ref = r_seed.apply(mag[im0])
		psi_ref = np.arctan2(m_ref[1], m_ref[0])
		at = np.array([tf + e for e in evals if tf + e <= t_end])
		for scale in (0.0, 0.01, 0.03):
			for aid in (False, True):
				r, j, out = r_seed, im0 + 1, []
				for k in np.flatnonzero((ts > t0) & (ts <= t_end)):
					r = r * Rotation.from_rotvec(gyro[k] * (1 + scale) * sc['gyro_integral_dt'][k] * 1e-6)
					while aid and j < len(tm) and tm[j] <= ts[k]:
						if r.apply([0, 0, 1])[2] < np.cos(np.radians(MAX_TILT)):   # skip mag while tumbling: level projection is poor
							j += 1
							continue
						m_w = r.apply(mag[j])
						e = np.arctan2(np.sin(psi_ref - np.arctan2(m_w[1], m_w[0])), np.cos(psi_ref - np.arctan2(m_w[1], m_w[0])))
						r = Rotation.from_rotvec([0, 0, e * (tm[j] - tm[j - 1]) / TAU]) * r
						j += 1
					out.append((ts[k], r.as_quat()))
				tt = np.array([o[0] for o in out])
				idx = np.searchsorted(tt, at).clip(0, len(tt) - 1)
				_, yaw = errors(Rotation.from_quat(np.array([o[1] for o in out]))[idx], truth_at(d, tt[idx]))
				res.setdefault((f'scale {scale * 100:.0f} %', 'gyro + mag heading' if aid else 'gyro only'), []).append((at - tf, yaw))
		ys = d['yaw_estimator_status']
		ty = ys['timestamp'] * 1e-6
		iy = np.searchsorted(ty, at).clip(0, len(ty) - 1)
		tr = truth_at(d, ty[iy]).as_euler('ZYX')[:, 0]
		res.setdefault(('scale 0 %', 'PX4 EKF-GSF yaw'), []).append((at - tf, np.degrees(np.abs(np.angle(np.exp(1j * (ys['yaw_composite'][iy] - tr)))))))
	lines = ['| gyro scale error | heading source | ' + ' | '.join(f'heading err +{e:g} s [deg]' for e in evals) + ' |', '|' + '---|' * (2 + len(evals))]
	for (sc_, name), vals in res.items():
		cells = []
		for e in evals:
			pts = [y for at_, yy in vals for a_, y in zip(at_, yy) if abs(a_ - e) < 1e-6]
			cells.append(f'{np.median(pts):.1f} (max {np.max(pts):.0f})' if pts else '-')
		lines.append(f'| {sc_} | {name} | ' + ' | '.join(cells) + ' |')
	return '\n'.join(lines)


def main():
	os.makedirs(OUT, exist_ok=True)
	if sys.argv[1:] == ['aid']:
		md = '# IDEA-020: EKF2 aid-source fusion after the fault vs EKF attitude error\n\n' + aid_table() + '\n'
		open(os.path.join(OUT, 'est_aid_sources.md'), 'w').write(md)
		print(md)
	elif sys.argv[1:] == ['gyro']:
		md = ('# IDEA-023: raw-gyro dead reckoning vs EKF2 (mean tilt / yaw error vs truth, deg)\n\n'
		      'SITL gyro (near ideal); scale / bias rows inject errors to mimic a real IMU.\n\n' + gyro_table() + '\n')
		open(os.path.join(OUT, 'est_gyro.md'), 'w').write(md)
		print(md)
	elif sys.argv[1:] == ['kf']:
		md = '# IDEA-022 step B offline: gyro attitude + position/velocity KF vs EKF2 (vs truth, from the fault)\n\n' + kf_table() + '\n'
		open(os.path.join(OUT, 'est_kf.md'), 'w').write(md)
		print(md)
	elif sys.argv[1:] == ['heading']:
		md = ('# Heading error in the spin vs gyro scale error (median over 16 runs, i022c on GT and onboard state)\n\n'
		      + heading_table() + '\n')
		open(os.path.join(OUT, 'est_heading.md'), 'w').write(md)
		print(md)
	elif sys.argv[1:] == ['ekf']:
		md = '# IDEA-021: EKF2 magnetometer configurations, policy on EKF state (mean over runs)\n\n' + ekf_variants() + '\n'
		open(os.path.join(OUT, 'est_ekf_variants.md'), 'w').write(md)
		print(md)
	else:
		print(__doc__)


if __name__ == '__main__':
	main()
