#!/usr/bin/env python3
"""The PX4/Gazebo bench protocol in the Gazebo-gap model, scored the way the bench's ftc_analyze.py scores it.

20 units (failed motor x 5 starts), fault at 3 s. Until the handover the rotors get PX4's logged reaction (ftc.px4_standin: failed rotor to max, its partner to the floor),
or hover thrust with --standin hover; from the handover on the policy flies (shadow mode: its filters ran all along).
Scores, per setting: crashes (tilt > 90 deg, or more than 2.7 m height loss), xy RMS error from the hold point over the last 5 s of fault + 15 s,
orbit radius (xy RMS about the mean over that window), peak tilt, max xy excursion, altitude loss.
  python bench_sim.py runs/<id>/policy.pkl [--handover 80] [--standin px4|hover] [--only gt_stop ...] [--out results.json]
"""
import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import eval as ev
import ftc
import train

BENCH_ALT = 3.0  # m, bench hover height (crash = below 0.3 m)
GT = ftc.GZ_EVAL | {"delay": 2, "lag": 0, "noise": 0.2}  # bench with Gazebo ground truth: a few ms of latency
SETTINGS = {
    "gt_stop": GT,
    "gt_prop": GT | {"spin": 1.0},
    "gz_eval_stop": ftc.GZ_EVAL,
    "gz_eval_prop": ftc.GZ_EVAL | {"spin": 1.0},
    "rotor_j1_stop": GT | {"prop_j": 1.0},  # physical rotor inertia (real hardware rather than Gazebo)
} | {f"ekf{d}_stop": GT | {"att_max": np.radians(d), "att_rate": np.radians(30.0)} for d in (10, 20, 30, 40, 60)}
# The bench's flight phases at the fault (data_00..39), each unit with its own heading; "t_fail" [s] after the start.
YAWS = np.linspace(-np.pi, np.pi, ev.N, endpoint=False)
PHASES = {"hover": {}, "forward": {"vel": [2.0, 0.0, 0.0]}, "climb": {"vel": [0.0, 0.0, 1.0]},
          "descent": {"vel": [0.0, 0.0, -1.0]}, "yaw": {"wz": 1.5}}
SETTINGS |= {f"{ph}_{mode}": GT | {"spin": float(mode == "prop"), "yaw": YAWS, "t_fail": 0.3} | st
             for ph, st in PHASES.items() for mode in ("stop", "prop")}
# Held-out "hardware" plant deviations (never trained on): one draw per unit, fixed seed. Extremes, not means.
_r = np.random.default_rng(0)
_cg = _r.uniform(0.01, 0.02, ev.N)[:, None] * np.stack([np.cos(a := _r.uniform(0, 2 * np.pi, ev.N)), np.sin(a),
                                                        np.zeros(ev.N)], -1)
HW = {"mass": _r.choice([0.9, 1.1], ev.N), "cg": _cg, "thrust": {"t_err": _r.uniform(0.9, 1.1, (ev.N, 4)), "sag": 0.85},
      "lag": {"motor_tau": 2.0}, "J": _r.choice([0.8, 1.2], ev.N)}
HW = {k: v if isinstance(v, dict) else {k: v} for k, v in HW.items()}
HW["all"] = {kk: vv for v in HW.values() for kk, vv in v.items()} | {"prop_j": 1.0}
HW = {k: v | {"pre": "policy"} for k, v in HW.items()} | {"none": {"pre": "policy"}}  # a plant error needs feedback before the fault
# wider than train.py --hw-rand (mass 0.9-1.1, CG 0-2 cm, thrust 0.9-1.1, sag 0.85-1, motor lag 0.8-2, J 0.8-1.2)
_cg2 = _r.uniform(0.02, 0.03, ev.N)[:, None] * np.stack([np.cos(a := _r.uniform(0, 2 * np.pi, ev.N)), np.sin(a), np.zeros(ev.N)], -1)
HWX = {"mass": {"mass": _r.choice([0.85, 1.15], ev.N)}, "cg": {"cg": _cg2},
       "thrust": {"t_err": _r.uniform(0.85, 1.15, (ev.N, 4)), "sag": 0.8}, "lag": {"motor_tau": 2.5}, "J": {"J": _r.choice([0.7, 1.3], ev.N)}}
HWX["all"] = {kk: vv for v in HWX.values() for kk, vv in v.items()} | {"prop_j": 1.0}
HWX = {k: v | {"pre": "policy"} for k, v in HWX.items()}
SETTINGS |= {f"hwx_{k}_{mode}": GT | {"spin": float(mode == "prop")} | v for k, v in HWX.items() for mode in ("stop", "prop")}
SETTINGS |= {f"hw_{k}_{mode}": GT | {"spin": float(mode == "prop")} | v for k, v in HW.items() for mode in ("stop", "prop")}
# Partial loss (rotor at 30-80 % thrust); the policy flies the healthy drone (and, at >= 50 %, through the fault)
PART = (30, 50, 65, 80)
SETTINGS |= {f"part{h}_stop": GT | {"health": h / 100, "pre": "policy"} for h in PART}
SETTINGS |= {f"part{h}_{ph}": SETTINGS[f"{ph}_stop"] | {"health": h / 100, "pre": "policy"} for h in (30, 50) for ph in PHASES}
# Onboard estimator (bench IDEA-022 B). cal_*: passive (kf 0), the drift the bench measured offline on its logs.
SETTINGS |= {"cal_scale1": GT | {"g_s": 0.01}, "cal_scale3": GT | {"g_s": 0.03}, "cal_bias2": GT | {"g_b": np.radians(2.0)},
             "ob_ideal": GT | {"noise": 0.0, "lag": 0, "kf": 1.0}}
OB = {"kf": 1.0, "gps_period": 10, "gps_delay": 10, "gps_vn": 0.2, "gps_pj": 0.02}  # 10 Hz GPS, 100 ms
OB_NOM = OB | {"g_s": 0.01, "g_b": np.radians(1.0), "head": np.radians(6.0)}
SETTINGS |= {f"ob_scale{x}": GT | OB | {"g_s": x / 100} for x in (0, 1, 2, 3, 4, 6)}
SETTINGS |= {f"ob_bias{x}": GT | OB | {"g_b": np.radians(x)} for x in (1, 2, 4)}
SETTINGS |= {f"ob_head{x}": GT | OB | {"head": np.radians(x)} for x in (5, 10, 20, 30)}
SETTINGS |= {f"ob_gps{f}_{d}": GT | OB | {"gps_period": 100 // f, "gps_delay": d // 10} for f, d in ((30, 0), (10, 100), (5, 200))}
SETTINGS |= {f"ob_{ph}_{mode}": SETTINGS[f"{ph}_{mode}"] | OB_NOM for ph in PHASES for mode in ("stop", "prop")}
SETTINGS |= {f"hw_all_{ph}_{mode}": SETTINGS[f"{ph}_{mode}"] | HW["all"] for ph in PHASES for mode in ("stop", "prop")}


def run(params, rand, handover_ms, standin="px4"):
    names, pos, k, t_fail = ev.units(True)
    rand = dict(rand)
    t_fail = jnp.full(ev.N, round(rand.pop("t_fail", ftc.T_FAIL) * ftc.CTRL_HZ))
    health = rand.pop("health", 0.0)  # failed rotor's thrust share (partial loss); PX4's reaction only below 50 %
    k = jnp.where(k < 1.0, health, 1.0)
    pre = rand.pop("pre", "hover")  # who flies the healthy drone: open-loop hover thrust, or the policy (closed loop)
    sim = ftc.make_sim(ev.N, gz=True)
    sim_step = sim.build_step_fn()
    start = int(t_fail[0]) + handover_ms // 10

    def ctrl(e, _):
        before = jnp.where((e.t >= e.t_fail)[:, None] & (standin == "px4"), ftc.px4_standin(e), 0.0)
        mine = (e.t >= start) | (((e.t < e.t_fail) | (health >= 0.5)) & (pre == "policy"))
        a = jnp.where(mine[:, None], train.policy_mean(params, ftc.observe(e)), before)
        e, _, crashed = ftc.step(sim_step, e, a)
        s = e.data.states
        return e, (s.pos[:, 0], s.quat[:, 0], crashed, e.q_err, e.est[:, :6] - jnp.concatenate([s.pos[:, 0], s.vel[:, 0]], -1))

    e = ftc.reset_where(ftc.init(sim), jnp.ones(ev.N, bool), pos, t_fail, k, rand)
    p, q, c, qe, pv_err = map(np.asarray, jax.jit(lambda e: jax.lax.scan(ctrl, e, length=start + 15 * ftc.CTRL_HZ))(e)[1])
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2), -1, 1)))
    # onboard attitude error: angle between estimated and true body z (the bench's estimation.py definition)
    z_t = np.asarray(ftc.rotmat(jnp.asarray(q)))[..., 2]
    z_e = np.asarray(ftc.rotmat(ftc.quat_mul(jnp.asarray(qe), jnp.asarray(q))))[..., 2]
    att_e = np.degrees(np.arccos(np.clip(np.sum(z_t * z_e, -1), -1, 1)))
    tot_e = np.degrees(2 * np.arccos(np.clip(np.abs(qe[..., 3]), 0, 1)))
    yaw_e = np.sqrt(np.maximum(tot_e ** 2 - att_e ** 2, 0))  # estimation.py: heading part of the total error
    xy = p[start:, :, :2] - p[start, :, :2]
    last = xy[-5 * ftc.CTRL_HZ:]
    t0 = int(t_fail[0])
    peak = tilt[t0:t0 + 3 * ftc.CTRL_HZ].max(0)  # first 3 s after the fault
    alt = p[t0, :, 2] - p[t0:, :, 2].min(0)
    # the bench's crash rule (crazyflow has no floor): flipped past 90 deg, or ground contact from a 3 m hover
    flipped = (tilt[t0:] > 90.0).any(0)
    crashed = flipped | (alt > BENCH_ALT - 0.3)
    ok = ~crashed
    f = lambda x: round(float(np.mean(x[ok])), 3) if ok.any() else None  # noqa: E731
    return {"crashed": int(crashed.sum()), "crashed_units": np.flatnonzero(crashed).tolist(), "flipped": int(flipped.sum()), "inverted": int((tilt[t0:] > 150.0).any(0).sum()), "xy_rms_last5": f(np.sqrt(np.mean(np.sum(last ** 2, -1), 0))),
            "orbit_radius": f(np.sqrt(np.mean(np.sum((last - last.mean(0)) ** 2, -1), 0))),
            "peak_tilt_max": round(float(peak.max()), 1), "peak_tilt_mean": round(float(peak.mean()), 1),
            "max_xy": f(np.linalg.norm(xy, axis=-1).max(0)), "alt_loss_mean": round(float(alt.mean()), 2),
            "alt_loss_max": round(float(alt.max()), 2),
            **{f"tilt_err_{d}s": round(float(np.median(att_e[min(t0 + d * ftc.CTRL_HZ, len(att_e) - 1)])), 1) for d in (1, 5, 10)},
            **{f"yaw_err_{d}s": round(float(np.median(yaw_e[min(t0 + d * ftc.CTRL_HZ, len(yaw_e) - 1)])), 1) for d in (5, 10)},
            "xy_units": np.round(np.sqrt(np.mean(np.sum(last ** 2, -1), 0)), 4).tolist(),
            "peak_units": np.round(peak, 2).tolist(), "alt_units": np.round(alt, 4).tolist(),
            "v_err_rms": round(float(np.sqrt(np.mean(np.sum(pv_err[t0:, :, 3:] ** 2, -1)))), 3),
            "p_err_rms": round(float(np.sqrt(np.mean(np.sum(pv_err[t0:, :, :3] ** 2, -1)))), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--handover", type=int, default=80, help="ms after the fault")
    ap.add_argument("--standin", default="px4", choices=["px4", "hover"], help="what flies before the handover")
    ap.add_argument("--only", nargs="*", default=None, help="subset of the settings")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params = pickle.load(open(a.policy, "rb"))
    res = {}
    for name, rand in SETTINGS.items():
        if a.only and name not in a.only:
            continue
        res[name] = run(params, rand, a.handover, a.standin)
        print(f"{name:14s} {res[name]}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
