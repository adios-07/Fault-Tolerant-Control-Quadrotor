#!/usr/bin/env python3
"""Stress test: where does a policy break? One batched rollout per mode on the Gazebo-gap model.

sweep: one error at a time, far past the training ranges, 20 units per point (4 motors x 5 starts, starts alternate
       stop / prop loss, per-unit heading). Prints crashes per 20, xy RMS last 5 s, peak tilt, mean altitude loss.
mc:    every error drawn wide at once per episode; logistic regression on standardised factors ranks what breaks.
The policy flies the healthy drone (pre-fault feedback, like PX4), PX4's logged reaction flies from a complete loss
until the handover (80 ms unless swept), then the policy. Crash = tilt > 90 deg after the fault, or height loss
> start height - 0.3 m (3 m hover, 1 m for "low altitude").
  python stress.py runs/<id>/policy.pkl [--mode sweep|mc|both] [--mc-n 4000] [--out stress.json]
"""
import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import bench_sim as b
import eval as ev
import ftc
import train

ftc.QUEUE_LEN, ftc.HIST_LEN, ftc.GPS_HIST = 40, 7, 51  # 80 ms delay, 60 ms lag, 500 ms GPS delay
U = ev.N  # 20 units per sweep point
M_OF, S_OF = np.arange(U) // 5, np.arange(U) % 5
RP = np.asarray(ftc.ROTOR_POS)[:, :2]
TO_FAILED = RP / np.linalg.norm(RP, axis=-1, keepdims=True)  # unit vector from the centre to each rotor
OB = {"kf": 1.0, "gps_period": 10, "gps_delay": 10, "gps_vn": 0.2, "gps_pj": 0.02}  # onboard estimator, 10 Hz GPS


def cg_dir(cm, ang):
    """CG offset of `cm` at angle `ang` from the direction of the failed rotor, per unit (U,3)."""
    d = TO_FAILED[M_OF]
    c, s = np.cos(ang), np.sin(ang)
    v = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], np.zeros(U)], -1)
    return v * cm / 100


def second(frac, diag):
    """k with the failed motor dead and a second rotor at `frac`: its diagonal partner or its neighbour."""
    k = 1.0 - np.eye(4)[M_OF]
    other = np.asarray(ftc.ROT180[1])[M_OF] if diag else np.asarray(ftc.MIRROR[1])[M_OF]
    k[np.arange(U), other] = frac
    return k


def tilt(deg, all_rotors, seed=1):
    """Thrust-axis tilt (bent mount): random direction per rotor, on the failed rotor's neighbours or all rotors."""
    r = np.random.default_rng(seed)
    a = r.uniform(0, 2 * np.pi, (U, 4))
    t = np.stack([np.cos(a), np.sin(a), np.zeros((U, 4))], -1) * np.radians(deg)
    return t if all_rotors else t * (1.0 - np.eye(4)[M_OF])[..., None]


def points():
    """(group, label, overrides). Overrides: ftc rand keys (scalar, per-point or (U,...) per unit) and the extras
    k (U,4), handover [ms], alt0 [m], t_fail [s], start_off [m]."""
    P = []
    add = lambda g, l, o: P.append((g, l, o))  # noqa: E731
    for v in (0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5):
        add("mass x", v, {"mass": v})
    for v in (0.6, 0.7, 0.8, 1.2):
        add("thrust curve x (all)", v, {"t_err": v})
    for v in (0.2, 0.3):
        add("thrust curve spread +-", v, {"t_err": np.random.default_rng(2).uniform(1 - v, 1 + v, (U, 4))})
    for v in (0.8, 0.7, 0.6, 0.5):
        add("battery sag (max thrust x)", v, {"sag": v})
    for v in (2, 3, 4, 6):
        add("motor lag x", v, {"motor_tau": v})
    for v in (0.4, 0.6, 1.5, 2.0, 2.5):
        add("inertia J x", v, {"J": v})
    for v in (0.03, 2.0, 3.0):
        add("rotor inertia x", v, {"prop_j": v})
    for cm in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        add("CG toward failed rotor cm", cm, {"cg": cg_dir(cm, 0.0)})
    for cm in (2.0, 3.0, 5.0):
        add("CG away from failed rotor cm", cm, {"cg": cg_dir(cm, np.pi)})
    for cm in (2.0, 3.0, 5.0):
        add("CG sideways cm", cm, {"cg": cg_dir(cm, np.pi / 2)})
    for ms in (20, 40, 60, 80):
        add("actuation delay ms", ms, {"delay": ms // 2})
    for ms in (20, 40, 60):
        add("state lag ms", ms, {"lag": ms // 10})
    for v in (2, 5, 10, 20):
        add("estimate noise x", v, {"noise": v})
    for ms in (200, 300, 500):
        add("handover ms", ms, {"handover": ms})
    for v in (6, 8, 10):
        add("gyro bias deg/s (onboard)", v, OB | {"g_b": np.radians(v)})
    for v in (45, 60, 90, 120):
        add("heading offset deg (onboard)", v, OB | {"head": np.radians(v)})
    for hz, ms in ((2, 300), (1, 500), (1 / 3, 500)):
        add("GPS Hz/delay ms (onboard)", f"{hz:.2g}/{ms}", OB | {"gps_period": round(100 / hz), "gps_delay": ms // 10})
    yaws = b.YAWS
    for lab, st in (("forward 5 m/s", {"vel": [5.0, 0, 0]}), ("climb 3 m/s", {"vel": [0, 0, 3.0]}),
                    ("descent 3 m/s", {"vel": [0, 0, -3.0]}), ("yaw 4 rad/s", {"wz": 4.0})):
        add("flight state at fault", lab, st | {"yaw": yaws, "t_fail": 0.3})
    add("flight state at fault", "start 2 m off", {"start_off": 2.0})
    add("flight state at fault", "1 m above ground", {"alt0": 1.0})
    for v in (3.0, 6.0, 10.0):
        add("steady wind m/s", v, {"wind": [v, 0.0, 0.0]})
    add("gusty wind m/s", "3 +- 3", {"wind": [3.0, 0.0, 0.0], "gust": [3.0, 2.0, 0.0]})
    for deg in (5, 10):
        add("thrust tilt deg, healthy rotors", deg, {"tilt": tilt(deg, False)})
    for deg in (3, 5):
        add("thrust tilt deg, all rotors", deg, {"tilt": tilt(deg, True)})
    for frac in (0.9, 0.8, 0.7, 0.5):
        add("2nd rotor weak (neighbour)", frac, {"k": second(frac, False)})
    for frac in (0.7, 0.5):
        add("2nd rotor weak (diagonal)", frac, {"k": second(frac, True)})
    for q in (0.5, 1.5, 2.0):
        add("partial 65 %, torque ratio", q, {"health": 0.65, "q_ratio": q})
    add("partial 30 %, torque ratio", 1.5, {"health": 0.3, "q_ratio": 1.5})
    for s_ in (0.5, 2.0, 5.0):
        add("fault onset ramp s", s_, {"ramp": round(s_ * ftc.CTRL_HZ)})
    return P


EXTRA = ("k", "handover", "alt0", "t_fail", "start_off", "health")


def unit_values(o):
    """Per-unit arrays for one point: each value broadcast to (U, ...)."""
    out = {}
    names, pos, k, _ = ev.units(True)
    health = o.get("health", 0.0)
    out["k"] = np.asarray(o.get("k", np.where(np.asarray(k) < 1, health, 1.0)), np.float32)
    out["handover"] = np.full(U, o.get("handover", 80))
    out["alt0"] = np.full(U, o.get("alt0", b.BENCH_ALT))
    out["t_fail"] = np.full(U, round(o.get("t_fail", ftc.T_FAIL) * ftc.CTRL_HZ))
    off = np.zeros((U, 3))
    if "start_off" in o:
        a = np.linspace(0, 2 * np.pi, U, endpoint=False)
        off = o["start_off"] * np.stack([np.cos(a), np.sin(a), np.zeros(U)], -1)
    out["pos"] = np.asarray(pos) + off
    rand = b.GT | {"spin": (S_OF % 2).astype(np.float32)} | {kk: v for kk, v in o.items() if kk not in EXTRA}
    for kk, v in rand.items():
        v = np.asarray(v, np.float32)
        shape = np.shape(np.asarray(ftc.NOMINAL[kk])) if kk in ftc.NOMINAL else ()
        if kk in ("vel", "wind", "gust", "cg", "g_s", "g_b", "att_seed"):
            shape = (3,)
        elif kk == "t_err":
            shape = (4,)
        elif kk == "tilt":
            shape = (4, 3)
        out[kk] = np.broadcast_to(v, (U,) + shape) if v.shape != (U,) + shape else v
    return out


def rollout(params, cols):
    """cols: dict of per-world arrays (W, ...). Returns per-world crash, xy RMS last 5 s, peak tilt, alt loss."""
    W = len(cols["t_fail"])
    sim = ftc.make_sim(W, gz=True)
    sim_step = sim.build_step_fn()
    t_fail = jnp.asarray(cols["t_fail"], jnp.int32)
    start = t_fail + jnp.asarray(cols["handover"] // 10, jnp.int32)
    steps = int(np.max(cols["t_fail"] + cols["handover"] // 10)) + 15 * ftc.CTRL_HZ
    severe = jnp.asarray(cols["k"].min(-1) < 0.5)
    rand = {kk: jnp.asarray(v) for kk, v in cols.items() if kk in ftc.NOMINAL}

    def ctrl(e, _):
        pol = train.policy_mean(params, ftc.observe(e))
        mine = (e.t >= start) | (e.t < e.t_fail) | ~severe  # PX4 stand-in only for a severe fault, until the handover
        a = jnp.where(mine[:, None], pol, ftc.px4_standin(e))
        e, _, _ = ftc.step(sim_step, e, a)
        s = e.data.states
        return e, (s.pos[:, 0], s.quat[:, 0])

    e = ftc.reset_where(ftc.init(sim), jnp.ones(W, bool), jnp.asarray(cols["pos"], jnp.float32), t_fail,
                        jnp.asarray(cols["k"]), rand)
    p, q = map(np.asarray, jax.jit(lambda e: jax.lax.scan(ctrl, e, length=steps))(e)[1])
    tilt_ = np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2), -1, 1)))
    T = np.arange(steps)[:, None]
    after = T >= np.asarray(cols["t_fail"])[None]
    t0 = np.asarray(cols["t_fail"])
    w = np.arange(W)
    alt = p[t0, w, 2] - np.where(after, p[..., 2], np.inf).min(0)
    flipped = (np.where(after, tilt_, 0.0) > 90.0).any(0)
    crashed = flipped | (alt > cols["alt0"] - 0.3)
    st = np.asarray(start)
    xy = p[-5 * ftc.CTRL_HZ:, :, :2] - p[st, w, :2][None]
    peak = np.where(after & (T < t0[None] + 3 * ftc.CTRL_HZ), tilt_, 0.0).max(0)
    return crashed, np.sqrt(np.mean(np.sum(xy ** 2, -1), 0)), peak, alt


def concat(per_point):
    return {kk: np.concatenate([c[kk] for c in per_point]) for kk in per_point[0]}


def sweep(params):
    P = points()
    cols = [unit_values(o) for _, _, o in P]
    keys = set().union(*cols)
    for c in cols:  # every point needs every key: fill with the point-free default
        for kk in keys - set(c):
            ref = next(x[kk] for x in cols if kk in x)
            c[kk] = np.broadcast_to(np.asarray(b.GT.get(kk, ftc.NOMINAL.get(kk)), ref.dtype), ref.shape)
    crashed, xy, peak, alt = rollout(params, concat(cols))
    res, last = [], None
    for i, (g, l, _) in enumerate(P):
        sl = slice(U * i, U * (i + 1))
        ok = ~crashed[sl]
        row = {"group": g, "value": l, "crashed": int(crashed[sl].sum()),
               "xy": round(float(xy[sl][ok].mean()), 3) if ok.any() else None,
               "peak": round(float(peak[sl].max()), 1), "alt": round(float(alt[sl].mean()), 2)}
        res.append(row)
        if g != last:
            print(f"\n{g}")
            last = g
        print(f"  {str(l):>10s}: crash {row['crashed']:2d}/20  xy {str(row['xy']):>6s} m  peak {row['peak']:5.1f} deg  "
              f"alt loss {row['alt']:.2f} m")
    return res


def mc(params, n, seed=0, scale=1.0):
    """Everything wide at once. scale shrinks every range toward nominal (1 = brutal, 0.5 = moderate).
    Returns factor table, crash vector and the logistic-regression ranking."""
    r = np.random.default_rng(seed)
    u = lambda lo, hi, nom, size=n: nom + scale * (r.uniform(lo, hi, size) - nom)  # noqa: E731
    m = r.integers(0, 4, n)
    k = 1.0 - np.eye(4)[m]
    health = np.where(r.random(n) < 0.3, r.uniform(0.3, 0.8, n), 0.0)
    k[np.arange(n), m] = health
    other = np.asarray(ftc.MIRROR[1])[m]
    weak2 = np.where(r.random(n) < 0.2, u(0.5, 1.0, 1.0), 1.0)
    k[np.arange(n), other] = np.minimum(k[np.arange(n), other], weak2)
    ang, cg_cm = r.uniform(0, 2 * np.pi, n), u(0, 3, 0)
    d = TO_FAILED[m]
    cg = np.stack([np.cos(ang) * d[:, 0] - np.sin(ang) * d[:, 1], np.sin(ang) * d[:, 0] + np.cos(ang) * d[:, 1],
                   np.zeros(n)], -1) * cg_cm[:, None] / 100
    wind = u(0, 8, 0)
    wa = r.uniform(0, 2 * np.pi, n)
    tilt_deg = u(0, 6, 0)
    ta = r.uniform(0, 2 * np.pi, (n, 4))
    moving = r.random(n) < 0.5
    sp = moving * u(0, 4, 0)
    va = r.uniform(0, 2 * np.pi, n)
    f = {"mass": u(0.75, 1.3, 1.0), "t_spread": u(0, 0.25, 0), "sag": u(0.6, 1.0, 1.0),
         "motor_tau": u(0.8, 4.0, 1.0), "J": u(0.6, 1.8, 1.0), "prop_j": np.exp(u(np.log(0.05), np.log(2.0), np.log(0.1))),
         "delay_ms": u(0, 60, 4), "lag_ms": np.round(u(0, 40, 0) / 10) * 10.0, "noise": u(0, 5, 0.2),
         "handover_ms": u(0, 300, 80), "bias_dps": u(0, 6, 0), "head_deg": u(0, 45, 0),
         "gps_period": np.round(u(10, 100, 10)), "wind": wind, "tilt_deg": tilt_deg,
         "q_ratio": u(0.5, 2.0, 1.0), "cg_toward_cm": cg_cm * np.cos(ang), "cg_side_cm": cg_cm * np.abs(np.sin(ang)),
         "partial": (health > 0).astype(float), "weak2": 1.0 - weak2, "speed": sp, "wz": moving * u(-4, 4, 0),
         "onset_s": np.where(r.random(n) < 0.3, u(0, 3, 0), 0.0)}
    te = 1.0 + r.uniform(-1, 1, (n, 4)) * f["t_spread"][:, None]
    names, pos, _, _ = ev.units(True)
    cols = {"k": k.astype(np.float32), "handover": np.round(f["handover_ms"] / 10).astype(int) * 10,
            "alt0": np.full(n, b.BENCH_ALT), "t_fail": np.where(moving, 30, 300),
            "pos": np.asarray(ftc.TARGET) + r.uniform(-0.3, 0.3, (n, 3))}
    rand = b.GT | {"spin": (r.random(n) < 0.5).astype(np.float32), "mass": f["mass"], "t_err": te, "sag": f["sag"],
                   "motor_tau": f["motor_tau"], "J": f["J"], "prop_j": f["prop_j"], "delay": np.round(f["delay_ms"] / 2),
                   "lag": f["lag_ms"] / 10, "noise": f["noise"], "cg": cg, "kf": np.ones(n), "gps_delay": np.full(n, 10),
                   "gps_period": f["gps_period"], "gps_vn": np.full(n, 0.2), "gps_pj": np.full(n, 0.02),
                   "g_b": np.radians(f["bias_dps"])[:, None] * np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
                   "head": np.radians(f["head_deg"]) * np.sign(r.uniform(-1, 1, n)),
                   "wind": wind[:, None] * np.stack([np.cos(wa), np.sin(wa), np.zeros(n)], -1),
                   "tilt": np.stack([np.cos(ta), np.sin(ta), np.zeros((n, 4))], -1) * np.radians(tilt_deg)[:, None, None],
                   "q_ratio": f["q_ratio"], "ramp": np.round(f["onset_s"] * ftc.CTRL_HZ),
                   "vel": sp[:, None] * np.stack([np.cos(va), np.sin(va), np.zeros(n)], -1), "wz": f["wz"],
                   "yaw": r.uniform(-np.pi, np.pi, n)}
    cols |= {kk: np.asarray(v, np.float32) for kk, v in rand.items()}  # scalars stay scalars (reset_where broadcasts)
    crashed, xy, peak, alt = rollout(params, cols)
    X = np.stack(list(f.values()), -1)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    beta = logistic(Z, crashed.astype(float))
    rank = sorted(zip(f, beta[1:]), key=lambda t: -abs(t[1]))
    print(f"\nMonte Carlo: {n} episodes, crashed {crashed.mean():.1%}; xy RMS of survivors {xy[~crashed].mean():.2f} m")
    print("logistic regression on standardised factors (log-odds of a crash per 1 std; + = raises crash risk):")
    for name, bt in rank:
        lo, hi = np.percentile(f[name], [10, 90])
        cr_lo, cr_hi = crashed[f[name] <= lo].mean(), crashed[f[name] >= hi].mean()
        print(f"  {name:14s} {bt:+.2f}   crash rate lowest 10 % {cr_lo:.0%}, highest 10 % {cr_hi:.0%}")
    return {"crash_rate": float(crashed.mean()), "coef": {nm: float(bt) for nm, bt in rank},
            "factors": {nm: v.tolist() for nm, v in f.items()}, "crashed": crashed.tolist()}


def logistic(Z, y, iters=50, l2=1e-3):
    """Newton's method for L2-regularised logistic regression; returns [intercept, coefficients]."""
    A = np.concatenate([np.ones((len(Z), 1)), Z], 1)
    beta = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ beta))
        H = A.T @ (A * (p * (1 - p))[:, None]) + l2 * np.eye(A.shape[1])
        beta -= np.linalg.solve(H, A.T @ (p - y) + l2 * beta)
    return beta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--mode", default="both", choices=["sweep", "mc", "both"])
    ap.add_argument("--mc-n", type=int, default=4000)
    ap.add_argument("--mc-scale", type=float, default=1.0, help="1 = brutal ranges, 0.5 = halfway to nominal")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params = pickle.load(open(a.policy, "rb"))
    out = {}
    if a.mode in ("sweep", "both"):
        out["sweep"] = sweep(params)
    if a.mode in ("mc", "both"):
        out["mc"] = mc(params, a.mc_n, scale=a.mc_scale)
    if a.out:
        json.dump(out, open(a.out, "w"))
