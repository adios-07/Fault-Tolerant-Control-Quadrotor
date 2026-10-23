#!/usr/bin/env python3
"""Score a controller on the 20 council units: failed motor m in 0..3 x start seed s in 0..4.

Each unit: start hovering near TARGET (+-0.1 m), rotor m fails completely at t = 3 s, 15 s episode.
Score = RMS position error over t in [8, 15] s; a crash scores 5.0. Lower is better.
  python eval.py --policy mellinger|hover|runs/<id>/policy.pkl --out runs/<id>/scores.json [--no-fault] [--gz]
                 [--prop] [--ekf DEG] [--set "{'delay': 2}"]
"""
import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import ftc
from crazyflow.control import Control
from crazyflow.sim import functional as F

MOTORS, SEEDS = 4, 5
N = MOTORS * SEEDS
CRASH_SCORE = 5.0


def units(fault: bool):
    """Unit order is motor-major: world w = SEEDS * m + s."""
    m = np.repeat(np.arange(MOTORS), SEEDS)
    s = np.tile(np.arange(SEEDS), MOTORS)
    pos = np.stack([ftc.TARGET + jax.random.uniform(jax.random.key(int(si)), (3,), minval=-0.1, maxval=0.1)
                    for si in s])
    k = np.ones((N, 4)) if not fault else 1.0 - np.eye(4)[m]
    names = [f"m{mi}_s{si}" for mi, si in zip(m, s)]
    return names, jnp.asarray(pos), jnp.asarray(k), jnp.full(N, int(ftc.T_FAIL * ftc.CTRL_HZ))


def rollout(policy: str, fault: bool, gz: bool = False, extra=None):
    names, pos, k, t_fail = units(fault)
    control = Control.state if policy == "mellinger" else Control.rotor_vel
    assert not (gz and policy == "mellinger"), "--gz delays the rotor_vel command, which Mellinger bypasses"
    sim = ftc.make_sim(N, control, gz=gz)
    sim_step = sim.build_step_fn()
    e = ftc.reset_where(ftc.init(sim), jnp.ones(N, bool), pos, t_fail, k, (ftc.GZ_EVAL if gz else ftc.NOMINAL) | (extra or {}))

    if policy == "mellinger":  # crazyflow's stock controller, given the hover target (same as main.py)
        cmd = jnp.zeros((N, 1, 16)).at[..., :3].set(ftc.TARGET).at[..., 12].set(1.0)

        def ctrl_step(e, _):
            k_now = jnp.where((e.t >= e.t_fail)[:, None], e.k, 1.0)
            data = ftc.apply_fault(F.state_control(e.data, cmd), e.nom_t, e.nom_q, k_now)
            data = sim_step(data, ftc.SUB)
            s = data.states
            crashed = ftc.is_crashed(s.pos[:, 0], ftc.rotmat(s.quat[:, 0]), s.ang_vel[:, 0])
            return e._replace(data=data, t=e.t + 1), (s.pos[:, 0], crashed)
    else:
        if policy == "hover":
            act = lambda obs: jnp.zeros((N, ftc.ACT_DIM))  # open-loop hover thrust
        else:
            import train
            params = pickle.load(open(policy, "rb"))
            act = lambda obs: train.policy_mean(params, obs)

        def ctrl_step(e, _):
            e, _, crashed = ftc.step(sim_step, e, act(ftc.observe(e)))
            return e, (e.data.states.pos[:, 0], crashed)

    steps = int(ftc.T_END * ftc.CTRL_HZ)
    _, (pos_t, crashed_t) = jax.jit(lambda e: jax.lax.scan(ctrl_step, e, length=steps))(e)
    return names, np.asarray(pos_t), np.asarray(crashed_t)


def score(pos_t, crashed_t):
    err = np.linalg.norm(pos_t - np.asarray(ftc.TARGET), axis=-1)  # (T, N)
    window = slice(int(8 * ftc.CTRL_HZ), None)
    rms = np.sqrt(np.mean(err[window] ** 2, axis=0))
    return np.where(crashed_t.any(0), CRASH_SCORE, rms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="mellinger")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fault", action="store_true")
    ap.add_argument("--gz", action="store_true", help="Gazebo-gap model at fixed settings (ftc.GZ_EVAL)")
    ap.add_argument("--prop", action="store_true", help="prop loss: the failed rotor keeps turning")
    ap.add_argument("--ekf", type=float, default=0.0, help="EKF attitude error after the fault [deg], ramping at --ekf-rate")
    ap.add_argument("--ekf-rate", type=float, default=30.0, help="deg/s")
    ap.add_argument("--set", default="{}", help="more reset settings, a dict literal, e.g. \"{'delay': 2}\"")
    a = ap.parse_args()
    import ast
    extra = {"spin": float(a.prop), "att_max": np.radians(a.ekf), "att_rate": np.radians(a.ekf_rate)} | ast.literal_eval(a.set)
    names, pos_t, crashed_t = rollout(a.policy, not a.no_fault, a.gz, extra)
    sc = score(pos_t, crashed_t)
    res = {n: round(float(v), 5) for n, v in zip(names, sc)}
    crash_t = np.where(crashed_t.any(0), crashed_t.argmax(0) / ftc.CTRL_HZ, np.nan)
    print(f"{a.policy} fault={not a.no_fault}{' gz' if a.gz else ''}{' prop' if a.prop else ''}{f' ekf {a.ekf:g} deg' if a.ekf else ''} {a.set if a.set != '{}' else ''}: mean {sc.mean():.3f}  crashed {int(crashed_t.any(0).sum())}/{N}"
          f"  first crash t={np.nanmin(crash_t) if np.isfinite(crash_t).any() else '-'} s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
