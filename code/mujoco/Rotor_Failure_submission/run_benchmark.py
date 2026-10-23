#!/usr/bin/env python3
"""Task 1d baseline benchmark: one rotor of the x500v2 is cut mid-hover, two controllers fly the same scenario.

  python run_benchmark.py [--motor 0] [--t-fail 3.0] [--prop] [--out report]

- "Mellinger": crazyflow's stock geometric controller (position + attitude incl. yaw, x500v2 gains copied from the
  hb_x500 set), the stand-in for a stock PX4-style cascade that keeps controlling yaw. It is not fault-aware.
- "PFTC i025a": the learned passive fault-tolerant controller (uav_ftc high level + policy, runs/i025a/policy.pkl).
Both start hovering at ftc.TARGET (1.5 m), rotor --motor fails completely at --t-fail, 15 s in total, plain crazyflow
x500v2 model with Gazebo's speed-dependent rotor drag (its yaw damping sets the spin rate) and otherwise ftc.NOMINAL:
no actuation delay, exact state, physical rotor inertia, no thrust floor.
Writes <out>/benchmark_kpis.md (KPI table) and <out>/figures/benchmark_<case>.png (position, attitude, body rates,
rotor thrusts vs time, failure marked).
"""
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ftc  # noqa: E402
import train  # noqa: E402
from crazyflow.control import Control  # noqa: E402
from crazyflow.sim import functional as F  # noqa: E402

T_END = ftc.T_END
RECOVER_R = 0.5  # m: "recovered" = position error below this for the rest of the run


def run(name, motor, t_fail, spin, policy):
    control = Control.state if name == "Mellinger" else Control.rotor_vel
    sim = ftc.make_sim(1, control, gz=True, delay=name != "Mellinger")  # Gazebo rotor drag (yaw damping), no delay
    sim_step = sim.build_step_fn()
    k = jnp.ones((1, 4)).at[0, motor].set(0.0)
    e = ftc.reset_where(ftc.init(sim), jnp.ones(1, bool), ftc.TARGET[None], jnp.array([round(t_fail * ftc.CTRL_HZ)]), k,
                        {"spin": float(spin)})
    cmd = jnp.zeros((1, 1, 16)).at[..., :3].set(ftc.TARGET).at[..., 12].set(1.0)  # Mellinger: hover at TARGET, yaw 0

    def ctrl(e, _):
        if name == "Mellinger":
            k_now = jnp.where((e.t >= e.t_fail)[:, None], e.k, 1.0)
            data = ftc.apply_fault(F.state_control(e.data, cmd), e.nom_t, e.nom_q, k_now)
            e = e._replace(data=sim_step(data, ftc.SUB), t=e.t + 1)
        else:
            e, _, _ = ftc.step(sim_step, e, train.policy_mean(policy, ftc.observe(e)))
        s, c = e.data.states, e.data.params.rpm2thrust[:, 0]  # (1, 4, 3) per-rotor thrust curve incl. the fault
        r = s.rotor_vel[:, 0]
        thrust = c[..., 0] + c[..., 1] * r + c[..., 2] * r ** 2
        return e, (s.pos[0, 0], s.quat[0, 0], s.ang_vel[0, 0], thrust[0])

    out = jax.jit(lambda e: jax.lax.scan(ctrl, e, length=int(T_END * ftc.CTRL_HZ)))(e)[1]
    return [np.asarray(x) for x in out]


def euler(q):
    """Roll, pitch, yaw [deg] (ZYX) and tilt [deg] from xyzw quaternions."""
    x, y, z, w = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    return np.degrees(np.stack([roll, pitch, yaw, tilt], -1))


def kpis(t, p, q, w, t_fail):
    """KPI definitions (README, Task 1d / Track A). Values after a crash are excluded."""
    tilt = euler(q)[:, 3]
    crash = (p[:, 2] < 0.1) | (tilt > 90.0) | ~np.isfinite(p).all(-1)
    crash_t = t[np.argmax(crash)] if crash.any() else None
    after = (t >= t_fail) & (np.cumsum(crash) == 0)
    i0 = np.searchsorted(t, t_fail)
    err = np.linalg.norm(p - np.asarray(ftc.TARGET), axis=-1)
    bad = np.flatnonzero(after & (err >= RECOVER_R))
    t_rec = None if crash_t is not None else (0.0 if bad.size == 0 else (t[bad[-1] + 1] - t_fail if bad[-1] + 1 < len(t) else None))
    last5 = t >= T_END - 5.0
    return {"crash at [s]": "-" if crash_t is None else f"{crash_t:.2f}",
            "altitude loss [m]": p[i0, 2] - p[after, 2].min(),
            "max tilt [deg]": "> 90" if crash_t is not None else tilt[after].max(),
            "max horizontal drift [m]": np.linalg.norm(p[after, :2] - p[i0, :2], axis=-1).max(),
            "xy RMS last 5 s [m]": "-" if crash_t is not None else np.sqrt(np.mean(np.sum((p[last5, :2] - p[i0, :2]) ** 2, -1))),
            "time to recover [s]": "never" if t_rec is None else t_rec,
            "spin rate last 5 s [rad/s]": "-" if crash_t is not None else np.abs(w[last5, 2]).mean()}


def plot(t, results, t_fail, path, title):
    fig, ax = plt.subplots(4, len(results), figsize=(6.5 * len(results), 11), sharex=True, squeeze=False)
    for j, (name, (p, q, w, f)) in enumerate(results.items()):
        e = euler(q)
        crash = (p[:, 2] < 0.1) | (e[:, 3] > 90.0)
        if crash.any():  # nothing after the crash is flight: blank it
            p, e, w, f = (np.where(np.arange(len(t))[:, None] > np.argmax(crash), np.nan, x) for x in (p, e, w, f))
        rows = [(p, ["x", "y", "z"], "position [m]"), (e[:, [0, 1, 3]], ["roll", "pitch", "tilt"], "attitude [deg]"),
                (w, ["p (x)", "q (y)", "r (z)"], "body rate [rad/s]"), (f, [f"rotor {i}" for i in range(4)], "rotor thrust [N]")]
        for i, (y, labels, ylab) in enumerate(rows):
            a = ax[i, j]
            for k_, lab in enumerate(labels):
                a.plot(t, y[:, k_], lw=1, label=lab)
            a.axvline(t_fail, color="k", ls="--", lw=1)
            a.set_ylabel(ylab)
            a.grid(alpha=0.3)
            a.legend(loc="upper right", fontsize=8, ncol=4)
            if i == 1:
                a.set_ylim(-180, 180)
        ax[0, j].set_title(f"{name}: {title}")
        ax[-1, j].set_xlabel("time [s] (dashed: rotor failure)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=0, help="failed rotor, PX4 order (0 front-right CCW)")
    ap.add_argument("--t-fail", type=float, default=ftc.T_FAIL)
    ap.add_argument("--prop", action="store_true", help="propeller loss (rotor keeps spinning) instead of a stopped motor")
    ap.add_argument("--policy", default=str(Path(__file__).parent / "runs/i025a/policy.pkl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    policy = pickle.load(open(a.policy, "rb"))
    case = f"motor{a.motor}_{'prop' if a.prop else 'stop'}"
    t = (np.arange(int(T_END * ftc.CTRL_HZ)) + 1) / ftc.CTRL_HZ
    res = {n: run(n, a.motor, a.t_fail, a.prop, policy) for n in ("Mellinger", "PFTC i025a")}
    title = f"rotor {a.motor} {'propeller lost' if a.prop else 'stopped'} at {a.t_fail:g} s"
    plot(t, res, a.t_fail, out / "figures" / f"benchmark_{case}.png", title)
    rows = {n: kpis(t, *r[:3], a.t_fail) for n, r in res.items()}
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = [f"Benchmark ({title}; run_benchmark.py, x500v2 + Gazebo rotor drag, hover at 1.5 m)", "",
             "| KPI | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    lines += [f"| {k} | " + " | ".join(fmt(r[k]) for r in rows.values()) + " |" for k in next(iter(rows.values()))]
    md = out / "benchmark_kpis.md"
    old = md.read_text().split("\n\n\n") if md.exists() else []
    blocks = [b for b in old if title not in b and b.strip()] + ["\n".join(lines)]
    md.write_text("\n\n\n".join(blocks) + "\n")
    print("\n".join(lines))
    print(f"\nplot: {out / 'figures' / f'benchmark_{case}.png'}")
