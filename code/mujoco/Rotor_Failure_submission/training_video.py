#!/usr/bin/env python3
"""Video of a policy learning: for snapshots along one training run, fly the snapshot (rotor --motor fails at 3 s) and
show it next to the reward / crash-rate curve with a cursor at that snapshot's step.
Needs a run trained with --snaps, e.g.
  python train.py --id viz_m0 --steps 6e7 --envs 2048 --motor 0 --p-fault 0.5 --snaps
  python training_video.py runs/viz_m0            # -> runs/viz_m0/training.mp4
Uses the plain training sim (no Gazebo gap), like the run itself.
"""
import argparse
import pickle
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ftc
import train

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--clips", type=int, default=16)
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--motor", type=int, default=0)
a = ap.parse_args()
out = a.out or a.run / "training.mp4"

rows = [re.findall(r"([\d.]+)M steps.*reward ([\d.-]+).*crash ([\d.]+)", ln) for ln in open(a.run / "log.txt")][1:]
steps, reward, crash = np.array([r[0] for r in rows if r], float).T
snaps = sorted((a.run / "snaps").glob("*.pkl"))
assert snaps, f"no snapshots in {a.run}/snaps (train with --snaps)"
snap_steps = np.array([int(p.stem) for p in snaps]) / 1e6
# log-spaced picks: most of the learning happens early
want = np.geomspace(snap_steps[0], snap_steps[-1], a.clips)
picks = sorted({int(np.abs(snap_steps - w).argmin()) for w in want})

W, H, EVERY = 640, 480, 2  # every 2nd 100 Hz step = 50 fps
sim = ftc.make_sim(1)
sim_step = sim.build_step_fn()
k = jnp.ones((1, 4)).at[0, a.motor].set(0.0)
fail_step = round(ftc.T_FAIL * ftc.CTRL_HZ)
e0 = ftc.reset_where(ftc.init(sim), jnp.ones(1, bool), ftc.TARGET[None], jnp.array([fail_step]), k)
step = jax.jit(lambda e, p: ftc.step(sim_step, e, train.policy_mean(p, ftc.observe(e))))
cam = {"lookat": np.asarray(ftc.TARGET), "distance": 4.0, "elevation": -20.0, "azimuth": 135.0}


def curve(at, caption):
    fig, ax = plt.subplots(2, 1, figsize=(W / 100, H / 100), dpi=100, sharex=True)
    ax[0].plot(steps, reward, color="tab:blue")
    ax[0].set_ylabel("mean reward")
    ax[1].plot(steps, crash, color="tab:red")
    ax[1].set_ylabel("crash share")
    ax[1].set_xlabel("training steps (M)")
    for x in ax:
        x.axvline(at, color="k", lw=1.5)
        x.grid(alpha=0.3)
    fig.suptitle(caption, fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# system ffmpeg reads raw frames on stdin (imageio has no ffmpeg backend in this venv)
vid = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W}x{H}",
                        "-r", str(ftc.CTRL_HZ // EVERY), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                       stdin=subprocess.PIPE)
for j in picks:
    params = pickle.load(open(snaps[j], "rb"))
    e, frames, crashed_at = e0, [], None
    for i in range(round(a.seconds * ftc.CTRL_HZ)):
        if crashed_at is None:
            e, _, crashed = step(e, params)
            if bool(crashed[0]):
                crashed_at = (i + 1) / ftc.CTRL_HZ
        if i % EVERY == 0:
            if crashed_at is None or not frames:
                sim.data = e.data
                frame = sim.render(mode="rgb_array", width=W, height=H, cam_config=cam)
            frames.append(frame)  # after a crash the last frame is held
    status = f"crashed at {crashed_at:.1f} s" if crashed_at else f"rotor {a.motor} lost at {ftc.T_FAIL:.0f} s, still flying"
    right = curve(snap_steps[j], f"{snap_steps[j]:.1f}M steps: {status}")
    print(f"{snap_steps[j]:7.1f}M  {status}")
    for f in frames:
        vid.stdin.write(np.hstack([f, right]).tobytes())
vid.stdin.close()
assert vid.wait() == 0, "ffmpeg failed"
sim.close()
print(f"wrote {out}")
