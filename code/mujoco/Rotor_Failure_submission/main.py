#!/usr/bin/env python3
"""Watch the fault-tolerant policy: the drone (ftc.DRONE) hovers at ftc.TARGET, rotor --motor dies at 3 s, the policy
recovers into a spinning hover. The policy is never told which rotor failed. Runs until Ctrl+C.
  python main.py [--motor 0..3] [--t-fail 3.0] [--policy runs/i025a/policy.pkl] [--nominal]
Runs the Gazebo-gap model (ftc.GZ_EVAL) by default; --nominal shows the plain training sim.
"""
import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import ftc
import train

ap = argparse.ArgumentParser()
ap.add_argument("--motor", type=int, default=0)
ap.add_argument("--policy", default=str(Path(__file__).parent / "runs/i025a/policy.pkl"))  # x500v2, + partial loss, exactly symmetric (bench round 4)
ap.add_argument("--nominal", action="store_true", help="plain sim instead of the Gazebo-gap model")
ap.add_argument("--t-fail", type=float, default=ftc.T_FAIL, help="seconds of healthy hover before the rotor fails")
a = ap.parse_args()

sim = ftc.make_sim(1, gz=not a.nominal)
sim_step = sim.build_step_fn()
params = pickle.load(open(a.policy, "rb"))
if ftc.DRONE != "hb_x500" and "runs/i008/" in a.policy:
    print(f"warning: {a.policy} was trained on hb_x500 (old motor order), but ftc.DRONE = {ftc.DRONE}")
k = jnp.ones((1, 4)).at[0, a.motor].set(0.0)
e = ftc.reset_where(ftc.init(sim), jnp.ones(1, bool), ftc.TARGET[None], jnp.array([round(a.t_fail * ftc.CTRL_HZ)]), k,
                     None if a.nominal else ftc.GZ_EVAL)
step = jax.jit(lambda e: ftc.step(sim_step, e, train.policy_mean(params, ftc.observe(e))))

# Viewer only: crazyflow renders the props as fixed meshes, so spin them from rotor_vel ourselves.
# Spin direction is minus mixing_matrix row 3 (the rotor turns against its yaw reaction torque).
# shortcut: true speed (~74k RPM) aliases at 50 fps, so props show a slowed rate; scale is visual only.
SPIN = -np.asarray(sim.data.params.mixing_matrix).reshape(-1, 3, 4)[0, 2] * 1e-5  # rad per frame per RPM
props = [sim.mj_model.body(f"prop{i}:0").id for i in range(4)]
prop_quat0 = sim.mj_model.body_quat[props].copy()
prop_angle = np.zeros(4)

every = 2  # render every 2nd 100 Hz step = 50 fps
fail_step, i, crashed_at = round(a.t_fail * ftc.CTRL_HZ), 0, None
print(f"motor {a.motor} fails at t = {a.t_fail:.1f} s; runs until Ctrl+C")
try:
    while True:
        start = time.time()
        for _ in range(every if crashed_at is None else 0):  # after a crash the last frame stays on screen
            e, _, crashed = step(e)
            i += 1
            if i == fail_step:
                print(f"t = {a.t_fail:.1f} s: motor {a.motor} failed")
            if bool(crashed[0]):
                crashed_at = i / ftc.CTRL_HZ
                print(f"crashed at t = {crashed_at:.2f} s (frozen, Ctrl+C to quit)")
                break
        if crashed_at is None:
            prop_angle += SPIN * np.asarray(e.data.states.rotor_vel[0, 0])
            for j, b in enumerate(props):
                spin = np.array([np.cos(prop_angle[j] / 2), 0, 0, np.sin(prop_angle[j] / 2)])
                mujoco.mju_mulQuat(sim.mj_model.body_quat[b], prop_quat0[j], spin)
        sim.data = e.data
        sim.render()
        time.sleep(max(0.0, every / ftc.CTRL_HZ - (time.time() - start)))
except KeyboardInterrupt:
    pass
finally:
    p, w = np.asarray(e.data.states.pos[0, 0]), np.asarray(e.data.states.ang_vel[0, 0])
    print(f"\nt = {i / ftc.CTRL_HZ:.2f} s: position error {np.linalg.norm(p - np.asarray(ftc.TARGET)):.3f} m, "
          f"yaw rate {w[2]:.1f} rad/s")
    sim.close()
