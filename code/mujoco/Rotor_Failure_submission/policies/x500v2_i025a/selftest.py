#!/usr/bin/env python3
"""Replay self-test: feeds a recorded episode (selftest_episode.npz) into pftc_ref.PFTC and checks that every
rotor thrust matches the recording. Proves the local numpy + onnxruntime stack reproduces the training-side
controller exactly; a C++/ROS2 port can be checked the same way against the npz arrays.

Episode: x500v2 hovering at [0, 0, 1.5], motor 1 (back-left, PX4 motor 2) fails completely at t = 3.00 s,
PX4 (stand-in: its reaction logged on the bench, failed rotor to max and its partner to the floor) flies until
t = 3.08 s while PFTC runs in shadow mode (track), then PFTC flies (step) until t = 15 s. Gazebo-gap dynamics.
    python3 selftest.py            # needs numpy, onnxruntime
"""
from pathlib import Path

import numpy as np

from pftc_ref import PFTC

here = Path(__file__).parent
d = np.load(here / "selftest_episode.npz")
pftc = PFTC(str(here / "policy.onnx"))
worst = 0.0
for t in range(len(d["p"])):
    args = (d["p"][t], d["v"][t], d["q_xyzw"][t], d["w"][t])
    if d["shadow"][t]:
        pftc.track(*args, d["target"], d["u_applied"][t])
        thrust = d["u_applied"][t]
    else:
        thrust = pftc.step(*args, d["acc_b"][t], d["target"])
    worst = max(worst, float(np.abs(thrust - d["thrust"][t]).max()))
handover = int(np.argmin(d["shadow"]))
print(f"{len(d['p'])} ticks, handover at tick {handover}, max |thrust - recorded| = {worst:.2e} N")
assert worst < 1e-4, "outputs differ from the recording: check onnxruntime/numpy versions or the port"
print("PASS")
