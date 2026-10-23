#!/usr/bin/env python3
"""Task 1b maths, computed for the x500v2: allocation matrix, why yaw is lost, and the relaxed-hover equilibrium.

  python relaxed_hover.py [--sim] [--cg] [--out report/relaxed_hover.txt]

1. Allocation A: [T, tau_x, tau_y, tau_z] = A f, with rotor 0 dead.
2. Naive three-rotor hover (body level, partner rotor off): the yaw torque left over and the spin rate at which
   Gazebo's rotor-drag yaw damping balances it.
3. Relaxed hover (Mueller & D'Andrea): constant body rates w and thrusts, the body spinning about the world vertical
   e = w/|w|. Unknowns: thrusts of the 3 working rotors and w (6); equations: body torque balance (3) with thrust
   levers about the CG, rotor drag torque, Gazebo rotor-drag damping and gyroscopic terms, and the vertical force
   sum_i f_i (z_B . e) = m g (1). Two free parameters: reported (a) at the least largest thrust and (b) with the two
   thrusts the policy chose in simulation fixed, compared with that simulation (--sim).
4. --cg: largest-thrust equilibrium vs a CG offset (toward / sideways / away from the failed rotor).
Plant: ftc.py / crazyflow x500v2 (physical rotor inertia, as run_benchmark.py).
"""
import argparse
from pathlib import Path

import numpy as np

import ftc  # before scipy: ftc sets SCIPY_ARRAY_API, which crazyflow's jax rotations need
from scipy.optimize import minimize  # noqa: E402

P = ftc._P
M, G = ftc.MASS, ftc.G
J = np.asarray(P["J"])
MIX = np.asarray(P["mixing_matrix"])  # rows: y sign, -x sign, yaw-torque sign per rotor
R = np.asarray(ftc.ROTOR_POS)  # (4, 3) rotor positions, body frame, m
KAPPA = float(np.asarray(P["rpm2torque"])[2] / np.asarray(P["rpm2thrust"])[2])  # drag torque / thrust, m
KF, CD, JP = ftc.MOTOR_CONSTANT, ftc.ROTOR_DRAG, ftc.PROP_J0  # N/(rad/s)^2, rotor drag, kg m^2
FMAX = ftc.F_MAX
FAILED = 0


def allocation():
    return np.vstack([np.ones(4), R[:, 1], -R[:, 0], KAPPA * MIX[2]])


def resid(x, cg=np.zeros(3), jp=JP):
    """Steady relaxed-hover equations; x = [3 working-rotor thrusts, w_x, w_y, w_z]."""
    f = np.insert(x[:3], FAILED, 0.0)
    w = x[3:6]
    om = np.sqrt(np.maximum(f, 0) / KF)  # rotor speeds, rad/s
    tau = np.cross(R - cg, np.stack([np.zeros(4), np.zeros(4), f], -1)).sum(0)  # thrust levers about the CG
    tau[2] += KAPPA * MIX[2] @ f  # rotor drag torques
    v = np.cross(w, R)
    v[:, 2] = 0.0  # Gazebo MulticopterMotorModel drag uses the in-plane rotor velocity
    tau += np.cross(R, -(om * CD)[:, None] * v).sum(0)
    tau += np.cross(w, np.array([0.0, 0.0, jp * MIX[2] @ om]))  # rotor gyroscopic torque (crazyflow torque_inertia)
    tau -= np.cross(w, J @ w)
    e = w / (np.linalg.norm(w) + 1e-12)
    return np.concatenate([tau, [f.sum() * e[2] - M * G]])


def solve(fix=None, cg=np.zeros(3)):
    """Minimise the largest thrust subject to the equilibrium (fix: {working-rotor index 0..2: thrust})."""
    fix = fix or {}
    best = None
    for wz in (15.0, -15.0, 25.0, -25.0):
        x0 = np.array([6.0, 1.0, 6.0, 0.0, 0.0, wz, 8.0])
        for i, v in fix.items():
            x0[i] = v
        bnd = [(fix[i], fix[i]) if i in fix else (0.0, 20.0) for i in range(3)] + [(-60, 60)] * 3 + [(0, 20)]
        res = minimize(lambda x: x[6], x0, method="SLSQP", bounds=bnd,
                       constraints=[{"type": "eq", "fun": lambda x: resid(x[:6], cg)},
                                    {"type": "ineq", "fun": lambda x: x[6] - x[:3]}], options={"maxiter": 500})
        if res.success and np.abs(resid(res.x[:6], cg)).max() < 1e-6 and (best is None or res.x[6] < best.x[6]):
            best = res
    return None if best is None else best.x[:6]


def describe(x):
    f = np.insert(x[:3], FAILED, 0.0)
    w = x[3:6]
    e = w / np.linalg.norm(w)
    return (f"thrusts [N] {np.round(f, 2).tolist()}, body rates [rad/s] {np.round(w, 2).tolist()}, |w| {np.linalg.norm(w):.1f}"
            f" rad/s, spin axis {np.degrees(np.arccos(abs(e[2]))):.1f} deg from body z")


def simulate(t_end=40.0, avg=10.0):
    """i025a in the run_benchmark.py plant, rotor 0 stopped at 3 s: mean thrusts and body rates over the last `avg` s."""
    import pickle
    import run_benchmark as rb
    rb.T_END = t_end
    p, q, w, f = rb.run("PFTC i025a", FAILED, 3.0, False, pickle.load(open(Path(__file__).parent / "runs/i025a/policy.pkl", "rb")))
    last = slice(-int(avg * ftc.CTRL_HZ), None)
    tilt = rb.euler(q[last])[:, 3]
    return f[last].mean(0).astype(float), w[last].mean(0).astype(float), float(tilt.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="compare with i025a's simulated steady state (40 s rollout)")
    ap.add_argument("--cg", action="store_true", help="largest-thrust equilibrium vs CG offset")
    ap.add_argument("--out", default=str(Path(__file__).parent / "report/relaxed_hover.txt"))
    a = ap.parse_args()
    L = []
    A = allocation()
    np.set_printoptions(precision=4, suppress=True)
    L += [f"x500v2: m {M} kg, J diag {np.diag(J).tolist()} kg m^2, arm (x, y) +-{abs(R[0, 0]):.5f} m, kF {KF} N/(rad/s)^2,"
          f" kappa = kM/kF {KAPPA:.4f} m, rotor inertia {JP:.3e} kg m^2, max thrust {FMAX} N",
          "", "1. Allocation [T, tau_x, tau_y, tau_z]^T = A f (rotor order 0 FR CCW, 1 BL CCW, 2 FL CW, 3 BR CW):", str(A),
          f"   rank A = {np.linalg.matrix_rank(A)}; with column {FAILED} zeroed: rank {np.linalg.matrix_rank(np.delete(A, FAILED, 1))}"
          " (3 rotors cannot set 4 independent outputs)"]
    A3 = np.delete(A, FAILED, 1)
    f3 = np.linalg.solve(A3[:3], [M * G, 0.0, 0.0])  # thrust and roll/pitch balance fix the 3 thrusts
    tz = A3[3] @ f3
    om = np.sqrt(np.insert(f3, FAILED, 0.0) / KF)
    damp = CD * np.sum(om * np.sum(R[:, :2] ** 2, -1))  # Gazebo rotor drag: yaw torque per rad/s of body yaw rate
    L += ["", "2. Naive hover on rotors 1-3 (body level): T = m g and tau_x = tau_y = 0 fix all three thrusts:",
          f"   f = {np.round(np.insert(f3, FAILED, 0.0), 3).tolist()} N (the partner rotor 1 must be off),"
          f" leaving tau_z = {tz:.3f} N m, so the yaw rate grows at {tz / J[2, 2]:.1f} rad/s^2.",
          f"   Rotor-drag yaw damping at these speeds: {damp:.4f} N m/(rad/s) -> steady spin ~ {abs(tz) / damp:.1f} rad/s."]
    x = solve()
    L += ["", "3a. Relaxed hover, least largest thrust: " + describe(x)]
    if a.sim:
        f_sim, w_sim, tilt_sim = simulate()
        fix = {0: f_sim[1], 1: f_sim[2]}  # working rotors 1 and 2 at their simulated thrusts
        xs = solve(fix)
        L += ["3b. Simulated i025a (run_benchmark.py plant, mean of the last 10 s of a 40 s run): "
              f"thrusts [N] {np.round(f_sim, 2).tolist()}, body rates {np.round(w_sim, 2).tolist()} rad/s,"
              f" |w| {np.linalg.norm(w_sim):.1f} rad/s, mean tilt {tilt_sim:.1f} deg",
              "    Equilibrium with rotors 1 and 2 fixed at those thrusts: " + ("no solution" if xs is None else describe(xs)),
              f"    Residual of the equations at the simulated mean state: "
              f"{np.round(resid(np.concatenate([f_sim[1:], w_sim])), 3).tolist()} (N m x3, N)"]
    if a.cg:
        L += ["", "4. Least-largest-thrust equilibrium vs CG offset (cm along the direction to the failed rotor; max thrust):"]
        d = R[FAILED, :2] / np.linalg.norm(R[FAILED, :2])
        for lab, ang in (("toward failed", 0.0), ("sideways", np.pi / 2), ("away", np.pi)):
            row = []
            for cm in (0.0, 1.0, 2.0, 3.0, 5.0):
                c, s = np.cos(ang), np.sin(ang)
                xc = solve(cg=np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], 0.0]) * cm / 100)
                row.append("none" if xc is None else f"{max(xc[:3]):.1f} N (partner {xc[0]:.2f})")
            L.append(f"   {lab:14s} " + " | ".join(f"{cm:g} cm: {r_}" for cm, r_ in zip((0, 1, 2, 3, 5), row)))
    txt = "\n".join(L)
    print(txt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(txt + "\n")
