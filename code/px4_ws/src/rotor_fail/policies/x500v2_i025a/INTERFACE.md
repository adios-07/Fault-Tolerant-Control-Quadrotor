# Passive fault-tolerant low-level policies for the x500v2 (Holybro X500 v2)

Learned low-level controllers that keep the quadrotor hovering after one rotor fails completely, without
being told which rotor failed. They were trained in crazyflow on drone `x500v2`, which matches the Gazebo model
`PX4_WS/src/rotor_fail/models/custom_x500`. The interface follows Chen et al. 2025 (arXiv 2503.02649) and
github.com/HITSZcjh/uav_ftc. Training code is `../train.py`, the environment is `../ftc.py`, and the numbers are in
`../runs/EXPERIMENTS.md`.

## Which policy

**For PX4/Gazebo use `x500v2_i025a`.** It is i023b plus partial rotor loss (30-80 % thrust) and an exactly
symmetric structure: every motor fault gives the same result. i023b is i022c (corrected Gazebo-gap model: Gazebo's 0.1x rotor inertia, PX4's
logged reaction before the handover, prop loss, faults in forward/climb/descent/yaw flight) retrained against the
bench's onboard estimator (gyro-integrated attitude with scale/bias/heading errors, GPS-aided KF) and against airframe
errors (mass, CG, thrust curves, battery sag, motor lag, inertia). In our replica of the bench (`../bench_sim.py`)
i025a has 0 crashes for complete stop, prop loss and partial loss in every flight phase, peak tilt 70-82 deg, and
0.07-0.13 m error (0.4-0.7 m at 65-80 % partial). Details: `x500v2_i025a/results.txt`.

| folder | bench replica, GT state: crashes / peak tilt / xy error | phases (hover, forward, climb, descent, yaw): crashes per 20 | EKF error tolerated | use |
|---|---|---|---|---|
| `x500v2_i025a` | 0 / 72 deg / 0.12-0.13 m | 0, 0, 0, 0, 0 (also partial 30-80 %) | onboard estimator: bias 4 deg/s, heading offset 20 deg | **PX4/Gazebo** |
| `x500v2_i023b` | 0 / 73 deg / 0.11-0.12 m | 0, 0, 0, 0, 0 | onboard estimator: bias 4 deg/s, heading offset 20-30 deg; EKF2 not tested | superseded (no partial loss, not symmetric) |
| `x500v2_i022c` | 0 / 76 deg / 0.11-0.13 m | 0, 0, 0, 0, 0 | ~10 deg (20 deg: 4/20 crash) | superseded (bench round 4) |
| `x500v2_i018` | 0 / 88 deg / 0.35 m (orbit 0.34 m) | 0, 7-8, 20, 0, 10 | none (10 deg: 20/20) | superseded (bench round 2) |
| `x500v2_i012` | - | - | - | plain-sim reference only |
| `x500v2_i014b`, `x500v2_i016` | - | - | - | superseded |

i012/i014b/i016 were trained before the Gazebo-gap model existed. Their plain-sim numbers above use the corrected
rotor model (physical rotor speed, Gazebo rotor inertia). Details are in `x500v2_i018/results.txt` and
`../runs/EXPERIMENTS.md`.

Each folder has `policy.onnx` (the deployable network), `policy.pkl` (the same weights for this repo) and its
scores.

## Architecture

```
 state estimate ──► high-level controller (code) ──► [p_des, q_des, a_z_des]
                     PID position + primary axis           │
                                                           ▼
 state, IMU, filters ─────────────► 27-d observation ──► policy.onnx ──► a ∈ [-1,1]^4 ──► thrust [N] ──► ω [rad/s]
```

`../pftc_ref.py` implements everything around `policy.onnx` in numpy (about 150 lines), in the same order as training.
**Port that file into the ROS2 node** rather than re-deriving it. It was checked in closed loop: crazyflow driven only
by `pftc_ref.PFTC` and onnxruntime reproduces the JAX policy exactly, including the bench path below (i018 under the
Gazebo-gap dynamics, 80 ms handover: 0.162 m, 0 crashes, the same as JAX).

**Run it in shadow mode from take-off.** Call `track()` every 10 ms while PX4 flies, with the rotor thrusts PX4 actually
applies, and switch the motors to `step()` when the detector fires. A cold start (filters reset) 80 ms into a failure
crashes 20/20 in the sim model. With shadow mode, i023b and i025a survive handovers up to 160 ms (i022c: 1/20 crash at 160 ms).

```python
from pftc_ref import PFTC
pftc = PFTC("policies/x500v2_i025a/policy.onnx")       # floor=0.27 N by default (PX4 armed minimum)
# every 10 ms, before detection: keep the policy's filters current
pftc.track(p, v, q_xyzw, w, target, u_applied)       # u_applied: thrusts PX4 commands now [N], PX4 order
# every 10 ms, after detection: the policy flies
thrust = pftc.step(p, v, q_xyzw, w, acc_b, target)    # N per rotor, PX4 order
omega = PFTC.rotor_speed(thrust)                      # rad/s per rotor for Gazebo
```

## Conventions

- World frame: z up. Body frame: x forward, y left, z up.
- `p`, `v`: position and velocity in the world frame, m and m/s.
- `q_xyzw`: body-to-world quaternion, x y z w order.
- `w`: body angular rates, rad/s.
- `acc_b`: specific force in the body frame, m/s². This is what an IMU reads: +9.81 on z at rest. Training used
  R^T((v_k - v_{k-1})/dt + [0, 0, 9.81]).
- Rate: 100 Hz for the policy and the high-level controller (dt = 0.01 s). Training physics ran at 500 Hz.
- Motor order is PX4's:
  - 0 = front-right (+0.177, -0.177), CCW
  - 1 = back-left (-0.177, +0.177), CCW
  - 2 = front-left (+0.177, +0.177), CW
  - 3 = back-right (-0.177, -0.177), CW

## Observation (27 floats, float32, in this order and scaling)

| idx | signal | scaling |
|---|---|---|
| 0-2 | p - target | / 5 |
| 3-5 | v | / 5 |
| 6-9 | quaternion **w x y z**, sign chosen so w ≥ 0 | 1 |
| 10-12 | body rates wx, wy, wz | / [5, 5, 20] |
| 13-16 | low-passed last rotor thrusts u (tau 0.025 s), N | / 10 |
| 17-19 | acc_b | / [5, 5, 20] |
| 20-21 | high-level p_des, q_des (low-passed, clipped ±10) | / 5 |
| 22 | wz command (always 0) | - |
| 23 | high-level a_z_des (low-passed, clipped ±50) | / 20 |
| 24-26 | filtered angular acceleration, (w - w_prev + 0.12·prev)/(0.12 + dt) | / 50 |

No fault information is in the observation. The policy infers the fault from how the drone responds.

## High-level controller (uav_ftc, same gains)

- Position: v_target = clip(target - p, ±10), with kp_pos = 1. Then acc = [2, 2, 6]·(v_target - v), with horizontal
  acc limited to a 10° tilt and vertical acc clipped to ±5 m/s².
- Desired thrust direction: n_des = (acc + g ẑ)/‖·‖, and ṅ = clip((n_des - n_last)/dt, ±0.5).
- Primary axis (body n_B = ẑ, kx = ky = 5): h = Rᵀn_des, with h3 = max(h_z, 0.1) as a guard.
  - tmp = -5·h_xy - wz·[h_y, -h_x] - (Rᵀṅ)_xy
  - p_des = tmp_y/h3, q_des = -tmp_x/h3, a_z_des = ‖acc + g ẑ‖
- All three commands are low-passed with bandwidth 50 rad/s.

## Action → rotor command

- thrust_i = clip(3.433 + a_i · (11.8 - 3.433), 0.27, 11.8) N. Here 3.433 N is hover thrust (1.40 kg · 9.81 / 4) and
  0.27 N is PX4's armed floor (150 rad/s), which i018 and later are trained with.
- Gazebo MulticopterMotorModel uses F = 1.18e-5 · ω², so ω_i = √(thrust_i / 1.18e-5) rad/s. That gives 539 rad/s at
  hover and 1000 rad/s at max.
- Send ω straight to the motor model's command topic (gz `Actuators` velocity, rotor order 0-3 as above) and bypass
  the PX4 mixer/allocator. The policy *is* the allocator.

## Changing the hover point

Nothing in the policy is tied to 1.5 m. The observation and the high-level controller only use `p - target`, so pass
any `target`. This was checked: at target [1, 1, 2.5] the closed-loop score is identical (0.617 m, 0 crashes).

Limits:
- Training starts were within ±0.5 m of the target, so ramp setpoint changes instead of jumping more than about 0.5 m.
- Heading is not controlled. After a failure the drone must spin in yaw (15-40 rad/s), because 3 rotors cannot
  cancel the yaw torque. That spin is expected.

## Known gaps before trusting a Gazebo result

Now in training (i025a): partial loss 30-80 %, speed-dependent rotor drag with yaw damping, rotor inertia 0.08-1.2x the physical
2.65e-5 kg·m² (Gazebo's effective value is 0.1x because of rotorVelocitySlowdownSim = 10), actuation delay 0-20 ms,
estimate lag 0-20 ms and noise, the 0.27 N floor, prop loss (rotor still turning) and dead motor, PX4's logged
reaction for 0-120 ms before the handover, faults in forward/climb/descent/yaw flight at any heading, and an EKF-like
the bench onboard estimator's errors (gyro scale/bias drift, heading offset, delayed GPS in a KF), and airframe errors
(mass ±10 %, CG 0-2 cm, thrust curves ±10 %, sag to 85 %, motor lag 0.8-2x, J ±20 %). Still open:
- **Heading drift:** heading error must stay below about 20-30° (the high-level controller rotates its position
  correction by it). A 1 % gyro z scale error drifts heading ~11°/s during the spin; the estimator needs heading aiding.
- **CG toward the failed rotor:** an offset of about 1 cm or more in that direction crashes about 1 in 5. Keep the CG
  within about 0.5 cm of the motor centre.
- **PX4's EKF2 during the spin** is not usable as the state source (the bench's own estimator replaces it).
- **Detection margin:** handovers up to 160 ms survive in the sim model.
- **Parameters:** mass 1.40 kg and 11.8 N max thrust are estimates. J is PX4's x500 scaled to 1.40 kg. Rotor lag is
  12.5 ms up / 25 ms down, the same as Gazebo.

## Re-export

```
myenv/bin/python Rotor_Failure/export_onnx.py Rotor_Failure/runs/<id>/policy.pkl <out>/policy.onnx
```
The exporter checks ONNX against JAX on 12,000 closed-loop observations and asserts a difference below 1e-5.
Measured: 4.2e-7 (i012, i014b, i016), 1.3e-6 (i018), 1.1e-6 (i022c, i023b, i025a). Opset 17, IR version 8. Input `obs` [N, 27], output `action` [N, 4].
