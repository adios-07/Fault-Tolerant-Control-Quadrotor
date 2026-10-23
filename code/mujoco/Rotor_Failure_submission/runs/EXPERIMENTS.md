# Experiment log (one row per run, failures too)

Metric: 20 units (failed motor 0-3 x 5 start seeds), complete failure at t=3 s, RMS position error over
t in [8,15] s, crash = 5.0, lower is better. Arbiter: `--lower-is-better --noise 0.05` (noise not yet measured).
Variant: hb_x500-FTC (thrust 0-17 N). Training on CPU, taskset 24 cores, ~170k steps/s.

| run | card | config | steps | mean score | units better | verdict | note |
|---|---|---|---|---|---|---|---|
| baseline_mellinger | IDEA-001 | crazyflow Mellinger state ctrl | - | 5.000 | parent | - | healthy (no fault) RMS 0.002 m; with fault all 20 crash at ~3.35 s |
| (hover, no file) | IDEA-001 | constant hover thrust | - | 5.000 | - | - | healthy open-loop drift 0.099 m; with fault crash ~3.2 s (injection works) |
| throughput (deleted) | IDEA-002 | 2048 envs x 64, motor 0 dead from t=0 | 3M | - | - | PASS | 156k steps/s end-to-end on CPU (> 100k kill) |
| smoke1 | IDEA-003 | motor 0 dead from t=0, from scratch | 20M | 5.000 | 0/20 | FAILED | survival rising in training (pass smoke kill); crashes at 0.47 s in eval: never saw a healthy drone |
| s1_m0 | IDEA-003 | motor 0, t_fail U(1,4) s, init smoke1 | 30M | 3.769 | 5/20 | NEUTRAL | motor 0 units all ~0.077 m; motors 1-3 crash |
| s2_all | IDEA-003 | random motor, p_fault 0.9, init s1_m0 | 40M | 3.769 | 5/20 | NEUTRAL | stuck: training crash share flat ~0.69, std 0.056; only motor 0 solved. S2 rung FAILED -> card stops |
| i006 | IDEA-006 | random motor from scratch, t_fail U(0,4) s, p_fault 0.9 | 120M | 5.000 | 0/20 | FAILED | training crash share plateau ~0.9 from 30M; eval crashes ~1 s after failure on all units; kill hit at 60M |
| i007 | IDEA-007 | Selector-Controller (4 MLPs + supervised selector), random motor from scratch, t_fail U(0,4) | 90M (stopped) | 3.767 | 5/20 | NEUTRAL | selector acc 1.00 from 5M; crash share 0.88 -> 0.70, slowly improving when stopped (user chose to free CPU for 008) |
| (oracle probe) | IDEA-008 probe | s1_m0 under symmetry transforms, KNOWN failed motor (active FTC, not a valid entry) | - | 0.077 | 20/20 | - | proves transforms correct |
| **i008** | IDEA-008 | symmetric Selector-Controller: frozen s1_m0 x {id, mirror, rot180, both} + supervised selector, t_fail U(1,4), p_fault 0.9 | 30M (selector only) | **0.060** | **20/20** | **PROMOTE** | no crashes; healthy no-fault hover RMS 0.140 m (vs mellinger 0.002); selector acc 1.00 in training |

Robustness of i008 (reviewer probe, 32 units per case, mean RMS m by failed motor 0/1/2/3, crashes):
base 0.060/0.060/0.060/0.060 (0) · obs noise 0.01 0.063/0.062/0.063/0.062 (0) · obs noise 0.03 1.925/0.696/0.698/0.694 (6)
· start +-0.5 m 0.060 all (0) · fault time U(1,6) s 0.156/0.119/0.108/0.068 (0) · partial k=0.5 0.176 all (0)

Healthy-hover fix, explored then dropped (user kept i008 as v1, code reverted):
s1_m0 alone, no fault: 0.212 m (so the 0.140 m of i008 is the base policy, not the mixing) · PPO trained with
no faults (30M): 0.104 m (paper reward barely sees 10 cm: 0.1 x 0.1^2 = 0.001 vs 1.5 alive) · classical rate
controller (paper eq. 5 + 4-rotor mixing) on the same high-level commands: 0.000 m healthy, crashes on fault.
Planned 5-mode selector (healthy expert as mode 4) was stopped before it finished.

Drone switch (after v1): `ftc.DRONE` is now crazyflow `x500v2`, a new drone matched to the Holybro X500 v2 kit and
our Gazebo custom_x500: 1.40 kg, J diag(0.0152, 0.0152, 0.028), 0-11.8 N per rotor (T/W 3.44), torque ratio 0.016,
rotor lag 12.5/25 ms, hover-linearised rotor drag 0.174 N/(m/s) in body x/y, PX4 motor order
(0 FR CCW, 1 BL CCW, 2 FL CW, 3 BR CW). hb_x500 is unchanged. All runs above are hb_x500 (old motor order 0 FR,
1 BR, 2 BL, 3 FL) and are valid only with `DRONE = Drone.hb_x500` (checked: i008 still scores 0.060).
x500v2 has no trained policy yet. Mellinger on x500v2: healthy 0.008 m, with fault 20/20 crash (first at 3.37 s).

x500v2 retrain, v1 recipe unchanged (local CPU, 24 cores, ~190-235k steps/s, ~12 min total):

| run | config | steps | mean score | verdict | note |
|---|---|---|---|---|---|
| x5_smoke | motor 0 dead from t=0, from scratch | 20M | - | - | not evaluated (rung 1 of the ladder) |
| x5_m0 | motor 0, t_fail U(1,4), init x5_smoke | 30M | 3.845 | gate FAILED | motor-0 units 0.382 m (hb_x500 s1_m0: 0.077); spin ~16 rad/s; motors 1-3 crash (expected) |
| x5_i008 | symmetric selector on x5_m0, 30M | 30M | 5.000 | FAILED | 20/20 crash at ~4.95 s; selector correct (w = 1.0 on the failed motor by t=3.3 s); healthy no-fault 0.027 m |
| x5_mellinger | Mellinger baseline on x500v2 | - | 5.000 | - | 20/20 crash, first at 3.37 s; healthy 0.008 m |

Diagnosis: the selector is right, so the failure is the motor-0 base policy, which has a narrow recovery basin on
x500v2. It survives alone, but the ~0.3 s of mixed actions while the selector switches pushes it out, and it saturates
12% of the time after the fault.

| x5_m0b | x5_m0 + 60M, same config (8 cores) | 60M | 3.780 | NEUTRAL | motor-0 units 0.121 m (was 0.382) |
| x5_i008b | selector on x5_m0b | 30M | 5.000 | FAILED | 20/20 crash, first 3.88 s; healthy 0.077 m |
| x5s_smoke | from scratch, seed 1 | 20M | - | stopped | its m0 rung (p_fault 1.0) was killed at 13M once the cause below was found |
| (oracle probe) | x5_m0b symmetric, failed motor known all episode | - | 0.121 | - | 0.121/0.122/0.122/0.122 by motor, 0 crashes: the PX4-order symmetry maps are correct |
| (switch probe) | uniform blend before the fault, one-hot from fault + 0..300 ms | - | 5.000 | - | 20/20 crash even at 0 ms delay |

Cause: with `--p-fault 1.0` and only motor 0, the base policy always sees motor 0 fail within 1-4 s, so it flies in
failure mode on a healthy drone (pre-fault: 23 deg tilt, wz 14 rad/s, motor 0 at 2.5 N). The capped penalty makes
that cheap, and it never learns the flat-hover-to-spin transition. The selector's healthy blend hovers flat
(0 deg, wz 0), so the fault arrives in a state the base policy cannot handle. Fix being tested: x5_m0c =
x5_m0b + 60M with `--p-fault 0.5` (healthy episodes make pre-spinning costly), then selector x5_i008c.
| x5_m0c | x5_m0b + 60M, p_fault 0.5 | 60M | 3.883 | FAILED | motor-0 0.533 m; still pre-spins when healthy (20 deg, wz 15); policy std 0.03, stuck |
| x5_i008c | selector on x5_m0c | 30M | 5.000 | FAILED | 20/20 crash; healthy 0.035 m |

Next: restart rung 2 from x5_smoke (which learned flat-to-spin at t=0) with healthy episodes:
x5_m0d (p_fault 0.5) and x5_m0e (p_fault 0.8), 90M each, in parallel.
| x5_m0d | rung 2 from x5_smoke, p_fault 0.5 | 90M | 2.787 | NEUTRAL | m0 0.609 m, m1 0.540 m (alone); pre-fault healthy: tilt 37 deg, wz 14.5, motor 0 at 0 N; known-motor 0.61 all; blend switch 20/20 crash |
| x5_m0e | rung 2 from x5_smoke, p_fault 0.8 | 90M | 3.860 | NEUTRAL | m0 0.440 m; pre-fault healthy: tilt 40 deg, wz 14, motor 0 at 0 N; known-motor 0.44 all; blend switch 20/20 crash |

Card A (council round 2) gate FAILED on both: healthy episodes alone, with the cap-1 reward, do not stop pre-spinning
(it gets worse: the policy switches motor 0 off itself on a healthy drone). Curriculum-only fixes are ruled out.

Council round 2, local cards (base `<id>_m0` 90M from x5_smoke, selector `<id>` 30M; arbiter vs x5_mellinger):
| run | card | config | m0 units (base alone) | selector system | verdict | pre-fault healthy (base) |
|---|---|---|---|---|---|---|
| i011 | IDEA-011 | ent 0.003, p_fault 0.8 | 0.680 | 20/20 crash (no-fault also 20/20) | FAILED | 40 deg, wz 13.5, motor 0 off; switch at 0 ms: 12/20 crash (motors 1, 2 survive 1.57 m) |
| i009 | IDEA-009 | --reward smooth, p_fault 0.5 | 0.597 | 20/20 crash (no-fault also 20/20) | FAILED | 41 deg, wz 14.2, motor 0 off |
| i010 | IDEA-010 | smooth + t_fail U(2,5) | 1.298 | 20/20 crash; no-fault 0.078 | FAILED | 37-45 deg, wz 12.5 |
Reading: none removed pre-spinning. The smooth reward saturates (reviewer S1-B-2), so the 10x spin weight adds
almost no cost beyond ~1.0/step, the same as the cap-1 reward. All three inherit x5_smoke, which spins from t=0.

IDEA-014, from scratch (no --init), motor 0, t_fail U(1,4), p_fault 0.5, base 120M + selector 30M, 8 cores each:
| run | arm | base m0 alone | known motor (all) | selector system (20 units) | no-fault | verdict vs x5_mellinger |
|---|---|---|---|---|---|---|
| i014a | paper reward | 0.080 | 0.08 all, 0 crash | 2.542 mean, 10/20 crash (m0/m1 2.05, m2/m3 3.03) | 0.025 | NEUTRAL (10/20 better) |
| **i014b** | --reward smooth | 0.636 | 0.64 all, 0 crash | **0.617 on all 4 motors, 0/20 crash** | 0.074 | **PROMOTE** (20/20, +4.38) |
| i014c | paper + ent 0.003 | 1.010 | 1.01 all, 0 crash | 0.939, 0/20 crash | 0.137 | PROMOTE vs mellinger; FAILED vs i014b |
Pre-fault healthy flight is now flat in all three (tilt 0.1-1.4 deg, wz 0.3-0.7): training from scratch removed the
pre-spin habit that every x5_smoke-initialised base had. Switch-delay probe: i014b survives up to 50 ms, i014c up to
200 ms, i014a crashes 6-10/20 at any delay (its uniform pre-fault blend spins at wz +-11 with u 1.4/5.5 N).
First working x500v2 system: i014b, 0.617 m (v1 on hb_x500: 0.060 m). Precision is the remaining gap.

i014a selector fix (user: "fix the i014a selector"):
- Diagnosis: the uniform 4-copy blend before the fault spins (wz +-11, u 1.4/1.4/5.5/5.5 N), because the mirrored
  copies see the opposite yaw sign and their yaw corrections do not cancel; the base alone hovers flat when healthy.
- Probe (zero training): base alone flies until the fault, then one-hot on the failed motor's copy: 0.08 m on all 20
  at 0 ms switch delay; 5/20 crash at 50 ms, 15/20 at 100+ ms (the motor-0 copy flying a motor-1/2/3 fault).
- **i016**: 5-mode selector (existing `--expert` option) with the healthy mode = the i014a base itself, 30M:
  **0.081/0.080/0.079/0.079 by motor, 0/20 crash**; arbiter vs i014b: PROMOTE (+0.537, 20/20 better). No-fault
  hover 0.200 m. The selector puts 0.97-0.99 on the failed motor's copy one control step (10 ms) after the fault.

Fault-time sweep (same 20 units, fault at 3/5/7 s, scored over the last 7 s; user saw i016 fail in the viewer
with --t-fail 5):
| policy | 3 s | 5 s | 7 s |
|---|---|---|---|
| i016 | 0.080, 0 crash | 1.066, 4/20 crash (motors 2, 3) | 1.556, 6/20 crash (motors 2, 3) |
| i014b | 0.617, 0 crash | 0.617, 0 crash | 0.616, 0 crash |
Reading: i016's base flying alone drifts while healthy (0.33 m at 3 s, 0.64 m at 4.9 s), so later faults hit a
drifting drone and the CW-motor (2, 3) faults crash. The 3 s eval hides this. i014b is fault-time robust but
imprecise. Spin after the fault (25-38 rad/s) is the expected relaxed-hover yaw spin, not a defect.
| i015 | i014b_m0 + 120M (same config), selector 30M | 3.121, 8/20 crash (all fault times) | FAILED vs i014b | longer training brought pre-spinning back (healthy: 22 deg, wz 4.7) |

Deployment package (`policies/`, for a ROS2 + Gazebo test): i014b and i016 exported to ONNX by `export_onnx.py`
(max |onnx - jax| 4.2e-7 on 12,000 closed-loop observations). `pftc_ref.py` = numpy port of the high-level
controller, observation builder and action mapping around onnxruntime. Closed loop in crazyflow with only
pftc_ref + ONNX: i014b 0.617 m, i016 0.080 m (identical to eval.py); i014b at target [1, 1, 2.5]: 0.617 m, 0 crashes.

Council round 2, heavy cards on Kaggle T4 x2 (kernel adios07/rotor-failure-heavy, 52 min; T4 ran 17k-32k steps/s,
about 5-10x slower than 8 local cores). Logs: runs/i012.kaggle.log, runs/i013.kaggle.log.
| run | card | config | 20 units (3 s) | fault at 5 s / 7 s | no-fault | verdict |
|---|---|---|---|---|---|---|
| **i012** | IDEA-012 | `--joint` PPO on the symmetric mixture, from x5_i008b, random motor, p_fault 0.5, lr 1e-4, std reset 0.1, 60M | **0.319, 0 crash** | 0.319 / 0.319, 0 crash | 0.088 | **PROMOTE vs i014b** (+0.297, 20/20); FAILED vs i016 at 3 s only |
| i013 | IDEA-013 | healthy expert i013_h (30M) + handoff specialist i013_m0 (90M from x5_smoke) + 5-mode selector (30M) | 0.389, 0 crash | 0.389 / 0.396, 0 crash | 0.174 | PROMOTE vs i014b (+0.228, 20/20); FAILED vs i016 at 3 s only |
i012 is the best fault-time-robust x500v2 policy and the new default (main.py, policies/x500v2_i012, ONNX parity
4.2e-7, pftc_ref closed loop 0.319).

## Gazebo gap (PX4/Gazebo bench: i014b crashed on every motor-1 failure; healthy hover and interface were correct)
Model added to the sim (training/eval only; the 27-d interface is unchanged):
- x500v2 params now physical: rpm2thrust [0, 0, 1.294015e-7] (Gazebo F = 1.18e-5 omega^2, in RPM; hover 539 rad/s),
  rpm2torque 0.016x, prop_inertia 2.65e-5 (Gazebo rotor izz). A failed rotor now stops (command 0).
- `ftc.make_sim(gz=True)` plugins at 500 Hz: actuation delay queue (0-20 ms) and Gazebo MulticopterMotorModel rotor
  drag F_i = -|omega_i| c v_perp_i with the rotor's own velocity (w x r gives yaw damping); fixed drag matrix off.
- Env estimate `est`: true state delayed 0-2 policy steps + noise (EST_SIGMA x scale); high level and obs use it.
- Thrust floor 0.27 N (PX4 armed minimum, 150 rad/s).
- Checks: drag -0.174 N at 1 m/s; yaw damping 0.0109 Nm/(rad/s) healthy, 0.0087 after a failure (bench ~0.008);
  delay holds the command exactly; floor and lag exact. `eval.py --gz` = 10 ms delay, 10 ms lag, noise 0.5, floor.
Old policies under the new model (20 units, fault 3 s):
| policy | nominal (new rotor inertia) | --gz |
|---|---|---|
| i014b | 5.000, 20/20 crash (was 0.617) | 5.000, 20/20 |
| i012 | 0.376, 0 crash (was 0.319) | 5.000, 20/20 |
One effect at a time (gz plugins on, rest nominal), mean / crashes:
| effect | i012 | i016 |
|---|---|---|
| rotor drag (speed-dep. + yaw damping) | 1.080 / 0 | 0.052 / 0 |
| delay 10 ms | 1.116 / 0 | 0.116 / 0 |
| delay 20 ms | 4.635 / 18 | 5.000 / 20 |
| estimate lag 10 ms | 3.057 / 10 | 0.116 / 0 |
| estimate noise x0.5 | 1.154 / 0 | 0.049 / 0 |
| floor 0.27 N | 5.000 / 20 | 0.055 / 0 |
| all (GZ_EVAL) | 5.000 / 20 | 5.000 / 20 |
Reading: the model reproduces the Gazebo crash. i014b fails from rotor inertia alone; i012 relies on near-0 N on a
healthy rotor, which the PX4 floor forbids; i016 tolerates each effect except a 20 ms delay. Retrain: i017 = joint
fine-tune of i012 with --gz-rand, i018 = joint fine-tune of i016 (5-mode) with --gz-rand, 100M each.

Gazebo-gap retrain (joint PPO, --gz-rand, random motor, p_fault 0.5, lr 1e-4, std reset 0.1, 100M, 12 cores each):
| run | init | nominal | --gz | --gz no fault | one effect at a time | bench handover 80 ms (shadow / cold) |
|---|---|---|---|---|---|---|
| i017 | i012 | 0.460, 0 crash | 0.210, 0 crash | 0.102 | 0.23-0.32, 0 crash (drag-only, no floor: 1.04) | 20/20 / 20/20 (survives up to 40 ms) |
| **i018** | i016 (5-mode) | 4.479, 17/20 late crash | **0.130, 0 crash** | 0.313 | 0.16-0.19, 0 crash | **0.148, 0 crash** / 20/20 |
i018 shadow-mode handover: 20 ms 0.129, 40 ms 0.138, 80 ms 0.148 (fault 5 s 0.151, 7 s 0.150), 120 ms 20/20 crash.
Bench protocol in the sim: motors at hover thrust (stand-in for fault-unaware PX4) until the handover; "shadow" =
the policy's filters keep running on the applied commands (new `pftc_ref.PFTC.track`), "cold" = reset at handover.
pftc_ref + ONNX (shadow mode) under gz dynamics: 80 ms 0.162, 0 ms 0.133, 0 crashes, identical to JAX.
Packaged as policies/x500v2_i018 (ONNX parity 1.3e-6); main.py now defaults to i018 on the Gazebo-gap model.

## 2026-10-17: bench round 3 (i018 on PX4 SITL + Gazebo): orbit, prop loss, EKF, flight phases

Bench (8 runs/arm, GT state): i018 survives 8/8, but orbits the hold point (0.30-0.40 m radius, ~0.27 Hz), peaks at
86 deg tilt, xy transient 2.8 m; EKF state: 0/8, crash at 1.9-2.4 s. Logs analysed here (pyulog; logs not shipped):
- **Orbit cause = rotor inertia in the gap model.** Gazebo spins the rotor links at 1/rotorVelocitySlowdownSim = 1/10
  of the rotor speed, so its gyroscopic and spin-up torques are 0.1x ours. With prop_inertia x0.1 the sim reproduces
  the bench's steady state: thrust per motor [7.1 0.3 8.3 5.2] N (bench [7.4 0.4 8.7 4.8]), spin 23.7 rad/s (24.5),
  orbit 0.29 Hz / 0.45 m (0.27 Hz / 0.34 m). Changing the high-level gains at test time (kp_pos 0.6-1, kp_vel 2-3)
  leaves the orbit (0.35-0.6 m), so it is the learned inner loop's response, not the PID alone.
- **PX4 before the handover.** Logged actuator_motors: PX4 drives the failed rotor to 11.8 N and its diagonal partner to
  the 0.27 N floor within 80 ms (all 8 runs, same up to symmetry): ftc.PX4_REACT / px4_standin. With it as the
  stand-in, bench_sim.py reproduces i018's bench numbers: xy RMS last 5 s 0.353 (bench 0.35), orbit 0.335 (0.34),
  peak tilt 87.7 (86), max xy 2.79 (2.8), altitude loss 0.75 (0.72-0.80). The old hover-thrust stand-in gave 99 deg.
- **Prop loss** (`spin`: failed rotor keeps its commanded speed, inertia and drag; thrust and yaw torque gone):
  sim 17.4 rad/s, thrust [5.3 0.3 7.0 6.5] vs bench 17.6 rad/s, [5.4 0.3 7.1 6.3].
- **EKF2 error** vs GT: tilt error 3 deg at 0.5 s, 19 at 0.75 s, 26 at 1 s; heading error -15/-27 deg; velocity error
  1.0 m/s at 1 s, 2.3 at 1.5 s; then divergence with the crash. Model: attitude error rotation ramping at att_rate up
  to att_max after the fault (random axis), velocity error integrating g x the tilt error, GPS leak tau 1 s.
- Gazebo-gap eval (GZ_EVAL) now has prop_j 0.1: i018 0.358 m (was 0.130 with prop_j 1, unchanged code otherwise).

Training changes (train.py): --prop-j (log-uniform rotor inertia), --p-spin (prop-loss share), --delay-min 0,
--ekf-rand DEG, --reward precise (+0.3(1-exp(-|e|^2/0.25^2))), --handover MS (PX4 stand-in flies 0-MS ms after the
fault, steps masked), --moving P (start in flight: |v_xy| <= 2.5, |v_z| <= 1.2 m/s, |wz| <= 2 rad/s, any heading,
fault 0-0.5 s later). Local GPU now used (jax-cuda13 plugin): 62-66k steps/s at 16384 envs vs 42k on 8 CPU cores.

bench_sim.py (fault + 15 s, PX4 stand-in, 80 ms handover; gt = 4 ms delay, no lag, noise 0.2; crash = >90 deg or
> 2.7 m height loss). xy RMS last 5 s / orbit radius / peak tilt / crashes per 20:
| policy | init, changes | gt stop | gt prop | ekf 10 deg | ekf 20 deg | ekf 30 deg |
|---|---|---|---|---|---|---|
| i018 | - | 0.353 / 0.335 / 87.7 / 0 | 0.387 / 0.363 / 87.8 / 0 | 20 | 20 | - |
| i019a | i018, prop_j, spin, delay 0, precise | 0.142 / 0.104 / 85.1 / 0 | 0.164 / 0.119 / 85.2 / 0 | 16 | 20 | - |
| i019b | i019a + ekf 40 | 0.136 / 0.096 / 83.2 / 0 | 0.174 / 0.133 / 83.2 / 0 | 19 | 20 | - |
| i019c | i019a + ekf 20 | 0.135 / 0.093 / 82.2 / 0 | 0.146 / 0.108 / 82.3 / 0 | 15 | 20 | - |
| i020a | i019c + handover 120, ekf 20 (GPU) | 0.158 / 0.120 / 82.5 / 0 | 0.164 / 0.129 / 82.5 / 0 | 1 | 1 | - |
| i020b | i019c + handover 120, ekf 40 | 0.214 / 0.182 / 82.8 / 0 | 0.226 / 0.190 / 82.8 / 0 | 1 | 1 | 3 |
(i019 rows are with the PX4 stand-in; the ekf rows' xy error is large because the modelled velocity bias persists.)
Flight phases (bench data_00..39), i018 / i020a, crashes (>90 deg) per 20 headings: forward prop 7 / 1, climb prop
20 / 20, yaw stop 10 / 10, hover prop 0 / 0. Next: i021 = moving starts.

Round 3, continued (bench round 3b: 40 phase flights, data_00..39). bench_sim.py gained the phases (start in that
flight state with a per-unit heading, fault 0.3 s later) and the bench's crash rule (> 90 deg or > 2.7 m height loss).
i018 in the replica: forward 7-8/20, climb 20/20, yaw 10/20 above 90 deg (bench: tilt > 90 deg in those phases).
| run | init (+ changes) | gt stop / prop xy, peak | ekf 10 / 20 / 30 crashes | phases with crashes (per 20) |
|---|---|---|---|---|
| i021a | i020a + moving 0.5, ekf 20 (GPU) | 0.15 / 0.15, 78 | 0 / 0 / 5 | climb 9, yaw 4-5 |
| i021b | i020b + moving 0.5, ekf 40 | 0.16 / 0.12, 76 | 0 / 5 / 5 | climb 4-5, yaw 3-4 |
| i021c | i020a + moving 0.5, ekf 30 | 0.14 / 0.12, 77 | 0 / 4 / 4 | climb 5, yaw 2 |
| i022a | i021a + moving 0.7, tilt-pen 1, ekf 30 (GPU) | 0.16 / 0.14, 76 | 0 / 5 / 6 | climb 3, yaw 2 |
| **i022c** | i021c + moving 0.7, tilt-pen 1, ekf 30 | **0.11 / 0.13, 76** | 0 / 4 / 8 | **none (peak <= 89 deg)** |
i022c handover sweep: 0-120 ms 0 crashes; 160 ms 1/20 in climb prop and yaw stop. ONNX parity 1.1e-6; pftc_ref +
ONNX closed loop (PX4 stand-in, 80 ms) stop 0.077 m, prop 0.074 m. Packaged: policies/x500v2_i022c (+ transfer
tarball); main.py defaults to i022c.

Bench round 4 (2026-10-17, i022c in PX4 SITL + Gazebo; per-run tables in
bench/results/ftc_metrics_i022c.md at the repo root, not shipped). Replica prediction vs bench:
| setting | replica (bench_sim.py) | bench |
|---|---|---|
| GT hover, stop / prop: xy last 5 s, orbit, peak tilt, alt loss | 0.11 / 0.13 m, 0.05 / 0.06 m, 76 deg, 0.54 m | 0.07 / 0.09 m, 0.04 / 0.07 m, 74-76 deg, 0.50-0.67 m |
| phases, crashes | 0 in all | 0 / 40 |
| phase peak tilt: hover, forward, climb, descent, yaw | 80, 87, 89, 74, 89 | 75, 75, 83, 68, 87 |
| EKF2 state | ~10 deg tolerated, 20 deg 4/20, 30-40 deg 8/20 crash | 0/8; error at 1 s 14-20 deg tilt + 35-37 deg yaw (about 40 deg total), growing to 90-133 deg by 2.5 s |
| bench's onboard estimator (gyro attitude + own KF, IDEA-022 B) | - | 8/8, peak 74 deg, xy 0.10 m, orbit 0.06 m |
The replica is slightly pessimistic everywhere it was checked (peak tilt 2-12 deg high, xy 0.04 m high), so it is a
usable conservative screen. The EKF model under-predicts the bench crash rate: it caps the error, while EKF2's error
grows with the tilt it causes. Not pursued, because the bench's own estimator removes EKF2 from the loop (mag tuning
did not help: 0/8 with mag heading-only and init-only). Open for hardware: gyro scale error (bench offline: +1 % gives
22 deg tilt at 10 s at 20 rad/s spin), which the policy has only seen as the ~10 deg tolerance above.

Round 4b (2026-10-17): train against the bench's onboard estimator (IDEA-022 B) instead of EKF2, and check for
Gazebo overfit. New in ftc.py: plant keys (mass, J, motor_tau, t_err, sag, cg; per world) and the onboard estimator
(q_err integrated exactly from the gyro scale/bias error, heading offset, seed; JAX port of pos_kf.PosVelKF with
delayed, noisy GPS). train.py: --hw-rand, --cg-toward, --onboard P SCALE_XY SCALE_Z BIAS HEAD. bench_sim.py: hw_* (in
training range), hwx_* (wider), cal_* (passive drift), ob_* (onboard sweeps), "pre": "policy" (policy flies the healthy
drone: with open-loop hover before the fault a CG offset flips the drone before the fault).
Overfit check on i022c (policy flying before the fault), crashes per 20: mass +-10 % 6, CG 1-2 cm 3, thrust +-10 % and
sag 0.85 0 (xy 0.36 m), lag 2x 0, J +-20 % 0, all 10. So i022c was fitted to Gazebo's exact airframe.
Onboard estimator on i022c: bias up to 4 deg/s, heading offset up to 30 deg, GPS 5 Hz/200 ms all 0 crashes; gyro scale
1 % 4/20 and 20 m drift. Cause: z-axis scale error x the 20 rad/s spin is heading drift (11.5 deg/s per %); with
heading error > ~60-90 deg the high-level position loop pushes the wrong way. Not trainable; z scale trained to <= 0.5 %.
Calibration vs the bench offline numbers (tilt at 5/10 s): +1 % 6.5/24 (bench 11/22), +3 % 16/7 (30/39), bias 2 deg/s
1.7/4.9 (5/7); tilt error is bounded by about 2 x the spin-hover tilt and follows the heading error.
| run | init, changes | GT / phases | hw mass, cg, all | hwx all | ob scale 1 / 2 %, bias 4, head 30 |
|---|---|---|---|---|---|
| i022c | - | 0 crashes, peak 76-89 | 6, 3, 10 | 11-12 | 4 / 1, 0 (1.13 m), 0 (0.81 m) |
| i023a | i022c, hw-rand, onboard 0.8 3 0.3 2 30 (GPU, 150M) | 0, peak 72-88 | 0, 4, 4 | 14 | 4 / 3, 0, 0 (1.19 m) |
| **i023b** | i022c, hw-rand, onboard 0.8 3 0.5 4 45 (CPU, 150M) | **0, peak 69-85** | **0, 4, 3** | 14-15 | 0 / 6, 0 (1.13 m), 0 (0.67 m) |
| i024a | i023b + cg-toward 0.5 (GPU, 150M) | 0, peak 68-81 | 0, 3, 4 | 12-13 | 1 / 10, 0, 0 |
| i024b | i023b + cg-toward 0.7 (CPU, 120M) | 0, peak 70-81 | 0, 4, 4-5 | 12-14 | 0 / -, 0, 0 (0.49 m) |
i024a/b: motor lag 2x went from 0 to 3-6/20 crashes, CG barely moved. Every CG crash has the offset >= ~1 cm toward the
failed rotor. Chosen: i023b (handover 0-160 ms: 0 crashes; ONNX parity 1.1e-6; selftest PASS). Packaged
policies/x500v2_i023b + transfer tarball; main.py defaults to i023b.
Gotcha: export_onnx on the GPU compared against TF32 JAX matmuls (4.5e-3); it now sets matmul precision "highest".

Round 4c (2026-10-17): partial loss and exact motor symmetry. bench_sim.py: "health" key (failed rotor's thrust
share; >= 0.5: no PX4 stand-in, the policy flies through), part{30,50,65,80}_stop and part{30,50}_<phase>; per-unit
outputs (xy_units, peak_units, alt_units). train.py: --severity LO HI, --equivariant.
i023b on partial loss (never trained on it): 30 % 0/20, 50 % 7, 65 % 5, 80 % 0; part50 phases 2-7.
Symmetry: i023b's 4 fault copies share one network, but the selector and the healthy expert were free, so the policy
was not equivariant (max |pi(x) - P pi(T x)| = 1.7 on closed-loop observations). From matched starts, partial 50 %
crashed only motor 1 and 65 % only motor 2. Fix (train.mix_weights, train.expert_out): the logit for "motor i failed"
is the selector's motor-0-vs-healthy log-odds on obs_to_motor0(obs, i), healthy is logit 0, and the expert is averaged
over the 4 symmetries. Equivariance error 2.4e-7; converting i023b without training already gave partial 50 % 3/20
and healthy xy 0.15 -> 0.08 m. Training throughput drops to 33-37k steps/s (GPU), 27k (CPU), from 4x selector/expert.
| run | init, changes | GT / phases | partial 30/50/65/80 % (hover), part phases | hw all / hwx all | ob head 30 |
|---|---|---|---|---|---|
| i023b | - | 0 | 0, 7, 5, 0; 0-7 | 3 / 14-15 | 0.67 m |
| **i025a** | i023b, equivariant, partial 0.3, severity 0.3-0.8 (GPU, 150M) | **0, peak 70-82** | **0, 0, 0, 0; 0 (xy 0.07-0.72 m)** | 3-5 / 13 | 1.47 m |
| i025b | same, partial 0.5 (CPU, 150M) | 0, peak 77-88 | 0, 0, 0, 0; 0-2 (xy 1.3-3.7 m at 50-65 %) | 3-4 / 13 | 1.66 m |
Chosen: i025a. Matched-start per-motor spread 0.0 (xy, peak tilt, alt loss) for stop, prop and partial 30-80 %.
Handover 0-160 ms: 0 crashes. ONNX parity 1.1e-6, selftest PASS. Packaged policies/x500v2_i025a (no tarball);
main.py defaults to i025a. Intermittent faults deferred (user).

Stress test of i025a (2026-10-17; stress.py, cg_feasibility.py (now relaxed_hover.py --cg); full output runs/i025a/stress_report.txt). ftc.py
gained stress-only disturbances (off by default, i025a numbers unchanged): wind + gust (air-relative rotor drag),
thrust-axis tilt per rotor, faulted-rotor torque ratio, fault onset ramp; queue / state-lag / GPS buffers lengthened.
One batched rollout for ~100 sweep points x 20 units takes ~25 s on the GPU.
Breaking points (first value with crashes / >= 50 % crashed): thrust curve all rotors 0.6x (8/20); battery sag 0.6
(20/20, 0.7 fine); motor lag 4x (19/20, 3x fine); CG toward the failed rotor 1.0 cm (19/20, 0.75 cm fine; away 5 cm
and sideways 3 cm fine); actuation delay 60 ms (17/20, 40 fine); state lag 60 ms (10/20); estimate noise 10x (4/20);
handover 500 ms (20/20, 300 fine); heading offset 60 deg (3/20, 45 deg drifts 8 m); steady wind 10 m/s (17/20, 6 m/s
drifts 2 m); thrust-axis tilt 3 deg on all rotors (4/20) and 5 deg on the healthy rotors (3/20); a second rotor at
50 % next to the failed one (20/20; diagonal fine, neighbour at 70 % fine). No crash: mass 0.6-1.5x, J 0.4-2.5x,
rotor inertia 0.03-3x, GPS down to 1/3 Hz, fault at 5 m/s forward / 3 m/s climb or descent / 4 rad/s yaw, 1 m
altitude, gusts 3 +- 3 m/s, torque ratio 0.5-2 at 65 %, fault onset 0.5-5 s.
Monte Carlo, all at once: brutal ranges 89 % crash; half ranges 46 %, ranked handover, CG toward, actuation delay,
fault onset, motor lag (latency in total dominates; onset only matters combined with the stand-in's fixed handover).
CG feasibility: steady spinning hovers exist up to 5 cm in every direction (max thrust 5.0-6.2 N, partner rotor
2.4-4.3 N, spin axis 15-22 deg from body z). The policy's own hover keeps the partner at the 0.27 N floor, so a CG
toward the failed rotor (which needs less partner thrust) has no room: a policy limit, not physics.

Round 4d (2026-10-17/18): i026, thrust tilt + wider latency, from i025a. train.py: --tilt-rand DEG (70 % of episodes,
each rotor 0-DEG in any direction), --delay-max, --lag-max, --motor-tau-max. Both runs used delay <= 40 ms, lag <= 30
ms, motor lag <= 3x, partial 0.3: i026a (tilt 3, GPU, 94M + 56M resumed after a reboot with a fresh optimiser) and
i026b (tilt 4, CPU, 100M). Gotcha: the longer stress buffers (queue 41, history 7, GPS 51) made training 3x slower
(12k vs 35k steps/s); ftc.py now keeps them short (20 / 4 / 21) and stress.py enlarges them before make_sim.
Gate FAILED for both, i025a kept:
- partial loss regressed (part30 phases 0 -> 2-6/20 crashes, part30_stop i026b 20/20, part65 xy 0.7 -> 5 m or 20/20),
  GT/phase xy 0.12-0.13 -> 0.13-0.24 m;
- the target did not improve: thrust tilt 3 deg on all rotors still 4/20 (the same units), actuation delay 60 ms
  16 -> 20/20; only thrust curve 0.6x (8 -> 0/20), sag 0.6 (20 -> 8-11/20) and wind 10 m/s improved.
Tilt diagnosis (i025a): with no fault the tilted drone flies fine; with a fault the same 4 of 20 tilt draws crash
(motors 0 and 2, stop and prop alike), so it is specific tilt-direction combinations, most likely their net yaw torque
(up to ~0.1 N m per rotor at 3 deg, against ~0.2 N m of yaw drive) cancelling the spin the recovery needs.
