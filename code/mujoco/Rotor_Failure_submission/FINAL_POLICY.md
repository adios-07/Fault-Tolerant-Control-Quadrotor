# Final submitted controller: PFTC x500v2_i025a

This is the technical record of the one controller we submitted: what it is, how it was trained, and what it scored.

- **History of how we got here:** `README.md` and `runs/EXPERIMENTS.md`.
- **Gazebo / PX4 results:** the PX4 README (`../../px4_ws/src/rotor_fail/README.md`).

## 1. Deliverable

| Item | Value |
|---|---|
| Name | x500v2_i025a (passive fault-tolerant low-level policy) |
| Folder | `policies/x500v2_i025a/` (PX4 side: a copy in `../../px4_ws/src/rotor_fail/policies/x500v2_i025a/`) |
| Network for deployment | `policy.onnx`, opset 17, IR 8. Input `obs` float32 [N, 27], output `action` float32 [N, 4] in [−1, 1]. sha256 `c8a7e3b6309983afd5dafd4c7304b77db8798c697976ce774cc6b9c3bcf53926` |
| Same weights for this repo | `policy.pkl` (= `runs/i025a/policy.pkl`), sha256 `c22b7c9fc9e945de4c6460e6a79ae6ad1055ab13d890ab3949067a9e5b101b18` |
| Runner around the network | `pftc_ref.py`: numpy port of the high-level controller, the filters, the observation builder and the action mapping. The PX4 node uses it verbatim as `src/pftc.py`. |
| Checks | ONNX vs JAX: max difference 1.1e-6 on 12,000 closed-loop observations. `selftest.py` replays a recorded episode through `pftc_ref` + ONNX: 0.0 N difference, PASS. All files are in `SHA256SUMS`. |
| Other files in the folder | `results.txt` (all scores), `RUN.md`, `INTERFACE.md` (interface spec), `bench_sim.json` (raw replica scores) |

## 2. Vehicle model it was trained for

The model is the Holybro X500 v2, as crazyflow drone `x500v2` (`third_party/crazyflow_x500v2.patch`). It matches the
Gazebo `custom_x500`.

| Parameter | Value |
|---|---|
| Mass | 1.40 kg |
| Inertia J | diag(0.0152, 0.0152, 0.028) kg m² |
| Rotor positions | (±0.17678, ±0.17678, 0.069) m, in PX4 order: 0 front-right CCW, 1 back-left CCW, 2 front-left CW, 3 back-right CW |
| Thrust and drag | thrust $f=k_F\Omega^2$ with $k_F$ = 1.18e-5 N/(rad/s)²; drag torque 0.016·f; 0–11.8 N per rotor (T/W 3.44) |
| Hover | 3.433 N per rotor, 539 rad/s |
| Rotor lag | 12.5 ms up, 25 ms down |
| Rotor inertia | 2.65e-5 kg m² |
| Rotor drag | Gazebo `MulticopterMotorModel`: $F_i=-8.06428\text{e-}5\,\Omega_i\,v_{\perp,i}$ |
| PX4 armed thrust floor | 0.27 N |
| Rates | physics 500 Hz; policy and high-level controller 100 Hz |

## 3. Controller

### 3.1 High-level controller (fixed code)

These are the uav_ftc gains (Chen et al. 2025), from `ftc.high_level` and `pftc_ref.py`. Below, $\mathbf p,\mathbf v$
are the world position and velocity (z up), $\mathbf p^*$ the target, and $R$ the body-to-world rotation.

1. Velocity command and acceleration:
   - $\mathbf v^{cmd}=\mathrm{clip}(\mathbf p^*-\mathbf p,\pm10)$
   - $\mathbf a=\mathrm{diag}(2,2,6)(\mathbf v^{cmd}-\mathbf v)$, with the horizontal part limited to a 10° tilt and
     $a_z$ clipped to ±5 m/s².
2. Desired thrust direction:
   - $\mathbf n_{des}=(\mathbf a+g\mathbf e_3)/\lVert\cdot\rVert$
   - $\dot{\mathbf n}_{des}=\mathrm{clip}((\mathbf n_{des}-\mathbf n_{last})/0.01,\pm0.5)$
3. Primary-axis law with $\mathbf n_B=\mathbf e_3$ and $k_x=k_y=5$:
   - $\mathbf h=R^\top\mathbf n_{des}$, $h_3=\max(h_z,0.1)$
   - $\mathbf t=-5\mathbf h_{xy}-\omega_z(h_y,-h_x)-(R^\top\dot{\mathbf n}_{des})_{xy}$
   - $p_{des}=t_y/h_3$, $q_{des}=-t_x/h_3$, $a_{z,des}=\lVert\mathbf a+g\mathbf e_3\rVert$
4. All three commands are low-passed (bandwidth 50 rad/s) and clipped: ±10 rad/s for $p_{des}, q_{des}$, ±50 m/s²
   for $a_{z,des}$. No yaw command is ever given.

### 3.2 Observation (27 floats, `ftc.observe`)

| Index | Signal | Scaling |
|---|---|---|
| 0–2 | $\mathbf p-\mathbf p^*$ | /5 |
| 3–5 | $\mathbf v$ | /5 |
| 6–9 | quaternion w, x, y, z (sign chosen so w ≥ 0) | 1 |
| 10–12 | body rates $\omega_x,\omega_y,\omega_z$ | /(5, 5, 20) |
| 13–16 | last rotor thrust commands, low-passed ($\tau$ 0.025 s) [N] | /10 |
| 17–19 | specific force in the body frame $\mathbf a_b$ (IMU reading; +9.81 on z at rest) | /(5, 5, 20) |
| 20–21 | $p_{des}, q_{des}$ | /5 |
| 22 | yaw-rate command (always 0) | – |
| 23 | $a_{z,des}$ | /20 |
| 24–26 | angular acceleration, filtered: $(\boldsymbol\omega_k-\boldsymbol\omega_{k-1}+0.12\,\dot{\boldsymbol\omega}_{k-1})/(0.12+0.01)$ | /50 |

The observation contains **no fault information**. The policy is never told that, or which, rotor failed.

### 3.3 Action

Each rotor's thrust is $f_i=\mathrm{clip}(3.433+8.367\,a_i,\ 0.27,\ 11.8)$ N, and its speed is
$\Omega_i=\sqrt{f_i/1.18\text{e-}5}$ rad/s. The policy replaces the rate controller and the mixer.

### 3.4 Network: exactly symmetric 5-mode selector-controller (`train.py` `Joint`, `mix_weights`, `expert_out`)

**Symmetries.** $T_i$ is the mirror and/or 180° rotation that maps rotor $i$ onto rotor 0: a signed permutation of the
27 observations. $P_i$ is the matching permutation of the 4 actions (`ftc.TO_MOTOR0`, `obs_to_motor0`,
`act_from_motor0`).

**The parts:**
- **Fault controller:** one base MLP $\pi_0$ ("rotor 0 failed"). The four fault modes are
  $\pi_i(\mathbf o)=P_i^{-1}\,\mathrm{clip}(\pi_0(T_i\mathbf o))$.
- **Healthy expert:** an MLP $\pi_H$, averaged over the four symmetries:
  $\bar\pi_H(\mathbf o)=\tfrac14\sum_i P_i^{-1}\mathrm{clip}(\pi_H(T_i\mathbf o))$.
- **Selector:** an MLP $\sigma$ with 5 outputs. The logit for "rotor $i$ failed" is
  $\sigma_0(T_i\mathbf o)-\sigma_4(T_i\mathbf o)$, and the logit for "healthy" is 0.
  - The weights $\mathbf w$ are the softmax of these 5 logits.

**Action:** $\mathbf a=\sum_{i=0}^{3}w_i\pi_i(\mathbf o)+w_H\bar\pi_H(\mathbf o)$.

**Properties:**
- **Exact equivariance by construction:** $\pi(T\mathbf o)=P\pi(\mathbf o)$ for every symmetry. The measured error is
  2.4e-7. A failure of any rotor is flown exactly like the mirror image of a rotor-0 failure.
- **Every MLP** is 3 hidden layers × 64, tanh, orthogonal init.
- **Parameters** (actor + critic each): base 20,553; expert 20,553; selector 10,437. The ONNX graph holds the actors
  and the selector.
- **Cost:** 125,459 multiply-accumulates per call, 41 µs on one laptop CPU core (onnxruntime, batch 1).

### 3.5 Shadow mode and handover

- **Before detection:** `PFTC.track(p, v, q, w, target, u_applied)` runs every 10 ms with the thrusts PX4 actually
  applies, so the filters and high-level states stay warm.
- **From the detection tick on:** `PFTC.step(...)` flies, with no reset.
- **Tested range:** handover delays of 0–160 ms after the fault with 0 crashes.

## 4. Training of i025a

**Run:** `runs/i025a`. It is a fine-tune of i023b (§4.5) for 150M environment steps on an RTX 4060 Laptop GPU, at
33,074 steps/s (76 min). The full argument list is line 1 of `runs/i025a/log.txt`.

### 4.1 PPO

| Setting | Value |
|---|---|
| Batch | 16,384 parallel simulators × 64-step horizon (1.05M steps per update) |
| Updates | 4 epochs × 8 minibatches |
| PPO clip | 0.2 |
| Discount and GAE | $\gamma$ 0.99, $\lambda$ 0.95 |
| Optimiser | Adam, lr 1e-4, global-norm gradient clip 0.5 |
| Loss coefficients | value 0.5, entropy 0 |
| Reward scale | 0.1 |
| Policy noise | Gaussian, learned per-dimension σ, reset to 0.1 at the start (0.079 at the end) |
| Selector loss | supervised $\lVert\mathbf w-\mathbf w^{label}\rVert^2$, weight 1. Label: the failed rotor once a fault with $k<0.5$ has acted, otherwise "healthy" |
| Masked steps | steps flown by the PX4 stand-in are excluded from the loss |

### 4.2 Reward (`ftc.step`; reward "precise", tilt penalty 1)

The reward is $r=1.5-\min(P,1)$ while alive, and 0 on a crash (which ends the episode), with

$$
\begin{aligned}
P ={}& 0.1\lVert\mathbf e\rVert^2+0.01\lVert\mathbf v\rVert^2+0.02\,[(\omega_x-p_{des})^2+(\omega_y-q_{des})^2+c_z]\\
&+0.0008\lVert\mathbf u_k-\mathbf u_{k-1}\rVert^2+0.2\,(a_{b,z}-a_{z,des})^2\\
&+0.3\,(1-e^{-\lVert\mathbf e\rVert^2/0.25^2})+\max(0.342-R_{33},0)/0.342
\end{aligned}
$$

- **Symbols:** $\mathbf e=\mathbf p-\mathbf p^*$, and $\mathbf u_k$ is the 4 thrust commands at step $k$.
- **$c_z$:** $\omega_z^2$ while yaw is controllable. After a fault with $k<0.5$ it becomes
  $\max(\lvert\omega_z\rvert-40,0)^2$, so the spin is free up to 40 rad/s.
- **Crash:** $z<0.1$ m, more than 3 m from the target, upside down ($R_{33}<0$), $\lVert\boldsymbol\omega\rVert>80$
  rad/s, or NaN.

### 4.3 Episodes and randomisation (`train.sample_resets`, drawn per episode)

**Episodes.** 15 s long, starting within ±0.5 m of the target.
- **Fault:** in 50 % of episodes one random rotor fails, at a random time in 1–4 s, or 0–0.5 s after a moving start.
- **Fault types within faulted episodes:**
  - **partial loss**, 30 %: thrust and drag torque scaled by $k\sim U(0.3,0.8)$;
  - **complete loss**, the rest ($k=0$); half of these are **prop loss** (the rotor keeps spinning with its inertia and
    drag), half a **stopped motor**.

**Randomisation** (one draw per episode):

| Group | Setting | Range |
|---|---|---|
| flight state (70 % of episodes) | speed | horizontal ≤ 2.5 m/s, vertical ±1.2 m/s |
| | yaw rate | ±2 rad/s |
| | heading | any |
| PX4 before the handover | stand-in | PX4's logged reaction (failed rotor ramps to 11.8 N, partner to 0.27 N within 80 ms) for 0–120 ms after a severe fault |
| timing | actuation delay | 0–20 ms |
| | state lag | 0–20 ms |
| | state noise | 0–1 × σ, with σ = (p 0.02 m, v 0.05 m/s, quaternion 0.0087, ω 0.05 rad/s, $\mathbf a_b$ 0.1 m/s²) |
| rotor model | rotor drag | 0.5–1.5 × Gazebo |
| | rotor inertia | 0.08–1.2 × physical, log-uniform |
| | thrust floor | 0.27 N |
| airframe | mass | ±10 % |
| | CG offset | 0–2 cm, any horizontal direction |
| | per-rotor thrust/torque curve | ±10 % |
| | max thrust (battery sag) | 85–100 % |
| | motor time constant | 0.8–2× |
| | J | ±20 % |
| onboard-estimator errors (80 % of episodes; the model of the PX4 bench's gyro + KF estimator) | gyro scale | ±3 % on x/y, ±0.5 % on z |
| | gyro bias | ≤ 4°/s, random direction |
| | attitude seed error | ≤ 1° |
| | heading offset | ±45° |
| | KF | JAX copy of `pos_kf.PosVelKF` |
| | GPS | 5–10 Hz, 100–200 ms delay, velocity noise 0.1–0.3 m/s, position noise ≤ 5 cm |

### 4.4 Simulator used in training

The base is crazyflow `first_principles` dynamics (JAX, batched) with the ftc.py Gazebo-gap plugins at 500 Hz:
- actuation-delay queue;
- speed-dependent rotor drag with yaw damping, CG torque;
- per-world faults and airframe parameters;
- thrust floor;
- estimate lag, noise and the onboard-estimator error model.

### 4.5 Where the starting weights came from

i014a_m0 (PPO from scratch, rotor 0) → i016 (5-mode selector) → i018 (joint PPO, Gazebo gap) → i019c (precision
reward, rotor inertia, prop loss) → i020a (PX4 stand-in) → i021c / i022c (moving starts, tilt penalty) → i023b
(airframe and onboard-estimator errors) → **i025a** (exact symmetry, partial loss). The commands are in `README.md`,
Reproducibility.

## 5. Results of the submitted policy

**Replica of the PX4 bench** (`bench_sim.py`): 20 runs per setting = 4 motors × 5 starts, PX4 stand-in for 80 ms,
scored from the handover to fault + 15 s. Crash = tilt > 90° or height loss > 2.7 m. Source: `policies/x500v2_i025a/results.txt`.

| Setting | Crashes /20 | xy RMS last 5 s | Peak tilt |
|---|---|---|---|
| complete stop / prop loss, hover | 0 / 0 | 0.12 / 0.13 m | 72° |
| 5 flight phases (hover, 2 m/s forward, ±1 m/s climb/descent, 1.5 rad/s yaw) × stop/prop | 0 | 0.12–0.13 m (forward 0.77 m, distance flown before the handover) | 70–82° |
| partial loss 30 / 50 / 65 / 80 % thrust left | 0 / 0 / 0 / 0 | 0.09 / 0.07 / 0.72 / 0.36 m | 53 / 36 / 6 / 3° |
| partial 30 % and 50 %, all 5 phases | 0 | 0.10–0.71 m | – |
| handover delay 0 / 80 / 120 / 160 ms | 0 | – | ≤ 86° |

**Per-motor symmetry.** From matched starts, all 4 motors give identical xy error, peak tilt and altitude loss (spread
0.0) for stop, prop loss and partial 30–80 %.

**Onboard-estimator tolerance** (hover, 10 Hz GPS with 100 ms delay):

| Error | Crashes /20 / xy RMS |
|---|---|
| gyro bias 1 / 2 / 4°/s | 0 / 0.20 m, 0 / 0.33 m, 0 / 1.22 m |
| heading offset 10 / 20 / 30° | 0 / 0.18 m, 0 / 0.21 m, 0 / 1.47 m |
| GPS 5 Hz with 200 ms delay | 0 / 0.22 m |
| 1 % gyro scale | no crash, but it drifts 25 m (heading drift) |

**Airframe errors**, held out at these widths. Crashes per 20:
- **0:** mass ±10 % and ±15 %; thrust curve ±10 % / ±15 % with 85 % / 80 % sag; motor lag 2× and 2.5×; J ±20 % and
  ±30 %.
- **4/20:** CG 1–2 cm. The crashes are with the CG offset toward the failed rotor.
- **3–5/20:** all errors at once.

**Breaking points** (`stress.py`, `runs/i025a/stress_report.txt`). Shown as fine / first failure (crashes per 20):

| Error | Fine | First failure |
|---|---|---|
| CG toward the failed rotor | 0.75 cm | 1.0 cm (19) |
| thrust-axis tilt, all rotors | – | 3° (4) |
| actuation delay | 40 ms | 60 ms (16–17) |
| state lag | 40 ms | 60 ms (10–12) |
| motor lag | 3× | 4× (19) |
| battery sag | 0.7 | 0.6 (20) |
| thrust curve, all rotors | 0.7× | 0.6× (8) |
| heading offset | 45° | 60° (1–3) |
| steady wind | 6 m/s | 10 m/s (17) |

No failures in the tested ranges of mass 0.6–1.5×, J 0.4–2.5×, rotor inertia 0.03–3×, or GPS down to 1/3 Hz.

**Task 1d benchmark** (`run_benchmark.py`, rotor 0 stopped at 3 s, x500v2 + Gazebo rotor drag; `report/benchmark_kpis.md`):

| Controller | Crash | Altitude loss | Max tilt | Recovery | Final xy RMS | Spin |
|---|---|---|---|---|---|---|
| Mellinger | crashes at 3.41 s | – | – | – | – | – |
| i025a | no crash | 0.19 m | 40.5° | 1.40 s | 0.04 m | 25.9 rad/s |

**PX4 SITL + Gazebo** (PX4 README, *Recovery*): 128/128 runs survived with the onboard estimator, 0.11–0.12 m error
after a complete loss. EKF2 state: 0/8.

## 6. Operating limits and requirements

- **Heading:** heading error must stay below about 20° (30° costs accuracy). Growing heading error cannot be
  tolerated, so the estimator needs heading aiding on real hardware.
- **CG:** within about 0.5 cm of the motor centre. 1 cm toward the failed rotor crashes.
- **Motor alignment:** motor mounts need to be aligned well; 3° of tilt on all rotors crashes 4/20.
- **Latency:** the total control latency should stay below about 40 ms.
- **Targets:** ramp setpoint changes larger than about 0.5 m. Training starts were within ±0.5 m of the target.
- **Not modelled:** intermittent faults, two-rotor faults, IMU vibration and saturation, propeller aerodynamics during
  the spin.

## 7. Run it

```bash
# from this folder, with the .venv from RUN.md section 1 active (crazyflow is not shipped: see RUN.md)
cd policies/x500v2_i025a && sha256sum -c SHA256SUMS && python3 selftest.py && cd -   # must print PASS
python main.py --motor 1                   # MuJoCo viewer: hover, rotor 1 fails at 3 s, recovery
python run_benchmark.py                    # Task 1d KPIs + plots -> report/
python bench_sim.py runs/i025a/policy.pkl  # all replica settings
python stress.py runs/i025a/policy.pkl     # breaking points + Monte Carlo
```

Retrain this stage from i023b (about 76 min on the GPU):

```bash
python train.py --id i025a --steps 1.5e8 --envs 16384 --init runs/i023b/policy.pkl --joint --reset-std 0.1 \
  --lr 1e-4 --p-fault 0.5 --gz-rand --reward precise --tilt-pen 1 --moving 0.7 --handover 120 --delay-min 0 \
  --prop-j 0.08 1.2 --p-spin 0.5 --hw-rand --onboard 0.8 3 0.5 4 45 --equivariant --partial 0.3 --severity 0.3 0.8
```
