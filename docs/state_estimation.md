# State estimation during the post-failure spin: research report

**Question:** what can replace PX4 EKF2 as the state source for the fault-tolerant policy after a complete rotor
loss? The vehicle spins in yaw at 15-25 rad/s, tilts up to about 86 deg, and the magnetometer is rejected.

**Bench facts:**
- The i018 policy survives 8/8 on Gazebo ground-truth state and 0/8 on EKF2 (crashes after about 2 s).
- EKF2 attitude error vs ground truth reaches 30-70 deg within 2.5 s.

**Method:**
- 4 Haiku scrapers (learned estimators; privileged/end-to-end RL; classical high-rate filters; what the rotor-failure
  literature used).
- A lead that verified the key claims against full papers or code, and checked our own logs.
- The council (historian + reviewer).

Sensors allowed: IMU (250 Hz), magnetometer, barometer, GPS, ESC rotor speed. No camera, no LiDAR.

## Key finding from our own logs (lead, offline)
- **Raw-gyro dead reckoning is accurate in SITL.**
  - Integrating `sensor_combined` gyro at 250 Hz from the ground-truth attitude at fault onset stays 0.1-0.6 deg from
    ground truth at 1, 2.5, 5 and 10 s on the ground-truth-arm runs.
  - It stays mostly 1-6 deg by 10 s on the EKF-arm runs.
  - EKF2 on the same runs: 18-27 deg tilt error at 1 s, 25-125 deg at 2.5 s.
- **So the sensors are fine; EKF2's measurement corrections corrupt the attitude.**
- Likely suspect: `cs_mag_3d` (3D magnetometer fusion, which also corrects tilt) toggles 3-5 times in the first 2 s.
- EKF2 velocity error follows the attitude error: 1.0 m/s at 1 s, 3-5 m/s at 2.5-5 s. Height error 0.6-1.5 m.
- Caveat: the SITL gyro is near-ideal. Real gyro scale-factor and cross-axis errors at 20 rad/s are untested.

## Verified sources
| Source | What it shows | Caveat |
|---|---|---|
| Solanki & de Visser, AIAA SciTech 2022, [doi 10.2514/6.2022-0857](https://doi.org/10.2514/6.2022-0857) ([PDF](https://repository.tudelft.nl/file/File_bf63f301-ab76-45a8-b968-c1aedf21053b)) | Attitude with one fully damaged rotor, IMU + magnetometer only. Centrifugal-corrected accelerometer (ω_z²·[ϖ1, ϖ2, 0]), noise adapted on \|‖a‖−1\|, fuzzy gains. Yaw blends gyro-integrated yaw with the magnetometer, gated on field-norm deviation and yaw rate. RMS roll 1.46°, pitch 1.57°, yaw 9.89° at ~15.5 rad/s; Madgwick/complementary/Kalman yaw 143-165° | One Bebop 2 flight, processed offline (not closed loop) |
| uzh-rpg [fault_tolerant_control](https://github.com/uzh-rpg/fault_tolerant_control), `ftc_state_est/src/state_est_CF.cpp` | Mahony-style PI complementary filter (p 0.05, i 0.02). The accelerometer is corrected for the IMU offset ω×(ω×d) and, when \|r\| ≥ 10 rad/s, for relaxed hover: f = [−g·p/r·cosα, −g·q/r·cosα, g·tanα·sinα]. Yaw from VIO/OptiTrack, height from a rangefinder | GPL-3.0: port the idea, not the code. Needs vision for yaw and position |
| Sun et al., INDI with two opposing rotors lost, [arXiv 2002.07837](https://ar5iv.labs.arxiv.org/html/2002.07837) Sec. VIII-C | Wind tunnel: mocap position reduced to 10 Hz "to simulate GPS-like update rates", plus an IMU/magnetometer complementary filter for attitude. Controlled flight at 8.4 m/s, "robustness to significant attitude estimation errors" | The attitude errors raised the tracking error |
| Cioffi et al., Learned Inertial Odometry for drone racing, [arXiv 2210.15287](https://ar5iv.labs.arxiv.org/html/2210.15287) | A net takes 0.5 s of collective thrust + de-biased gyro and outputs displacement as an EKF update. ATE 0.56 m vs 10.10 m for the model-only EKF | About 5 min of real mocap training data; "cannot generalize to trajectories that have not been seen at training time" |
| Kaufmann et al., Deep Drone Acrobatics, [arXiv 2006.05768](https://ar5iv.labs.arxiv.org/html/2006.05768) | Teacher (MPC, full ground-truth state) → student on a history of 8 samples (VIO feature tracks + filtered IMU + reference). Sim-only training, real flight | The student uses vision |

## What the rotor-failure literature used (scraper D)
**No work found** flies relaxed-hover recovery on onboard-only, non-vision state (IMU + GPS + barometer +
magnetometer). The literature used:
- **Motion capture:** Mueller & D'Andrea 2014; Sun et al. 2020 indoor; [Nan et al. 2022](https://arxiv.org/abs/2109.12886)
  NMPC (EKF IMU + VICON); [Chen et al. 2025](https://arxiv.org/abs/2503.02649) (mocap into the PX4 EKF, outdoor left
  as future work).
- **Vision:** [Sun et al. 2021](https://arxiv.org/abs/2102.13406). Motion blur above 20 rad/s hurts VIO.
- **LiDAR-inertial odometry:** [Zhou et al. 2026](https://arxiv.org/html/2510.11306v2). They report IMU errors
  from centrifugal acceleration.
- **Closest to our sensor set:** Sun 2020's wind-tunnel test (10 Hz position + IMU/magnetometer filter).

No public Inter IIT 13.0 solution code was found.

## Learned estimators and end-to-end RL (scrapers A, B; partly abstract-level)
- **Learned inertial odometry:** TLIO ([2007.01867](https://arxiv.org/abs/2007.01867)),
  IONet ([1802.02209](https://arxiv.org/abs/1802.02209)), RoNIN ([1905.12853](https://arxiv.org/abs/1905.12853)) are
  pedestrian. AI-IMU ([1904.06064](https://arxiv.org/abs/1904.06064)) learns EKF noise for a car.
  DIDO ([2203.03149](https://arxiv.org/abs/2203.03149)) puts learned quadrotor dynamics in an EKF, regime unknown.
  None is tested at 15-40 rad/s.
- **Privileged / teacher-student:** works sim-to-real for flight. Examples: Loquercio et al. 2021
  ([2110.05113](https://arxiv.org/abs/2110.05113)), Xing et al. 2024 ([2403.12203](https://arxiv.org/abs/2403.12203)),
  RMA ([2107.04034](https://arxiv.org/abs/2107.04034)), asymmetric actor-critic
  ([1710.06542](https://arxiv.org/abs/1710.06542)). Every flight student found used vision, depth or an estimator;
  none flies on IMU/barometer/GPS alone while spinning. A student still needs GPS for position and a heading source.

## Solution cards (lead draft; council review below)
1. **EKF2 configuration probe:** `EKF2_MAG_TYPE` heading-only, then none, optionally switched at detection.
   - Fixes tilt, and velocity through tilt. Less than 1 day.
   - Pass: tilt error < 10° at 2.5 s on 8/8.
2. **Gyro dead reckoning from the pre-fault EKF attitude + a small position/velocity KF in `pftc_node`.**
   - Attitude: integrate raw gyro at 250 Hz. Add a low-gain tilt correction with the uzh-rpg rotation compensation
     (\|r\| ≥ 10 rad/s).
   - Yaw: gyro plus a gated Solanki-style magnetometer channel.
   - Position/velocity: accelerometer + GPS + barometer KF.
   - Step A: our attitude with ground-truth position/velocity. Step B: full own state.
   - 2-4 days. This is the baseline everything else must beat.
3. **Policy robustness retraining** (MuJoCo side), against an error model fitted to card 2's residuals.
4. **Learned residual** (Cioffi-style thrust/rpm + gyro → displacement) inside card 2's filter. Only if card 2 or
   the noisy-gyro test fails. 1-2 weeks.
5. **User idea (ii), end-to-end LSTM estimator.** Low value: attitude here is kinematics the gyro already solves, and
   learned inertial models don't generalise to unseen motion. Better as card 4.
6. **User idea (i), RL from raw sensor history (teacher-student).** Research track: still needs GPS + a heading
   source + recurrence. Weeks.

Recommended order:
1. Cards 1 and 2A in parallel.
2. Card 2B.
3. Fit the error statistics and hand them to the MuJoCo agent (card 3).
4. A realistic gyro-noise test. Go to card 4 only if it fails.
5. Cards 5 and 6 as research against card 2.

Fix the bench's ground-truth comparison first: slerp or nearest-sample, and check GT-to-IMU timestamp alignment.

## Open questions only an experiment answers
- Is EKF2's tilt corruption caused by 3D magnetometer fusion, by GNSS-velocity fusion, or by the output predictor?
- How much tilt, yaw and velocity error does the policy tolerate?
- What do real gyro scale-factor and cross-axis errors do at 20 rad/s?
- Is the magnetometer usable while spinning on hardware, with motor-current interference?
- Does 5-10 Hz GPS + accelerometer keep the orbit stable?
- The detection-time attitude must come from a pre-fault buffer, since EKF2 is already 3-8° off at 0.5 s.

## Council review (round 4)
- Historian: no repeats of past failures. Card 1 merges round-3 IDEA-017 (prior raised: mag is intermittently fused,
  not cleanly rejected). Card 2 merges IDEA-014/018 (prior raised strongly: the lead's offline gap is the evidence
  IDEA-018 was waiting for). Added: aid-source log table, calibration on healthy and 50 % fault logs.
- Reviewer (measured on ftc_00..02_ekf): EKF2 tilt error is 0.8 deg at the detection instant and grows afterwards,
  so seed from a ~150 ms pre-detection attitude buffer. SITL GPS is ~30 Hz (33 ms), not realistic. EKF2_MAG_TYPE=0
  (auto) in the logs. Blocking for card 2: frame unit test before live runs; GPS decimation + ideal-gyro wording for
  any claim beyond SITL.
- Idea log not shipped.
