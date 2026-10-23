# Fault-tolerant flight: learned policy after detection (land)

Complete loss of one rotor (`stop`: motor dead; `prop`: propeller gone, motor spinning) in hover. PX4 flies until the detector declares the failure, then the policy takes the motors (offboard direct actuator). Scored against Gazebo ground truth from the handover to fault + 15 s. Crash = height above ground < 0.3 m.

## Policy state: Gazebo ground truth (ideal state)

Survived 8/8; peak tilt max 71 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.10 m, orbit radius mean 0.08 m.

| run | phase | motor | fault | detect [ms] | handover [ms] | survived | crash at [s] | peak tilt [deg] at [s] | altitude loss [m] | xy RMS last 5 s [m] | orbit radius [m] | max xy error [m] | mean spin [rad/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ftc_00 | hover | 1 | stop | 80 | 81 | yes | - | 70 at 0.53 | 0.48 | 0.10 | 0.09 | 1.44 | 19.7 |
| ftc_01 | hover | 1 | prop | 80 | 80 | yes | - | 71 at 0.54 | 0.47 | 0.10 | 0.08 | 1.43 | 13.8 |
| ftc_02 | hover | 2 | stop | 80 | 81 | yes | - | 71 at 0.54 | 0.48 | 0.09 | 0.08 | 1.48 | 19.8 |
| ftc_03 | hover | 2 | prop | 80 | 81 | yes | - | 70 at 0.54 | 0.47 | 0.10 | 0.08 | 1.45 | 13.8 |
| ftc_04 | hover | 3 | stop | 80 | 81 | yes | - | 71 at 0.52 | 0.47 | 0.10 | 0.09 | 1.53 | 19.7 |
| ftc_05 | hover | 3 | prop | 80 | 81 | yes | - | 70 at 0.54 | 0.45 | 0.10 | 0.08 | 1.50 | 13.8 |
| ftc_06 | hover | 4 | stop | 80 | 81 | yes | - | 70 at 0.51 | 0.46 | 0.10 | 0.09 | 1.56 | 19.7 |
| ftc_07 | hover | 4 | prop | 80 | 81 | yes | - | 70 at 0.54 | 0.46 | 0.10 | 0.08 | 1.49 | 13.8 |

## Policy state: gyro attitude + own KF, fully onboard (IDEA-022 B)

Survived 8/8; peak tilt max 68 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.12 m, orbit radius mean 0.09 m.

| run | phase | motor | fault | detect [ms] | handover [ms] | survived | crash at [s] | peak tilt [deg] at [s] | altitude loss [m] | xy RMS last 5 s [m] | orbit radius [m] | max xy error [m] | mean spin [rad/s] | estimator max tilt err [deg] | estimator heading err 3-5 s [deg] | heading drift to end [deg] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ftc_00 | hover | 1 | stop | 80 | 81 | yes | - | 68 at 0.52 | 0.48 | 0.10 | 0.09 | 1.27 | 19.9 | 6.9 | 8.0 | -1.3 |
| ftc_01 | hover | 1 | prop | 80 | 81 | yes | - | 67 at 0.54 | 0.46 | 0.09 | 0.07 | 1.26 | 13.8 | 6.9 | 7.8 | -1.3 |
| ftc_02 | hover | 2 | stop | 80 | 82 | yes | - | 68 at 0.54 | 0.50 | 0.14 | 0.09 | 1.24 | 19.9 | 7.2 | 7.0 | -0.3 |
| ftc_03 | hover | 2 | prop | 80 | 80 | yes | - | 67 at 0.55 | 0.46 | 0.09 | 0.07 | 1.28 | 13.9 | 6.7 | 7.5 | -0.6 |
| ftc_04 | hover | 3 | stop | 80 | 81 | yes | - | 68 at 0.52 | 0.50 | 0.14 | 0.12 | 1.29 | 19.7 | 6.7 | 5.8 | +0.8 |
| ftc_05 | hover | 3 | prop | 80 | 81 | yes | - | 68 at 0.52 | 0.47 | 0.12 | 0.11 | 1.30 | 13.8 | 6.2 | 5.1 | +1.4 |
| ftc_06 | hover | 4 | stop | 80 | 81 | yes | - | 67 at 0.55 | 0.50 | 0.11 | 0.10 | 1.29 | 19.7 | 5.4 | 5.5 | +0.1 |
| ftc_07 | hover | 4 | prop | 80 | 82 | yes | - | 68 at 0.53 | 0.48 | 0.15 | 0.10 | 1.31 | 13.8 | 3.6 | 2.7 | +0.7 |

## Landing after the hold (hold 10 s once settled, descend at 0.4 m/s, motors cut on touchdown)

Pass = touchdown <= 0.5 m/s, never past 60 deg tilt, final tilt < 20 deg, motors cut within 1 s of contact, within 0.5 m of the hold point.

Landed (all gates) 16/16.

| run | motor | fault | descent at [s after handover] | touchdown v down [m/s] | lateral v [m/s] | tilt at contact [deg] | max tilt after [deg] | final tilt [deg] | spin < 1 rad/s after [s] | motors cut after contact [s] | distance from hold point [m] | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ftc_00 | 1 | stop | 14.3 | 0.41 | 0.20 | 9 | 11 | 1 | 0.30 | 0.02 | 0.08 | yes |
| ftc_01 | 1 | prop | 14.3 | 0.40 | 0.21 | 5 | 10 | 1 | 0.39 | 0.18 | 0.22 | yes |
| ftc_02 | 2 | stop | 14.4 | 0.40 | 0.25 | 6 | 12 | 2 | 0.44 | 0.09 | 0.17 | yes |
| ftc_03 | 2 | prop | 14.3 | 0.40 | 0.21 | 5 | 4 | 0 | 0.30 | 0.06 | 0.19 | yes |
| ftc_04 | 3 | stop | 14.4 | 0.40 | 0.20 | 10 | 12 | 1 | 0.48 | 0.16 | 0.18 | yes |
| ftc_05 | 3 | prop | 14.4 | 0.40 | 0.22 | 5 | 5 | 1 | 0.28 | 0.04 | 0.18 | yes |
| ftc_06 | 4 | stop | 14.4 | 0.40 | 0.27 | 6 | 9 | 1 | 0.47 | 0.19 | 0.11 | yes |
| ftc_07 | 4 | prop | 14.4 | 0.39 | 0.23 | 5 | 4 | 0 | 0.30 | 0.12 | 0.18 | yes |
| ftc_00 | 1 | stop | 14.0 | 0.43 | 0.20 | 12 | 13 | 0 | 0.87 | 0.66 | 0.18 | yes |
| ftc_01 | 1 | prop | 13.9 | 0.35 | 0.24 | 4 | 12 | 2 | 0.25 | 0.14 | 0.19 | yes |
| ftc_02 | 2 | stop | 13.9 | 0.40 | 0.27 | 8 | 6 | 1 | 0.33 | 0.05 | 0.04 | yes |
| ftc_03 | 2 | prop | 13.8 | 0.38 | 0.24 | 4 | 4 | 0 | 0.31 | 0.09 | 0.12 | yes |
| ftc_04 | 3 | stop | 14.2 | 0.37 | 0.35 | 9 | 6 | 1 | 0.43 | 0.11 | 0.08 | yes |
| ftc_05 | 3 | prop | 14.2 | 0.39 | 0.29 | 5 | 4 | 0 | 0.26 | 0.04 | 0.06 | yes |
| ftc_06 | 4 | stop | 14.2 | 0.42 | 0.27 | 12 | 9 | 1 | 0.38 | 0.05 | 0.21 | yes |
| ftc_07 | 4 | prop | 14.1 | 0.40 | 0.30 | 7 | 6 | 1 | 0.34 | 0.11 | 0.08 | yes |

