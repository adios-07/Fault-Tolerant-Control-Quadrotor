# Fault-tolerant flight: learned policy after detection (i025a_gyro)

Complete loss of one rotor (`stop`: motor dead; `prop`: propeller gone, motor spinning) in hover. PX4 flies until the detector declares the failure, then the policy takes the motors (offboard direct actuator). Scored against Gazebo ground truth from the handover to fault + 15 s. Crash = height above ground < 0.3 m.

## Phase set (data runs): policy state gyro, fault in every flight phase

- hover: Survived 8/8; peak tilt max 68 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.11 m, orbit radius mean 0.09 m
- forward: Survived 8/8; peak tilt max 74 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.11 m, orbit radius mean 0.09 m
- climb: Survived 8/8; peak tilt max 77 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.11 m, orbit radius mean 0.09 m
- descent: Survived 8/8; peak tilt max 64 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.12 m, orbit radius mean 0.10 m
- yaw: Survived 8/8; peak tilt max 83 deg, 0 runs above 90 deg; xy RMS last 5 s mean 0.12 m, orbit radius mean 0.09 m

| run | phase | motor | fault | detect [ms] | handover [ms] | survived | crash at [s] | peak tilt [deg] at [s] | altitude loss [m] | xy RMS last 5 s [m] | orbit radius [m] | max xy error [m] | mean spin [rad/s] | estimator max tilt err [deg] | estimator heading err 3-5 s [deg] | heading drift to end [deg] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| data_00 | hover | 1 | stop | 80 | 81 | yes | - | 68 at 0.53 | 0.50 | 0.10 | 0.09 | 1.25 | 19.8 | 6.4 | 7.6 | +0.4 |
| data_01 | hover | 1 | prop | 80 | 81 | yes | - | 67 at 0.54 | 0.46 | 0.11 | 0.08 | 1.23 | 13.8 | 6.3 | 7.1 | +0.2 |
| data_02 | hover | 2 | stop | 60 | 60 | yes | - | 67 at 0.53 | 0.44 | 0.10 | 0.09 | 1.13 | 19.8 | 8.2 | 7.9 | +0.6 |
| data_03 | hover | 2 | prop | 80 | 81 | yes | - | 67 at 0.53 | 0.44 | 0.09 | 0.07 | 1.26 | 13.8 | 7.9 | 7.4 | +0.4 |
| data_04 | hover | 3 | stop | 80 | 81 | yes | - | 67 at 0.52 | 0.48 | 0.11 | 0.10 | 1.33 | 19.8 | 6.6 | 3.2 | -0.2 |
| data_05 | hover | 3 | prop | 80 | 81 | yes | - | 68 at 0.52 | 0.49 | 0.14 | 0.12 | 1.26 | 13.8 | 6.5 | 5.7 | +0.0 |
| data_06 | hover | 4 | stop | 80 | 80 | yes | - | 68 at 0.53 | 0.48 | 0.12 | 0.11 | 1.30 | 19.7 | 6.7 | 5.8 | -0.1 |
| data_07 | hover | 4 | prop | 80 | 81 | yes | - | 67 at 0.52 | 0.46 | 0.13 | 0.10 | 1.30 | 13.8 | 6.9 | 6.1 | -0.0 |
| data_08 | forward | 1 | stop | 80 | 81 | yes | - | 59 at 0.38 | 0.38 | 0.09 | 0.09 | 1.96 | 19.8 | 3.4 | 4.3 | +3.1 |
| data_09 | forward | 1 | prop | 80 | 81 | yes | - | 60 at 0.37 | 0.42 | 0.10 | 0.09 | 1.83 | 13.9 | 1.7 | 2.0 | +0.2 |
| data_10 | forward | 2 | stop | 80 | 81 | yes | - | 74 at 0.54 | 0.62 | 0.12 | 0.09 | 1.67 | 19.8 | 2.7 | 2.0 | +0.3 |
| data_11 | forward | 2 | prop | 80 | 81 | yes | - | 70 at 0.51 | 0.51 | 0.11 | 0.09 | 1.64 | 13.9 | 1.9 | 1.4 | -0.0 |
| data_12 | forward | 3 | stop | 80 | 81 | yes | - | 73 at 0.53 | 0.62 | 0.11 | 0.10 | 1.69 | 19.8 | 1.5 | -0.4 | -0.5 |
| data_13 | forward | 3 | prop | 80 | 80 | yes | - | 68 at 0.52 | 0.51 | 0.10 | 0.09 | 1.69 | 13.8 | 1.9 | 0.8 | -0.3 |
| data_14 | forward | 4 | stop | 80 | 81 | yes | - | 59 at 0.38 | 0.39 | 0.11 | 0.09 | 1.96 | 19.8 | 1.1 | 0.8 | -0.1 |
| data_15 | forward | 4 | prop | 80 | 80 | yes | - | 61 at 0.36 | 0.44 | 0.11 | 0.08 | 1.94 | 13.8 | 1.5 | -0.1 | -0.3 |
| data_16 | climb | 1 | stop | 80 | 81 | yes | - | 76 at 0.56 | 0.56 | 0.08 | 0.08 | 1.45 | 19.8 | 7.4 | 6.6 | +0.1 |
| data_17 | climb | 1 | prop | 80 | 80 | yes | - | 76 at 0.54 | 0.57 | 0.08 | 0.07 | 1.48 | 13.8 | 7.2 | 7.6 | +0.1 |
| data_18 | climb | 2 | stop | 80 | 80 | yes | - | 77 at 0.53 | 0.58 | 0.10 | 0.09 | 1.47 | 19.7 | 8.5 | 7.7 | +0.5 |
| data_19 | climb | 2 | prop | 80 | 80 | yes | - | 77 at 0.53 | 0.57 | 0.10 | 0.07 | 1.48 | 13.8 | 8.2 | 9.9 | +0.4 |
| data_20 | climb | 3 | stop | 80 | 80 | yes | - | 77 at 0.55 | 0.60 | 0.14 | 0.11 | 1.70 | 19.7 | 6.3 | 4.8 | +0.2 |
| data_21 | climb | 3 | prop | 80 | 81 | yes | - | 77 at 0.53 | 0.57 | 0.13 | 0.10 | 1.62 | 13.8 | 5.3 | 5.3 | -0.2 |
| data_22 | climb | 4 | stop | 80 | 81 | yes | - | 77 at 0.55 | 0.61 | 0.16 | 0.11 | 1.61 | 19.6 | 5.9 | 4.3 | -0.5 |
| data_23 | climb | 4 | prop | 80 | 81 | yes | - | 76 at 0.56 | 0.56 | 0.12 | 0.09 | 1.68 | 13.8 | 6.7 | 5.4 | -0.0 |
| data_24 | descent | 1 | stop | 80 | 81 | yes | - | 64 at 0.51 | 0.63 | 0.09 | 0.09 | 1.74 | 19.9 | 6.9 | 8.2 | +0.5 |
| data_25 | descent | 1 | prop | 80 | 82 | yes | - | 63 at 0.50 | 0.64 | 0.10 | 0.08 | 1.67 | 13.9 | 6.7 | 7.6 | +0.2 |
| data_26 | descent | 2 | stop | 80 | 81 | yes | - | 63 at 0.52 | 0.62 | 0.11 | 0.09 | 1.76 | 20.0 | 6.6 | 8.0 | +0.6 |
| data_27 | descent | 2 | prop | 80 | 81 | yes | - | 63 at 0.52 | 0.60 | 0.12 | 0.08 | 1.68 | 14.0 | 7.2 | 7.0 | +0.0 |
| data_28 | descent | 3 | stop | 80 | 81 | yes | - | 64 at 0.52 | 0.62 | 0.13 | 0.11 | 1.77 | 19.9 | 6.0 | 5.0 | -0.4 |
| data_29 | descent | 3 | prop | 80 | 80 | yes | - | 63 at 0.51 | 0.61 | 0.14 | 0.11 | 1.76 | 13.9 | 6.0 | 5.4 | -0.2 |
| data_30 | descent | 4 | stop | 80 | 80 | yes | - | 64 at 0.50 | 0.63 | 0.11 | 0.11 | 1.87 | 19.8 | 6.2 | 4.8 | -0.4 |
| data_31 | descent | 4 | prop | 80 | 81 | yes | - | 63 at 0.51 | 0.62 | 0.13 | 0.11 | 1.78 | 13.9 | 6.4 | 5.2 | -0.2 |
| data_32 | yaw | 1 | stop | 80 | 81 | yes | - | 82 at 0.57 | 0.88 | 0.09 | 0.09 | 1.50 | 19.8 | 6.4 | 6.8 | +0.2 |
| data_33 | yaw | 1 | prop | 80 | 81 | yes | - | 80 at 0.56 | 0.82 | 0.09 | 0.07 | 1.33 | 13.8 | 7.7 | 6.7 | +0.2 |
| data_34 | yaw | 2 | stop | 80 | 81 | yes | - | 83 at 0.55 | 0.91 | 0.10 | 0.09 | 2.00 | 19.7 | 7.2 | 7.7 | +0.5 |
| data_35 | yaw | 2 | prop | 80 | 81 | yes | - | 79 at 0.55 | 0.77 | 0.14 | 0.09 | 1.69 | 13.8 | 6.6 | 6.8 | +0.2 |
| data_36 | yaw | 3 | stop | 60 | 60 | yes | - | 51 at 0.48 | 0.27 | 0.14 | 0.11 | 0.99 | 19.8 | 5.2 | 5.1 | -0.5 |
| data_37 | yaw | 3 | prop | 80 | 81 | yes | - | 54 at 0.48 | 0.28 | 0.10 | 0.10 | 1.08 | 13.9 | 5.1 | 5.1 | -0.2 |
| data_38 | yaw | 4 | stop | 80 | 81 | yes | - | 55 at 0.49 | 0.23 | 0.13 | 0.11 | 0.95 | 19.9 | 4.5 | 5.1 | -0.3 |
| data_39 | yaw | 4 | prop | 80 | 81 | yes | - | 55 at 0.51 | 0.23 | 0.14 | 0.10 | 0.95 | 13.9 | 4.3 | 5.1 | +0.1 |

