# Task 2c benchmark results

Runs: 24 tune + 24 test fault runs, one healthy session of 13.0 min. Latency = sim time from fault injection to the detector's decision.

## Test split: baseline vs tuned

| detector | detection rate | false verdict before the fault | latency mean / worst [ms] | isolation accuracy | root-cause accuracy | false alarms (healthy, report half) |
|---|---|---|---|---|---|---|
| baseline | 100 % (24/24) | 0 | 125 / 180 | 100 % (24/24) | 100 % (24/24) | 0 (< 0.46 /min, rule of three) |
| tuned | 100 % (24/24) | 0 | 80 / 80 | 100 % (24/24) | 100 % (24/24) | 0 (< 0.46 /min, rule of three) |

Arbiter (per-run cost, test split, lower is better): `24 units | gain +45 (+1.26 x noise 35.7) | better on 24/24 | range +20..+100 -> PROMOTE`

## Per fault mode (test, baseline)

| mode | detection rate | false verdict before the fault | latency mean / worst [ms] | isolation accuracy | root-cause accuracy |
|---|---|---|---|---|---|
| stop | 100 % (8/8) | 0 | 100 / 100 | 100 % (8/8) | 100 % (8/8) |
| weak | 100 % (8/8) | 0 | 100 / 100 | 100 % (8/8) | 100 % (8/8) |
| prop | 100 % (8/8) | 0 | 175 / 180 | 100 % (8/8) | 100 % (8/8) |

## Per flight phase (test, baseline)

| phase | detection rate | false verdict before the fault | latency mean / worst [ms] | isolation accuracy | root-cause accuracy |
|---|---|---|---|---|---|
| hover | 100 % (6/6) | 0 | 127 / 180 | 100 % (6/6) | 100 % (6/6) |
| forward | 100 % (6/6) | 0 | 123 / 180 | 100 % (6/6) | 100 % (6/6) |
| climb/descent | 100 % (6/6) | 0 | 127 / 180 | 100 % (6/6) | 100 % (6/6) |
| yaw | 100 % (6/6) | 0 | 123 / 180 | 100 % (6/6) | 100 % (6/6) |

## Tuning grid (tune split)

Baseline healthy tune-half false alarms: 0. Chosen: IMU_HOLD 3, SMOOTH 0.2.

| IMU_HOLD | SMOOTH | mean cost [ms] | false alarms (tune half) | allowed |
|---|---|---|---|---|
| 3 | 0.2 | 79.8 | 0 | yes |
| 3 | 0.3 | 78.2 | 1 | no |
| 3 | 0.5 | 68.2 | 2 | no |
| 3 | 0.7 | 63.2 | 2 | no |
| 4 | 0.2 | 99.8 | 0 | yes |
| 4 | 0.3 | 98.2 | 0 | yes |
| 4 | 0.5 | 88.2 | 0 | yes |
| 4 | 0.7 | 83.2 | 0 | yes |
| 5 | 0.2 | 106.5 | 0 | yes |
| 5 | 0.3 | 106.5 | 0 | yes |
| 5 | 0.5 | 102.3 | 0 | yes |
| 5 | 0.7 | 100.7 | 0 | yes |
| 8 | 0.2 | 126.5 | 0 | yes |
| 8 | 0.3 | 126.5 | 0 | yes |
| 8 | 0.5 | 122.3 | 0 | yes |
| 8 | 0.7 | 120.7 | 0 | yes |

## Extra runs: 50 % faults, final detector, live

Flown after tuning with the final detector (IMU_HOLD 3, SMOOTH 0.2). ".ulg exact" is param5 of the markers (sim time of the event); ".ulg arrival" is when the marker reached PX4.

| run | phase | mode | intensity | motor | detected motor | detected cause | replay [ms] | live [ms] | .ulg exact [ms] | .ulg arrival [ms] |
|---|---|---|---|---|---|---|---|---|---|---|
| extra_00 | hover | prop | 0.5 | 1 | 1 | prop_loss | 80 | 80 | 80 | 80 |
| extra_01 | forward | prop | 0.5 | 2 | 2 | prop_loss | 80 | 80 | 80 | 80 |
| extra_02 | climb | prop | 0.5 | 3 | 3 | prop_loss | 80 | 80 | 80 | 80 |
| extra_03 | yaw | prop | 0.5 | 4 | 4 | prop_loss | 80 | 80 | 80 | 80 |
| extra_04 | hover | weak | 0.5 | 1 | 1 | weak_motor | 80 | 80 | 80 | 80 |
| extra_05 | hover | stop | 1.0 | 2 | 2 | motor_stop | 80 | 80 | 80 | 80 |

## Replay validation

Offline vs live decision time on 48 runs: mean bias -2.1 ms, worst 20.0 ms (accept if mean |bias| < 10 ms).

## All test runs (baseline)

| run | phase | mode | intensity | motor | detected motor | detected cause | latency [ms] | early |
|---|---|---|---|---|---|---|---|---|
| test_00 | hover | stop | 1.0 | 1 | 1 | motor_stop | 100 |  |
| test_01 | hover | stop | 1.0 | 2 | 2 | motor_stop | 100 |  |
| test_02 | hover | weak | 1.0 | 3 | 3 | motor_stop | 100 |  |
| test_03 | hover | weak | 0.85 | 4 | 4 | weak_motor | 100 |  |
| test_04 | hover | prop | 1.0 | 1 | 1 | prop_loss | 180 |  |
| test_05 | hover | prop | 0.86 | 2 | 2 | prop_loss | 180 |  |
| test_06 | forward | stop | 1.0 | 3 | 3 | motor_stop | 100 |  |
| test_07 | forward | stop | 1.0 | 4 | 4 | motor_stop | 100 |  |
| test_08 | forward | weak | 1.0 | 1 | 1 | motor_stop | 100 |  |
| test_09 | forward | weak | 0.75 | 2 | 2 | weak_motor | 100 |  |
| test_10 | forward | prop | 1.0 | 3 | 3 | prop_loss | 159 |  |
| test_11 | forward | prop | 0.88 | 4 | 4 | prop_loss | 180 |  |
| test_12 | climb | stop | 1.0 | 1 | 1 | motor_stop | 100 |  |
| test_13 | descent | stop | 1.0 | 2 | 2 | motor_stop | 100 |  |
| test_14 | climb | weak | 1.0 | 3 | 3 | motor_stop | 100 |  |
| test_15 | descent | weak | 0.71 | 4 | 4 | weak_motor | 100 |  |
| test_16 | climb | prop | 1.0 | 1 | 1 | prop_loss | 180 |  |
| test_17 | descent | prop | 0.76 | 2 | 2 | prop_loss | 180 |  |
| test_18 | yaw | stop | 1.0 | 3 | 3 | motor_stop | 100 |  |
| test_19 | yaw | stop | 1.0 | 4 | 4 | motor_stop | 100 |  |
| test_20 | yaw | weak | 1.0 | 1 | 1 | motor_stop | 100 |  |
| test_21 | yaw | weak | 0.85 | 2 | 2 | weak_motor | 100 |  |
| test_22 | yaw | prop | 1.0 | 3 | 3 | prop_loss | 180 |  |
| test_23 | yaw | prop | 0.81 | 4 | 4 | prop_loss | 160 |  |
