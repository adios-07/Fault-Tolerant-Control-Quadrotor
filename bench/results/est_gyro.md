# IDEA-023: raw-gyro dead reckoning vs EKF2 (mean tilt / yaw error vs truth, deg)

SITL gyro (near ideal); scale / bias rows inject errors to mimic a real IMU.

Seed (EKF attitude 150 ms before detection) tilt error vs truth: max 0.06 deg over 22 runs.

| runs | estimator | tilt / yaw at +0.5 s | tilt / yaw at +1 s | tilt / yaw at +2 s | tilt / yaw at +5 s | tilt / yaw at +10 s |
|---|---|---|---|---|---|---|
| policy runs (15 s) | EKF2 | 5.6 / 1.8 | 22.1 / 21.4 | 44.4 / 42.1 | 100.5 / 53.1 | 90.2 / 40.1 |
| policy runs (15 s) | ideal gyro | 6.5 / 0.5 | 2.1 / 6.1 | 4.3 / 4.2 | 2.8 / 5.4 | 2.9 / 6.4 |
| policy runs (15 s) | scale +1 % | 6.5 / 0.9 | 2.6 / 6.1 | 7.5 / 6.6 | 11.1 / 28.3 | 21.5 / 57.1 |
| policy runs (15 s) | scale +3 % | 6.5 / 2.8 | 5.5 / 10.7 | 22.9 / 19.2 | 30.3 / 88.2 | 38.6 / 61.6 |
| policy runs (15 s) | bias 0.5 deg/s | 6.7 / 0.5 | 2.3 / 6.5 | 4.9 / 4.7 | 3.2 / 6.5 | 3.7 / 8.1 |
| policy runs (15 s) | bias 2 deg/s | 7.3 / 1.0 | 3.0 / 7.6 | 6.8 / 6.3 | 4.9 / 10.2 | 6.9 / 16.0 |
| 50 % fault runs (PX4 flies, ~3 s) | EKF2 | 1.3 / 4.6 | 0.4 / 4.3 | 0.4 / 3.2 | - | - |
| 50 % fault runs (PX4 flies, ~3 s) | ideal gyro | 1.4 / 4.7 | 0.7 / 5.3 | 0.8 / 5.2 | - | - |
| 50 % fault runs (PX4 flies, ~3 s) | scale +1 % | 1.4 / 4.8 | 0.9 / 5.5 | 0.6 / 5.7 | - | - |
| 50 % fault runs (PX4 flies, ~3 s) | scale +3 % | 1.5 / 5.0 | 1.9 / 5.9 | 1.4 / 6.7 | - | - |
| 50 % fault runs (PX4 flies, ~3 s) | bias 0.5 deg/s | 1.6 / 4.9 | 1.1 / 5.7 | 1.8 / 6.1 | - | - |
| 50 % fault runs (PX4 flies, ~3 s) | bias 2 deg/s | 2.6 / 5.5 | 3.1 / 7.0 | 5.6 / 8.5 | - | - |
