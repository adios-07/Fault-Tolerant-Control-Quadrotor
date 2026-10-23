# IDEA-022 step B offline: gyro attitude + position/velocity KF vs EKF2 (vs truth, from the fault)

| flights | GPS | runs | velocity RMS 0-5 s, gyro att + KF [m/s] | EKF2 [m/s] |
|---|---|---|---|---|
| policy on GT (15 s) | SITL GPS (~30 Hz) | 8 | 0.10 (max 0.13) | 4.07 |
| policy on GT (15 s) | GPS 10 Hz + 100 ms | 8 | 0.23 (max 0.28) | 4.07 |
| policy on EKF (crash ~2 s) | SITL GPS (~30 Hz) | 8 | 0.29 (max 0.76) | 2.72 |
| policy on EKF (crash ~2 s) | GPS 10 Hz + 100 ms | 8 | 0.47 (max 0.85) | 2.72 |
