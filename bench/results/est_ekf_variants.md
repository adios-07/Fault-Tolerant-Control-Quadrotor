# IDEA-021: EKF2 magnetometer configurations, policy on EKF state (mean over runs)

| EKF2 config | runs | survived | tilt RMS 0-5 s [deg] | yaw RMS 0-5 s [deg] | tilt at +1 s [deg] | crash at [s] |
|---|---|---|---|---|---|---|
| default (MAG_TYPE 0, auto) | 8 | 0/8 | 32.0 (29.2-39.4) | 22.0 | 22.2 | 1.9-2.4 |
| MAG_TYPE 1, heading only | 8 | 0/8 | 29.8 (4.8-55.9) | 19.2 | 19.8 | 0.4-2.8 |
| MAG_TYPE 6, init only | 8 | 0/8 | 30.9 (4.3-43.9) | 19.0 | 21.3 | 0.4-2.4 |
