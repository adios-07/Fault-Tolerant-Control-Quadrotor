# Heading error in the spin vs gyro scale error (median over 16 runs, i022c on GT and onboard state)

| gyro scale error | heading source | heading err +2 s [deg] | heading err +5 s [deg] | heading err +10 s [deg] | heading err +15 s [deg] |
|---|---|---|---|---|---|
| scale 0 % | gyro only | 6.0 (max 7) | 6.3 (max 7) | 6.2 (max 7) | 6.2 (max 7) |
| scale 0 % | gyro + mag heading | 21.3 (max 51) | 5.7 (max 12) | 6.6 (max 23) | 7.2 (max 19) |
| scale 1 % | gyro only | 9.1 (max 16) | 36.4 (max 46) | 90.6 (max 111) | 148.2 (max 177) |
| scale 1 % | gyro + mag heading | 18.8 (max 32) | 6.4 (max 14) | 6.7 (max 17) | 7.0 (max 33) |
| scale 3 % | gyro only | 28.6 (max 36) | 109.5 (max 126) | 81.5 (max 131) | 87.8 (max 178) |
| scale 3 % | gyro + mag heading | 13.5 (max 37) | 14.1 (max 24) | 12.0 (max 38) | 13.0 (max 44) |
| scale 0 % | PX4 EKF-GSF yaw | 89.0 (max 106) | 93.0 (max 153) | 124.8 (max 154) | 137.8 (max 157) |
