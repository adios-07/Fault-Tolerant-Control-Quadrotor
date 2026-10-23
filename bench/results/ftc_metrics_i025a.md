# Fault-tolerant flight: learned policy after detection (i025a)

Complete loss of one rotor (`stop`: motor dead; `prop`: propeller gone, motor spinning) in hover. PX4 flies until the detector declares the failure, then the policy takes the motors (offboard direct actuator). Scored against Gazebo ground truth from the handover to fault + 15 s. Crash = height above ground < 0.3 m.

> **No .ulg logs found in `runs/`.** Per-run logs were pruned for the submission (3.1 GB); the curated logs are not shipped, and the committed tables in `bench/results/` are the record. Re-fly the batch to regenerate this file:

```bash
export RTF=2 FTC_DEBUG_CSV=1
RUN_TAG=i025a python3 run_bench.py ftc
RUN_TAG=i025a_gyro python3 run_bench.py data
RUN_TAG=i025a python3 run_bench.py partial
FTC_TAIL=35 RUN_TAG=land python3 run_bench.py ftc
python3 ftc_analyze.py i025a; python3 ftc_analyze.py i025a_gyro
```

