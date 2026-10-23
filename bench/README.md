# Benchmark harness

Scripts that fly the seeded fault schedule (`run_bench.py`, `phase_flyer.py`, `schedule.json`) and score the recorded flights (`replay.py`, `analyze.py`, `ftc_analyze.py`, `estimation.py`, `plot_xyz.py`). Usage is in `code/px4_ws/src/rotor_fail/README.md`.
The per-run data (`runs/`: `.ulg` logs and `inputs.pkl`) and the curated `logs/` were not shipped. `analyze.py`, `replay.py`, `ftc_analyze.py`, `estimation.py` and `plot_xyz.py` read `runs/` and cannot run until you fly the benchmark again with `run_bench.py`. Only `run_bench.py` and `phase_flyer.py` work on a fresh checkout (they need the PX4 and Gazebo setup).
The recorded outputs of the original flights are in `results/` (tables in `.md`, plots in `.png`, `base.json` and `tuned.json` from `analyze.py`).
