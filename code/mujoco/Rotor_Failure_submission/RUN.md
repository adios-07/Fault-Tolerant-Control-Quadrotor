# How to run

All commands run from this folder (`code/mujoco/Rotor_Failure_submission` in the repo) with a venv named `.venv` activated, so `python` is the venv's Python.
Everything uses the submitted policy `runs/i025a/policy.pkl` (= `policies/x500v2_i025a/policy.onnx`).

## 1. Install (once)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
git clone https://github.com/learnsyslab/crazyflow && git -C crazyflow checkout c95f69b
git -C crazyflow apply ../third_party/crazyflow_x500v2.patch   # adds the x500v2 drone
pip install -e crazyflow -r requirements.txt
```

crazyflow is not shipped in this repo: clone it at commit `c95f69b` and apply `third_party/crazyflow_x500v2.patch`
as shown (the clone lands in `./crazyflow`).

**No NVIDIA GPU:** install `jax==0.11.2` instead of `jax[cuda13]`, and put `JAX_PLATFORMS=cpu` in front of every
command. Expect it to be about 2-5x slower.

## 2. Check the delivered policy (seconds)

```bash
cd policies/x500v2_i025a && sha256sum -c SHA256SUMS && python3 selftest.py && cd ../..   # must print PASS
```

## 3. Reproduce the report (each under a few minutes on a GPU)

| Command | What it gives | README section |
|---|---|---|
| `python run_benchmark.py` | Task 1d: stock controller vs PFTC, rotor 0 stopped. KPIs in `report/benchmark_kpis.md`, plot `report/figures/benchmark_motor0_stop.png` | Task 1d |
| `python run_benchmark.py --prop` | the same with a lost propeller | Task 1d |
| `python relaxed_hover.py --sim --cg` | Task 1b: allocation, yaw torque, relaxed-hover equilibrium vs simulation, CG feasibility -> `report/relaxed_hover.txt` | Task 1b |
| `python plot_report.py` | training curves and Pareto plot -> `report/figures/`, `report/pareto.md` | Training, Track A |
| `python bench_sim.py runs/i025a/policy.pkl --out bench.json` | every setting of the PX4-bench replica (fault types, phases, partial loss, airframe errors, estimator errors) | Replica tables |
| `python bench_sim.py runs/i025a/policy.pkl --only gt_stop part50_stop hw_all_stop` | a subset of those settings | – |
| `python stress.py runs/i025a/policy.pkl` | breaking points + Monte Carlo (`--mc-scale 0.5` for the moderate set) | Sim vs real |
| `python main.py --motor 1` | MuJoCo viewer: hover, rotor 1 fails at 3 s, spinning recovery (Ctrl+C to quit) | – |
| `python train.py --id viz_m0 --steps 6e7 --envs 2048 --motor 0 --p-fault 0.5 --snaps` then `python training_video.py runs/viz_m0` | video of a policy learning from scratch: snapshot rollouts next to the reward / crash curve -> `runs/viz_m0/training.mp4` | – |
| `python export_onnx.py runs/i025a/policy.pkl /tmp/policy.onnx` | ONNX export, checked against JAX (< 1e-5) | Deliverable |

The training video made with `training_video.py` (after `train.py --snaps`) is shown in
[`../../../media/mujoco/training.gif`](../../../media/mujoco/training.gif).

## 4. Retrain

The final stage, about 76 min on an RTX 4060:

```bash
python train.py --id i025a_retrain --steps 1.5e8 --envs 16384 --init runs/i023b/policy.pkl --joint \
  --reset-std 0.1 --lr 1e-4 --p-fault 0.5 --gz-rand --reward precise --tilt-pen 1 --moving 0.7 --handover 120 \
  --delay-min 0 --prop-j 0.08 1.2 --p-spin 0.5 --hw-rand --onboard 0.8 3 0.5 4 45 --equivariant --partial 0.3 \
  --severity 0.3 0.8
python bench_sim.py runs/i025a_retrain/policy.pkl --only gt_stop gt_prop part50_stop   # score it
```

The whole chain from scratch is in `README.md`, Reproducibility. The weights of every stage are in `runs/<id>/`,
with each stage's arguments on line 1 of `runs/<id>/log.txt`.

## Files

| Path | What it is |
|---|---|
| `ftc.py` | environment |
| `train.py` | PPO and the network |
| `eval.py` | Mellinger baseline and the 20-unit score |
| `bench_sim.py` | PX4-bench replica |
| `stress.py` | stress test |
| `relaxed_hover.py` | maths |
| `run_benchmark.py` | Task 1d benchmark |
| `plot_report.py` | figures |
| `export_onnx.py` | ONNX export |
| `pftc_ref.py` | deployable runner |
| `main.py` | viewer |
| `README.md` | the full report |
| `FINAL_POLICY.md` | the submitted controller's spec sheet |
