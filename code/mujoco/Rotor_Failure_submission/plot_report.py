#!/usr/bin/env python3
"""README figures from the run logs: training curves of the i025a chain and the precision / robustness trade-off.

  python plot_report.py   ->  report/figures/training_curves.png, report/figures/pareto.png, report/pareto.md
"""
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path(__file__).parent / "runs"
OUT = Path(__file__).parent / "report"
CHAIN = ["i014a_m0", "i018", "i019c", "i020a", "i021c", "i022c", "i023b", "i025a"]  # PPO stages (i016 = supervised selector)
LINE = re.compile(r"([\d.]+)M steps .* reward ([\d.]+) .* crash ([\d.]+)")


def curve(run):
    pts = [tuple(map(float, m.groups())) for m in map(LINE.search, (RUNS / run / "log.txt").read_text().splitlines()) if m]
    return np.array(pts)  # columns: M steps, mean reward per step, crash share of finished episodes


def training_curves():
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    off = 0.0
    for i, run in enumerate(CHAIN):
        c = curve(run)
        x = off + c[:, 0]
        for a, col in zip(ax, (1, 2)):
            a.plot(x, c[:, col], lw=1.2, color=f"C{i % 10}")
            a.axvline(off, color="grey", lw=0.5, ls=":")
        ax[0].text(off + 2, 1.47, run, fontsize=7, rotation=90, va="top")
        off = x[-1]
    ax[0].set_ylabel("mean reward per step")
    ax[1].set_ylabel("crash share of\nfinished episodes")
    ax[1].set_xlabel("PPO environment steps, cumulative over the chain [millions]")
    ax[0].set_title("Training chain to i025a: each stage fine-tunes the previous one.\nReward terms and randomisation "
                    "change between stages, so levels are not comparable across stages.", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures/training_curves.png", dpi=110)
    plt.close(fig)


def pareto():
    """Precision = mean xy RMS of gt_stop/gt_prop; robustness = crashes summed over the settings every candidate has,
    excluding the 1 % gyro-scale onboard cases (they drift away by design: heading error, not the policy)."""
    data = {p.parent.name: json.loads(p.read_text()) for p in sorted(RUNS.glob("*/bench_all.json"))}
    data = {r: d for r, d in data.items() if "part50_stop" in d}  # only candidates scored on partial loss too
    keys = set.intersection(*(set(d) for d in data.values()))
    keys = sorted(k for k in keys if not (k.startswith("ob_scale") or k.split("_")[1] in ("hover", "forward", "climb", "descent", "yaw")
                                          and k.startswith("ob_")))
    pts = {r: (np.mean([d["gt_stop"]["xy_rms_last5"], d["gt_prop"]["xy_rms_last5"]]), sum(d[k]["crashed"] for k in keys))
           for r, d in data.items()}
    front = [r for r, (x, y) in pts.items() if not any(x2 <= x and y2 <= y and (x2, y2) != (x, y) for x2, y2 in pts.values())]
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, (x, y) in pts.items():
        ax.scatter(x, y, color="C3" if r == "i025a" else ("C0" if r in front else "grey"), zorder=3)
        ax.annotate(r, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("hover position error after a complete loss: xy RMS last 5 s [m]")
    ax.set_ylabel(f"crashes summed over {len(keys)} robustness settings\n(x 20 runs each)")
    ax.set_title("Precision vs robustness of the candidate policies (blue: Pareto front, red: chosen)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures/pareto.png", dpi=110)
    plt.close(fig)
    rows = ["| policy | xy RMS [m] | crashes | on front |", "|---|---|---|---|"]
    rows += [f"| {r} | {x:.3f} | {y} | {'yes' if r in front else ''} |" for r, (x, y) in sorted(pts.items(), key=lambda t: t[1][1])]
    (OUT / "pareto.md").write_text(f"Settings ({len(keys)}): {', '.join(keys)}\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    training_curves()
    pareto()
