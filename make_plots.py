"""
make_plots.py
=============

Generates every figure consumed by the LaTeX report from the summary.json
files written by run_m2.py / run_m3_a.py / run_m3_b.py / run_m3_c.py.

Outputs go to report/figures/ as both PNG (for previewing) and PDF (for
the actual paper). matplotlib only - no seaborn dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(p: Path):
    with open(p) as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"saved {name}.pdf/.png")


# ---------- Figure 1: M3 Group A scaling curve ----------

def plot_group_a():
    path = OUT / "m3_groupA" / "summary.json"
    if not path.exists():
        print(f"skip group A: {path} missing")
        return
    data = _load(path)
    seq = next(r for r in data if r["name"] == "sequential")
    par = [r for r in data if r["name"] != "sequential"]
    par.sort(key=lambda r: r["workers"])
    workers = [r["workers"] for r in par]
    speedups = [r["speedup"] for r in par]
    elapsed = [r["elapsed_s"] for r in par]

    fig, axL = plt.subplots(figsize=(6.4, 4.0))
    axL.plot(workers, speedups, "o-", color="#1f77b4", linewidth=2,
             markersize=8, label="Measured speedup")
    axL.plot(workers, workers, "--", color="#888888", label="Ideal (linear)")
    axL.axhline(1.0, color="#d62728", linestyle=":",
                label="Sequential (1x)")
    axL.set_xlabel("Number of Ray workers")
    axL.set_ylabel("Speedup over sequential")
    axL.set_title(
        f"Group A: parallel scaling on 2M-node synthetic graph\n"
        f"(sequential baseline = {seq['elapsed_s']:.1f}s)"
    )
    axL.set_xticks(workers)
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper left")
    _save(fig, "groupA_scaling")

    # Companion bar chart of elapsed time.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["seq"] + [f"w={w}" for w in workers]
    times = [seq["elapsed_s"]] + elapsed
    colors = ["#d62728"] + ["#1f77b4"] * len(workers)
    ax.bar(labels, times, color=colors)
    for i, t in enumerate(times):
        ax.text(i, t * 1.01, f"{t:.0f}s", ha="center", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Group A: wall-clock time vs worker count (2M nodes)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "groupA_times")


# ---------- Figure 2: M3 Group B partitioning ----------

def plot_group_b():
    path = OUT / "m3_groupB" / "summary.json"
    if not path.exists():
        print(f"skip group B: {path} missing")
        return
    data = _load(path)
    data.sort(key=lambda r: r["elapsed_s"])
    strategies = [r["strategy"] for r in data]
    times = [r["elapsed_s"] for r in data]
    imbalance = [100 * r["avg_load_imbalance"] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    bars = ax1.bar(strategies, times, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    for b, t in zip(bars, times):
        ax1.text(b.get_x() + b.get_width() / 2, t * 1.01, f"{t:.0f}s",
                 ha="center", fontsize=9)
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Elapsed time by partitioning strategy (w=8, 2M nodes)")
    ax1.grid(True, alpha=0.3, axis="y")

    bars = ax2.bar(strategies, imbalance, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    for b, v in zip(bars, imbalance):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.1f}%",
                 ha="center", fontsize=9)
    ax2.set_ylabel("Avg load imbalance (%)")
    ax2.set_title("Per-iteration load imbalance")
    ax2.grid(True, alpha=0.3, axis="y")
    _save(fig, "groupB_partition")


# ---------- Figure 3: M3 Group C incremental strategies ----------

def plot_group_c():
    path = OUT / "m3_groupC" / "summary.json"
    if not path.exists():
        print(f"skip group C: {path} missing")
        return
    data = _load(path)
    # Group by version, only v1/v2 are interesting (v0 is just baseline).
    versions = ["v1", "v2"]
    strategies = ["full", "warm", "localised"]
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, strat in enumerate(strategies):
        vals = []
        for v in versions:
            row = next((r for r in data if r["version"] == v and r["strategy"] == strat), None)
            vals.append(row["elapsed_s"] if row else 0.0)
        xs = [j + i * width for j in range(len(versions))]
        bars = ax.bar(xs, vals, width=width, label=strat)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + width / 2, val * 1.01, f"{val:.0f}s",
                    ha="center", fontsize=8)
    ax.set_xticks([j + width for j in range(len(versions))])
    ax.set_xticklabels([f"{v} (+~10%)" for v in versions])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Group C: full vs warm vs localised PageRank "
                 "(800K-1M synthetic series)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "groupC_update_strategies")


# ---------- Figure 4: M2 centralised vs distributed ----------

def plot_m2():
    path = OUT / "m2" / "summary.json"
    if not path.exists():
        print(f"skip M2: {path} missing")
        return
    data = _load(path)
    seq = next(r for r in data if r["strategy"] == "sequential")
    workers = sorted({r["workers"] for r in data if r["strategy"] != "sequential"})
    def _lookup(strategy, w):
        for r in data:
            if r["strategy"] == strategy and r["workers"] == w:
                return r["elapsed_s"]
        return 0.0
    cent = [_lookup("centralized", w) for w in workers]
    dist = [_lookup("distributed", w) for w in workers]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = list(range(len(workers)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], cent, width, label="Centralised", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], dist, width, label="Distributed (n/a if 0)", color="#2ca02c")
    ax.axhline(seq["elapsed_s"], color="#d62728", linestyle="--",
               label=f"Sequential ({seq['elapsed_s']:.0f}s)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}" for w in workers])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("M2: centralised vs distributed parallel PageRank (2M nodes)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "m2_strategies")


def main() -> None:
    plot_m2()
    plot_group_a()
    plot_group_b()
    plot_group_c()


if __name__ == "__main__":
    main()
