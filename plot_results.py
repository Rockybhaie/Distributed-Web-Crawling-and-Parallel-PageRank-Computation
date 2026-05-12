
# plot_results.py: Generate publication-quality figures from saved metrics.
#
# Reads:
#   <input_dir>/groupA_summary.json
#   <input_dir>/groupB_summary.json
#   <input_dir>/groupC_summary.json
#   <input_dir>/<run>/metrics.json   (per-iteration timeseries)
#
# Writes (to <output_dir>, default output/figures/):
#   fig_groupA_speedup.{png,pdf}
#   fig_groupA_time_bars.{png,pdf}
#   fig_groupA_convergence.{png,pdf}
#   fig_groupB_imbalance.{png,pdf}
#   fig_groupB_comm.{png,pdf}
#   fig_groupB_time.{png,pdf}
#   fig_groupC_time_vs_accuracy.{png,pdf}
#   fig_groupC_iterations.{png,pdf}
#   fig_groupC_dirty_fraction.{png,pdf}
#   fig_memory_timeline.{png,pdf}


import json
import logging
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Apply a consistent house style.
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       100,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})



# Helpers

def _load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig, output_dir: str, basename: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"{basename}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    logger.info("Saved %s -> %s", basename, paths[0])
    return paths



# GROUP A figures

def plot_group_a(input_dir: str, output_dir: str) -> None:
    summary = _load_json(os.path.join(input_dir, "groupA", "groupA_summary.json"))
    if not summary:
        logger.warning("Group A summary not found - skipping Group A plots.")
        return

    seq_time = next((r["elapsed_s"] for r in summary if r["name"] == "sequential"), None)

    # 1. Time bar chart
    names = [r["name"] for r in summary]
    times = [r["elapsed_s"] for r in summary]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), times,
                  color=["#888"] + ["#2b8cbe"] * (len(names) - 1))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Group A - Sequential vs Parallel PageRank")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, output_dir, "fig_groupA_time_bars")

    # 2. Speedup curve
    parallel = [r for r in summary if r["workers"] >= 1 and r["name"] != "sequential"]
    if parallel and seq_time:
        ws = [r["workers"] for r in parallel]
        sp = [r["speedup"] for r in parallel]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ws, sp, "o-", color="#2b8cbe", label="Observed speedup")
        ax.plot(ws, ws, "--", color="#aaa", label="Ideal linear speedup")
        ax.set_xlabel("Number of workers")
        ax.set_ylabel("Speedup vs sequential")
        ax.set_title("Group A - Worker scaling (parallel centralised, range partition)")
        ax.legend()
        ax.set_xticks(ws)
        _save_fig(fig, output_dir, "fig_groupA_speedup")

    # 3. Convergence comparison (delta per iteration, log scale)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in summary:
        m = _load_json(os.path.join(r["dir"], "metrics.json"))
        if not m:
            continue
        iters = [it["iteration"] for it in m.get("iterations", [])]
        deltas = [max(it["delta"], 1e-300) for it in m.get("iterations", [])]
        ax.semilogy(iters, deltas, "o-", label=r["name"], markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Convergence delta (log scale)")
    ax.set_title("Group A - Convergence rate per strategy")
    ax.legend()
    _save_fig(fig, output_dir, "fig_groupA_convergence")



# GROUP B figures

def plot_group_b(input_dir: str, output_dir: str) -> None:
    summary = _load_json(os.path.join(input_dir, "groupB", "groupB_summary.json"))
    if not summary:
        logger.warning("Group B summary not found - skipping Group B plots.")
        return

    strategies = [r["partition_strategy"] for r in summary]
    times = [r["elapsed_s"] for r in summary]
    imbal = [r["avg_load_imbalance"] * 100.0 for r in summary]
    comm = [r["total_comm_kb"] for r in summary]
    rss = [r["peak_worker_rss_mb"] for r in summary]

    colors = ["#2b8cbe", "#fc8d59", "#74c476"]

    # 1. Time bars
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(strategies, times, color=colors[:len(strategies)])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Group B - Total time per partitioning strategy")
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, output_dir, "fig_groupB_time")

    # 2. Load imbalance bars
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(strategies, imbal, color=colors[:len(strategies)])
    ax.set_ylabel("Average load imbalance (%)")
    ax.set_title("Group B - Load imbalance: 0% perfectly balanced, 100% extreme")
    for b, v in zip(bars, imbal):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, output_dir, "fig_groupB_imbalance")

    # 3. Communication volume bars
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(strategies, comm, color=colors[:len(strategies)])
    ax.set_ylabel("Total communication volume (KB)")
    ax.set_title("Group B - Estimated bytes serialised through Ray")
    for b, v in zip(bars, comm):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, output_dir, "fig_groupB_comm")

    # 4. Memory bars
    if any(rss):
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(strategies, rss, color=colors[:len(strategies)])
        ax.set_ylabel("Peak per-worker RSS (MB)")
        ax.set_title("Group B - Peak memory per worker")
        for b, v in zip(bars, rss):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        _save_fig(fig, output_dir, "fig_groupB_memory")



# GROUP C figures

def plot_group_c(input_dir: str, output_dir: str) -> None:
    summary = _load_json(os.path.join(input_dir, "groupC", "groupC_summary.json"))
    if not summary:
        logger.warning("Group C summary not found - skipping Group C plots.")
        return

    transitions = [r["transition"] for r in summary]
    full_t = [r["full"]["elapsed_s"] for r in summary]
    warm_t = [r["warm"]["elapsed_s"] for r in summary]
    loc_t  = [r["localised"]["elapsed_s"] for r in summary]

    full_iter = [r["full"]["iterations"] for r in summary]
    warm_iter = [r["warm"]["iterations"] for r in summary]
    loc_iter  = [r["localised"]["iterations"] for r in summary]

    warm_acc = [r["warm"]["accuracy"]["max_abs_diff"] for r in summary]
    loc_acc  = [r["localised"]["accuracy"]["max_abs_diff"] for r in summary]

    dirty = [r["localised"].get("dirty_node_count") or 0 for r in summary]
    added = [r["delta_summary"].get("added_nodes", 0) for r in summary]
    new_total = [r["delta_summary"].get("new_node_count", 0) for r in summary]
    dirty_pct = [(d / nt * 100.0) if nt else 0.0
                 for d, nt in zip(dirty, new_total)]

    x = list(range(len(transitions)))
    width = 0.27

    # 1. Time bars: full vs warm vs localised, grouped per transition
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], full_t, width, color="#888",  label="Full (cold)")
    ax.bar(x, warm_t, width, color="#2b8cbe", label="Warm-start")
    ax.bar([i + width for i in x], loc_t, width, color="#74c476", label="Localised")
    ax.set_xticks(x)
    ax.set_xticklabels(transitions)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Group C - Time per update strategy across snapshots")
    ax.legend()
    _save_fig(fig, output_dir, "fig_groupC_time_strategies")

    # 2. Time vs accuracy scatter (the headline plot)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, t in enumerate(transitions):
        ax.scatter(full_t[i], 0.0, s=80, color="#888",
                   marker="o", label="Full (cold)" if i == 0 else None)
        ax.scatter(warm_t[i], warm_acc[i], s=80, color="#2b8cbe",
                   marker="s", label="Warm-start" if i == 0 else None)
        ax.scatter(loc_t[i], loc_acc[i], s=80, color="#74c476",
                   marker="^", label="Localised" if i == 0 else None)
        ax.annotate(t, (full_t[i], 0.0), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#666")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Accuracy degradation: max |score_strategy - score_full|")
    ax.set_title("Group C - Cost vs Accuracy Trade-off")
    ax.legend()
    if any(a > 0 for a in (warm_acc + loc_acc)):
        ax.set_yscale("symlog", linthresh=1e-8)
    _save_fig(fig, output_dir, "fig_groupC_time_vs_accuracy")

    # 3. Iterations per strategy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], full_iter, width, color="#888",  label="Full (cold)")
    ax.bar(x, warm_iter, width, color="#2b8cbe", label="Warm-start")
    ax.bar([i + width for i in x], loc_iter, width, color="#74c476", label="Localised")
    ax.set_xticks(x)
    ax.set_xticklabels(transitions)
    ax.set_ylabel("Iterations to convergence")
    ax.set_title("Group C - Iterations per update strategy")
    ax.legend()
    _save_fig(fig, output_dir, "fig_groupC_iterations")

    # 4. Dirty node fraction across snapshots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(transitions, dirty_pct, color="#74c476")
    ax.set_ylabel("Dirty nodes (% of total)")
    ax.set_title("Group C - Localised update: dirty-set size as graph grows")
    for i, v in enumerate(dirty_pct):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    _save_fig(fig, output_dir, "fig_groupC_dirty_fraction")

    # 5. Accuracy bars (warm vs localised)
    if any(warm_acc + loc_acc):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([i - width / 2 for i in x], warm_acc, width,
               color="#2b8cbe", label="Warm-start")
        ax.bar([i + width / 2 for i in x], loc_acc, width,
               color="#74c476", label="Localised")
        ax.set_xticks(x)
        ax.set_xticklabels(transitions)
        ax.set_ylabel("Max |score_strategy - score_full|")
        ax.set_title("Group C - Accuracy degradation per snapshot")
        ax.legend()
        ax.set_yscale("log")
        _save_fig(fig, output_dir, "fig_groupC_accuracy")



# Memory timeline (across any single run)

def plot_memory_timeline(input_dir: str, output_dir: str) -> None:
    paths = glob(os.path.join(input_dir, "**", "metrics.json"), recursive=True)
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for p in paths[:8]:
        m = _load_json(p)
        if not m:
            continue
        iters = [it["iteration"] for it in m.get("iterations", [])]
        rss = [it["driver_rss_mb"] for it in m.get("iterations", [])]
        if iters:
            ax.plot(iters, rss, "-",
                    label=m.get("run_name", os.path.basename(os.path.dirname(p))))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Driver RSS (MB)")
    ax.set_title("Driver memory across iterations (sample of runs)")
    ax.legend(fontsize=7, loc="best")
    _save_fig(fig, output_dir, "fig_memory_timeline")



# Top-level entry point

def generate_all_figures(input_dir: str = "output",
                         output_dir: str = "output/figures") -> None:
    logger.info("Generating figures from %s -> %s", input_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_group_a(input_dir, output_dir)
    plot_group_b(input_dir, output_dir)
    plot_group_c(input_dir, output_dir)
    plot_memory_timeline(input_dir, output_dir)
    logger.info("All figures generated in %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    generate_all_figures()
