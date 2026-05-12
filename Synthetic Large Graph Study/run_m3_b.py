"""
run_m3_b.py
===========

Milestone 3 Group B: partitioning-strategy comparison on the 2M-node
synthetic graph at a fixed worker count (w=8). All three strategies use
incremental_pagerank.run_full_recomputation; only the partition_strategy
field of M3Config changes.

Runs:
  1. range partition          (M2 baseline)
  2. hash partition           (node_id % k)
  3. edge_balanced partition  (greedy LPT on |in_links[u]|)
"""

from __future__ import annotations

import json
import logging

import ray

from common import (
    BIG_GRAPH_PATH, OUTPUT_DIR, build_m3_config,
    configure_logging, setup_paths,
)

setup_paths()
from incremental_pagerank import run_full_recomputation  # noqa: E402

log = configure_logging()
GB_OUT = OUTPUT_DIR / "m3_groupB"
GB_OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 8


def main() -> None:
    if not BIG_GRAPH_PATH.exists():
        raise SystemExit(f"Missing {BIG_GRAPH_PATH}; run generate_graphs.py first.")
    snap = str(BIG_GRAPH_PATH)

    summary = []
    for strategy in ("range", "hash", "edge_balanced"):
        run_name = f"{strategy}_w{WORKERS}"
        log.info("=== Group B / %s ===", run_name)
        ray.init(num_cpus=WORKERS, ignore_reinit_error=True,
                 logging_level=logging.WARNING, include_dashboard=False)
        try:
            cfg = build_m3_config(num_workers=WORKERS, partition_strategy=strategy)
            _, stats = run_full_recomputation(
                snap, cfg, output_dir=str(GB_OUT / run_name), run_name=run_name,
            )
        finally:
            ray.shutdown()
        t = stats.total_wall_time_s
        summary.append({
            "name": run_name, "strategy": strategy, "workers": WORKERS,
            "elapsed_s": t,
            "iterations": stats.extras.get("num_iterations", -1),
            "avg_load_imbalance": stats.avg_load_imbalance,
            "total_barrier_wait_s": stats.total_barrier_wait_s,
            "total_comm_kb": stats.total_comm_bytes / 1024.0,
        })
        log.info("%s done: %.1fs imbalance=%.1f%%",
                 run_name, t, 100 * stats.avg_load_imbalance)

    with open(GB_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    for r in summary:
        log.info("  %-20s elapsed=%.1fs imbalance=%.1f%%",
                 r["name"], r["elapsed_s"], 100 * r["avg_load_imbalance"])


if __name__ == "__main__":
    main()
