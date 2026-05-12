"""
run_m3_a.py
===========

Milestone 3 Group A: Sequential vs Parallel scaling on the 2M-node
synthetic graph. Reuses incremental_pagerank.run_sequential and
run_full_recomputation from Milestone 3 (unchanged).

Runs:
  1. Sequential (run_sequential)
  2. Parallel range, w=1
  3. Parallel range, w=2
  4. Parallel range, w=4
  5. Parallel range, w=8

Speedup = sequential_time / parallel_time. The headline figure of this
whole study lives here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ray

from common import (
    BIG_GRAPH_PATH, OUTPUT_DIR, build_m3_config,
    configure_logging, setup_paths,
)

setup_paths()
from incremental_pagerank import run_sequential, run_full_recomputation  # noqa: E402

log = configure_logging()
GA_OUT = OUTPUT_DIR / "m3_groupA"
GA_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not BIG_GRAPH_PATH.exists():
        raise SystemExit(f"Missing {BIG_GRAPH_PATH}; run generate_graphs.py first.")
    snap = str(BIG_GRAPH_PATH)

    summary = []

    # 1. Sequential
    log.info("=== Group A / sequential ===")
    seq_cfg = build_m3_config(num_workers=1, partition_strategy="range")
    _, seq_stats = run_sequential(snap, seq_cfg,
                                  output_dir=str(GA_OUT / "sequential"),
                                  run_name="sequential")
    seq_t = seq_stats.total_wall_time_s
    summary.append({"name": "sequential", "workers": 1, "strategy": "n/a",
                    "elapsed_s": seq_t, "iterations": seq_stats.extras.get("num_iterations", -1),
                    "speedup": 1.0})

    # 2-5. Parallel scaling
    for w in (1, 2, 4, 8):
        run_name = f"parallel_range_w{w}"
        log.info("=== Group A / %s ===", run_name)
        ray.init(num_cpus=w, ignore_reinit_error=True,
                 logging_level=logging.WARNING, include_dashboard=False)
        try:
            cfg = build_m3_config(num_workers=w, partition_strategy="range")
            _, stats = run_full_recomputation(snap, cfg,
                                              output_dir=str(GA_OUT / run_name),
                                              run_name=run_name)
        finally:
            ray.shutdown()
        t = stats.total_wall_time_s
        summary.append({
            "name": run_name, "workers": w, "strategy": "range",
            "elapsed_s": t, "iterations": stats.extras.get("num_iterations", -1),
            "speedup": seq_t / t if t > 0 else 0.0,
            "efficiency": (seq_t / t) / w if t > 0 else 0.0,
        })
        log.info("%s done: %.1fs speedup=%.2fx",
                 run_name, t, seq_t / t if t > 0 else 0)

    with open(GA_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    for r in summary:
        log.info("  %-22s w=%d elapsed=%.1fs speedup=%.2fx",
                 r["name"], r["workers"], r["elapsed_s"], r["speedup"])


if __name__ == "__main__":
    main()
