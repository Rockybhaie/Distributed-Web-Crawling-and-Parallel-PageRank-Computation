"""
run_m3_c.py
===========

Milestone 3 Group C: static vs incremental PageRank update strategies on
the 1M-scale synthetic series (v0=800K -> v1=900K -> v2=1M). For each
new snapshot we run:

  full       (cold restart from 1/N)               - ground truth
  warm       (warm-start from previous scores)     - cheaper, same answer
  localised  (only update dirty set + k-hop)       - cheapest, approximate

Plus a single full-recomputation on v0 to produce the initial scores
that the warm/localised runs need.

All runs use w=4 range partitioning. Speedup vs the full-recomputation
of the same snapshot is the headline metric; accuracy is reported as
max_abs_diff against the full result.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ray

from common import (
    GROUPC_V0, GROUPC_V1, GROUPC_V2,
    GROUPC_DELTA_01, GROUPC_DELTA_12,
    OUTPUT_DIR, build_m3_config, configure_logging, setup_paths,
)

setup_paths()
from incremental_pagerank import (  # noqa: E402
    run_full_recomputation, run_warm_start, run_localised_update,
    compare_to_reference,
)

log = configure_logging()
GC_OUT = OUTPUT_DIR / "m3_groupC"
GC_OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 4


def _load_delta(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    for p in (GROUPC_V0, GROUPC_V1, GROUPC_V2, GROUPC_DELTA_01, GROUPC_DELTA_12):
        if not p.exists():
            raise SystemExit(f"Missing {p}; run generate_graphs.py first.")

    delta_01 = _load_delta(GROUPC_DELTA_01)
    delta_12 = _load_delta(GROUPC_DELTA_12)

    summary = []

    # Launch Ray ONCE for all runs - matches AGENTS.md guidance (avoids GCS overload).
    ray.init(num_cpus=WORKERS, ignore_reinit_error=True,
             logging_level=logging.WARNING, include_dashboard=False)
    try:
        # ---- v0: baseline full run, produces prev_scores for v1 warm/localised ----
        log.info("=== Group C / v0 baseline full ===")
        cfg = build_m3_config(num_workers=WORKERS, partition_strategy="range")
        scores_v0, stats_v0 = run_full_recomputation(
            str(GROUPC_V0), cfg, output_dir=str(GC_OUT / "v0_full"),
            run_name="v0_full",
        )
        summary.append({
            "version": "v0", "strategy": "full", "elapsed_s": stats_v0.total_wall_time_s,
            "iterations": stats_v0.extras.get("num_iterations", -1),
            "dirty_pct": None, "max_abs_diff": 0.0, "speedup_vs_full": 1.0,
        })

        # ---- v1: full / warm / localised ----
        prev_scores = scores_v0
        for version, snap, delta in (
            ("v1", GROUPC_V1, delta_01),
            ("v2", GROUPC_V2, delta_12),
        ):
            log.info("=== Group C / %s full (ground truth) ===", version)
            scores_full, stats_full = run_full_recomputation(
                str(snap), cfg, output_dir=str(GC_OUT / f"{version}_full"),
                run_name=f"{version}_full",
            )
            t_full = stats_full.total_wall_time_s
            summary.append({
                "version": version, "strategy": "full", "elapsed_s": t_full,
                "iterations": stats_full.extras.get("num_iterations", -1),
                "dirty_pct": None, "max_abs_diff": 0.0, "speedup_vs_full": 1.0,
            })

            log.info("=== Group C / %s warm-start ===", version)
            scores_warm, stats_warm = run_warm_start(
                str(snap), prev_scores, cfg,
                output_dir=str(GC_OUT / f"{version}_warm"),
                run_name=f"{version}_warm",
            )
            t_warm = stats_warm.total_wall_time_s
            acc_warm = compare_to_reference(scores_warm, scores_full)
            summary.append({
                "version": version, "strategy": "warm", "elapsed_s": t_warm,
                "iterations": stats_warm.extras.get("num_iterations", -1),
                "dirty_pct": None, "max_abs_diff": acc_warm["max_abs_diff"],
                "speedup_vs_full": t_full / t_warm if t_warm > 0 else 0.0,
            })

            log.info("=== Group C / %s localised ===", version)
            scores_loc, stats_loc = run_localised_update(
                str(snap), prev_scores, delta, cfg,
                output_dir=str(GC_OUT / f"{version}_localised"),
                run_name=f"{version}_localised",
            )
            t_loc = stats_loc.total_wall_time_s
            acc_loc = compare_to_reference(scores_loc, scores_full)
            summary.append({
                "version": version, "strategy": "localised", "elapsed_s": t_loc,
                "iterations": stats_loc.extras.get("num_iterations", -1),
                "dirty_pct": stats_loc.run_metadata.get("dirty_pct"),
                "max_abs_diff": acc_loc["max_abs_diff"],
                "speedup_vs_full": t_full / t_loc if t_loc > 0 else 0.0,
            })

            # Chain prev_scores: the next step warm/loc starts from the most
            # accurate result of this step (the full run).
            prev_scores = scores_full
    finally:
        ray.shutdown()

    with open(GC_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 70)
    for r in summary:
        log.info("  %-3s %-10s elapsed=%.1fs speedup=%.2fx max_diff=%.2e",
                 r["version"], r["strategy"], r["elapsed_s"],
                 r["speedup_vs_full"], r["max_abs_diff"])


if __name__ == "__main__":
    main()
