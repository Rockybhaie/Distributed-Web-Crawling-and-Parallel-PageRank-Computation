
# evaluation_runner.py: Orchestrates the full Milestone 3 evaluation matrix.
#
# Three experimental groups, matching the spec exactly:
#
#   GROUP A  Sequential vs Parallel PageRank (on snapshot v0).
#       runs:
#           sequential                      (1 thread)
#           parallel_full_range w=1, 2, 4   (centralised, range partitioning)
#       metrics: total time, speedup vs sequential, efficiency, Amdahl
#
#   GROUP B  Different partitioning strategies (on snapshot v0, w=4).
#       runs:
#           parallel_full_range
#           parallel_full_hash
#           parallel_full_edge_balanced
#       metrics: load imbalance, communication volume, total time, memory
#
#   GROUP C  Static vs incremental computation.
#       For each transition v(K-1) -> vK:
#           full        on graph_vK     (this is the GROUND TRUTH)
#           warm        on graph_vK with prev_scores from full(v(K-1))
#           localised   on graph_vK with prev_scores + delta
#       metrics: time, iterations, max |scores - full|, dirty fraction
#
# All runs dump metrics to:
#   output/<group>/<run_name>/metrics.json
#   output/<group>/<run_name>/scores_full.json
# and a top-level output/evaluation_index.json keeps a flat index of every run.


import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

from m3_config import M3Config
from incremental_crawler import ensure_v0_snapshot
from incremental_pagerank import (
    run_sequential,
    run_full_recomputation,
    run_warm_start,
    run_localised_update,
    compare_to_reference,
)
from partitioning_strategies import (
    STRATEGY_RANGE, STRATEGY_HASH, STRATEGY_EDGE_BALANCED,
)

logger = logging.getLogger(__name__)



# Helpers

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _make_cfg(
    base: M3Config, *,
    workers: Optional[int] = None,
    partition_strategy: Optional[str] = None,
    update_strategy: Optional[str] = None,
) -> M3Config:
    cfg = deepcopy(base)
    if workers is not None:
        cfg.num_workers = workers
        cfg.num_partitions = workers
    if partition_strategy is not None:
        cfg.partition_strategy = partition_strategy
    if update_strategy is not None:
        cfg.update_strategy = update_strategy
    return cfg


def _ray_init(workers: int) -> None:
    ray.init(num_cpus=workers, ignore_reinit_error=True,
             logging_level=logging.WARNING)


def _ray_shutdown() -> None:
    if ray.is_initialized():
        ray.shutdown()


def _record_run(
    index: Dict[str, Any],
    group: str,
    run_name: str,
    run_dir: str,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    entry = {
        "group": group,
        "run_name": run_name,
        "run_dir": run_dir,
        "summary": {
            "total_wall_time_s": metrics.get("total_wall_time_s"),
            "num_iterations": metrics.get("num_iterations"),
            "avg_load_imbalance": metrics.get("avg_load_imbalance"),
            "total_comm_kb": metrics.get("total_comm_kb"),
            "peak_worker_rss_mb": metrics.get("peak_worker_rss_mb"),
            "driver_rss_peak_mb": metrics.get("driver_rss_peak_mb"),
            "final_delta": metrics.get("final_delta"),
            "metadata": metrics.get("metadata", {}),
            **(extras or {}),
        },
    }
    index.setdefault("runs", []).append(entry)



# GROUP A: sequential vs parallel

def run_group_a(cfg: M3Config, snapshot_path: str, quick: bool = False) -> List[Dict[str, Any]]:
    logger.info("=" * 70)
    logger.info("GROUP A: Sequential vs Parallel PageRank | snapshot=%s",
                snapshot_path)
    logger.info("=" * 70)
    group_dir = _ensure_dir(os.path.join(cfg.output_dir, "groupA"))
    runs: List[Dict[str, Any]] = []

    # Sequential (no Ray).
    logger.info("[A] Sequential ...")
    seq_cfg = _make_cfg(cfg, workers=1, update_strategy="sequential")
    seq_dir = _ensure_dir(os.path.join(group_dir, "sequential"))
    _, seq_stats = run_sequential(snapshot_path, seq_cfg, output_dir=seq_dir,
                                  run_name="sequential")
    seq_time = seq_stats.total_wall_time_s
    runs.append({"name": "sequential", "dir": seq_dir,
                 "elapsed_s": seq_time, "workers": 1,
                 "speedup": 1.0, "efficiency": 1.0})

    # Parallel centralised at varying worker counts.
    worker_counts = [1, 2, 4] if not quick else [1, 4]
    for w in worker_counts:
        run_name = f"parallel_full_range_w{w}"
        logger.info("[A] %s ...", run_name)
        run_dir = _ensure_dir(os.path.join(group_dir, run_name))
        par_cfg = _make_cfg(cfg, workers=w,
                            partition_strategy=STRATEGY_RANGE,
                            update_strategy="full")
        _ray_init(w)
        try:
            _, p_stats = run_full_recomputation(
                snapshot_path, par_cfg, output_dir=run_dir, run_name=run_name,
            )
        finally:
            _ray_shutdown()
        elapsed = p_stats.total_wall_time_s
        speedup = (seq_time / elapsed) if elapsed > 0 else 0.0
        runs.append({"name": run_name, "dir": run_dir,
                     "elapsed_s": elapsed, "workers": w,
                     "speedup": speedup, "efficiency": speedup / w})

    # Cross-run comparison summary
    summary_path = os.path.join(group_dir, "groupA_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    logger.info("Group A summary -> %s", summary_path)
    return runs



# GROUP B: partitioning strategy comparison

def run_group_b(cfg: M3Config, snapshot_path: str, quick: bool = False) -> List[Dict[str, Any]]:
    logger.info("=" * 70)
    logger.info("GROUP B: Partitioning Strategy Comparison | snapshot=%s",
                snapshot_path)
    logger.info("=" * 70)
    group_dir = _ensure_dir(os.path.join(cfg.output_dir, "groupB"))
    runs: List[Dict[str, Any]] = []

    strategies = [STRATEGY_RANGE, STRATEGY_HASH, STRATEGY_EDGE_BALANCED]
    if quick:
        strategies = [STRATEGY_RANGE, STRATEGY_EDGE_BALANCED]

    workers = cfg.num_workers
    for strat in strategies:
        run_name = f"parallel_full_{strat}_w{workers}"
        logger.info("[B] %s ...", run_name)
        run_dir = _ensure_dir(os.path.join(group_dir, run_name))
        b_cfg = _make_cfg(cfg, workers=workers,
                          partition_strategy=strat, update_strategy="full")
        _ray_init(workers)
        try:
            _, stats = run_full_recomputation(
                snapshot_path, b_cfg, output_dir=run_dir, run_name=run_name,
            )
        finally:
            _ray_shutdown()
        runs.append({
            "name": run_name, "dir": run_dir,
            "partition_strategy": strat,
            "elapsed_s": stats.total_wall_time_s,
            "avg_load_imbalance": stats.avg_load_imbalance,
            "total_comm_kb": stats.total_comm_bytes / 1024.0,
            "peak_worker_rss_mb": stats.peak_worker_rss_mb,
        })

    summary_path = os.path.join(group_dir, "groupB_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    logger.info("Group B summary -> %s", summary_path)
    return runs



# GROUP C: static vs incremental (the headline M3 axis)

def run_group_c(cfg: M3Config, quick: bool = False) -> List[Dict[str, Any]]:
    logger.info("=" * 70)
    logger.info("GROUP C: Static vs Incremental Computation")
    logger.info("=" * 70)
    group_dir = _ensure_dir(os.path.join(cfg.output_dir, "groupC"))
    runs: List[Dict[str, Any]] = []

    # Determine which transitions to test based on which snapshots exist.
    transitions: List[int] = []
    for k in range(1, len(cfg.growth_plan) + 1):
        if os.path.exists(os.path.join(cfg.snapshot_dir, f"graph_v{k}.pkl")):
            transitions.append(k)
    if quick:
        transitions = transitions[:1]
    if not transitions:
        logger.error(
            "No incremental snapshots found in %s. Run "
            "`python m3_main.py crawl-incremental --all` first.",
            cfg.snapshot_dir,
        )
        return runs

    logger.info("Will evaluate transitions: %s", transitions)

    # First, ensure we have the v0 reference scores.
    v0_path = os.path.join(cfg.snapshot_dir, "graph_v0.pkl")
    v0_scores_path = os.path.join(group_dir, "v0_full",
                                  "scores_full.json")
    if not os.path.exists(v0_scores_path):
        logger.info("[C] Computing v0 reference scores ...")
        v0_dir = _ensure_dir(os.path.join(group_dir, "v0_full"))
        v0_cfg = _make_cfg(cfg, partition_strategy=STRATEGY_RANGE,
                           update_strategy="full")
        _ray_init(v0_cfg.num_workers)
        try:
            run_full_recomputation(
                v0_path, v0_cfg, output_dir=v0_dir, run_name="v0_full",
            )
        finally:
            _ray_shutdown()

    with open(v0_scores_path, "r", encoding="utf-8") as f:
        prev_scores: Dict[str, float] = json.load(f)

    # For each subsequent snapshot, run all three update strategies.
    for k in transitions:
        snap_k = os.path.join(cfg.snapshot_dir, f"graph_v{k}.pkl")
        delta_path = os.path.join(cfg.snapshot_dir, f"delta_v{k-1}_v{k}.json")
        if not os.path.exists(delta_path):
            logger.warning("Missing delta file %s - skipping v%d", delta_path, k)
            continue
        with open(delta_path, "r", encoding="utf-8") as f:
            delta = json.load(f)

        # 1. FULL on v{k} - this is the ground truth for accuracy comparison.
        full_dir = _ensure_dir(os.path.join(group_dir, f"v{k}_full"))
        logger.info("[C] v%d full (ground truth) ...", k)
        full_cfg = _make_cfg(cfg, partition_strategy=STRATEGY_RANGE,
                             update_strategy="full")
        _ray_init(full_cfg.num_workers)
        try:
            full_scores, full_stats = run_full_recomputation(
                snap_k, full_cfg, output_dir=full_dir, run_name=f"v{k}_full",
            )
        finally:
            _ray_shutdown()

        # 2. WARM-START on v{k}
        warm_dir = _ensure_dir(os.path.join(group_dir, f"v{k}_warm"))
        logger.info("[C] v%d warm-start ...", k)
        warm_cfg = _make_cfg(cfg, partition_strategy=STRATEGY_RANGE,
                             update_strategy="warm")
        _ray_init(warm_cfg.num_workers)
        try:
            warm_scores, warm_stats = run_warm_start(
                snap_k, prev_scores, warm_cfg, output_dir=warm_dir,
                run_name=f"v{k}_warm",
            )
        finally:
            _ray_shutdown()
        warm_acc = compare_to_reference(warm_scores, full_scores)
        warm_stats_extras = {**warm_acc}

        # 3. LOCALISED on v{k}
        loc_dir = _ensure_dir(os.path.join(group_dir, f"v{k}_localised"))
        logger.info("[C] v%d localised ...", k)
        loc_cfg = _make_cfg(cfg, partition_strategy=STRATEGY_RANGE,
                            update_strategy="localised")
        _ray_init(loc_cfg.num_workers)
        try:
            loc_scores, loc_stats = run_localised_update(
                snap_k, prev_scores, delta, loc_cfg, output_dir=loc_dir,
                run_name=f"v{k}_localised",
            )
        finally:
            _ray_shutdown()
        loc_acc = compare_to_reference(loc_scores, full_scores)
        loc_stats_extras = {**loc_acc}

        # Append accuracy info into each run's metrics file for plotting.
        for run_dir, extras in (
            (warm_dir, warm_stats_extras),
            (loc_dir, loc_stats_extras),
        ):
            mp = os.path.join(run_dir, "metrics.json")
            with open(mp, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["accuracy_vs_full"] = extras
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, default=str)

        runs.append({
            "transition": f"v{k-1}->v{k}",
            "full":       {"dir": full_dir,
                           "elapsed_s": full_stats.total_wall_time_s,
                           "iterations": full_stats.num_iterations},
            "warm":       {"dir": warm_dir,
                           "elapsed_s": warm_stats.total_wall_time_s,
                           "iterations": warm_stats.num_iterations,
                           "accuracy": warm_acc},
            "localised":  {"dir": loc_dir,
                           "elapsed_s": loc_stats.total_wall_time_s,
                           "iterations": loc_stats.num_iterations,
                           "dirty_node_count": loc_stats.run_metadata.get("dirty_node_count"),
                           "accuracy": loc_acc},
            "delta_summary": delta.get("summary", {}),
        })

        # Use full's scores as the next round's prev_scores (chained reference).
        prev_scores = full_scores

    summary_path = os.path.join(group_dir, "groupC_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, default=str)
    logger.info("Group C summary -> %s", summary_path)
    return runs



# Public entry point

def run_evaluation(cfg: M3Config, group: str = "all", quick: bool = False) -> Dict[str, Any]:
    _ensure_dir(cfg.output_dir)

    # Make sure v0 exists (Group A and B run on it).
    ensure_v0_snapshot(cfg)
    v0_path = os.path.join(cfg.snapshot_dir, "graph_v0.pkl")

    out: Dict[str, Any] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "config": cfg.__dict__, "groups": {}}

    if group in ("A", "all"):
        out["groups"]["A"] = run_group_a(cfg, v0_path, quick=quick)
    if group in ("B", "all"):
        out["groups"]["B"] = run_group_b(cfg, v0_path, quick=quick)
    if group in ("C", "all"):
        out["groups"]["C"] = run_group_c(cfg, quick=quick)

    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    index_path = os.path.join(cfg.output_dir, "evaluation_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Evaluation index -> %s", index_path)
    return out
