"""
run_m2.py
=========

Re-runs Milestone 2's parallel PageRank (centralised + distributed
strategies) on the 2M-node synthetic graph. We deliberately bypass the
NetworkX validation phase that main_pagerank.py runs by default, because
on 2M nodes NetworkX would take ~30 min and is irrelevant to the
parallel-speedup story.

Experiments (5 runs):
  1. Sequential baseline                 (custom_pagerank from M2)
  2. Centralised aggregation, w=4
  3. Centralised aggregation, w=8
  4. Distributed reduction (RankActor), w=4
  5. Distributed reduction (RankActor), w=8

Results are written to output/m2/<run_name>/metrics.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import ray

from common import (
    BIG_GRAPH_PATH, OUTPUT_DIR, build_pagerank_config,
    configure_logging, setup_paths,
)

setup_paths()
from graph_loader import load_graph, partition_nodes  # noqa: E402
from pagerank_sequential import custom_pagerank  # noqa: E402
from pagerank_parallel import (  # noqa: E402
    run_pagerank_centralized, run_pagerank_distributed,
)

log = configure_logging()
M2_OUT = OUTPUT_DIR / "m2"
M2_OUT.mkdir(parents=True, exist_ok=True)


def _save_run(name: str, payload: dict) -> None:
    d = M2_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s/metrics.json", d)


def main() -> None:
    if not BIG_GRAPH_PATH.exists():
        raise SystemExit(f"Missing {BIG_GRAPH_PATH}; run generate_graphs.py first.")

    # Load once - reused for all runs.
    config_for_load = build_pagerank_config(
        graph_path=str(BIG_GRAPH_PATH), num_workers=8, output_dir=str(M2_OUT),
    )
    log.info("Loading 2M-node graph...")
    t0 = time.perf_counter()
    gd = load_graph(config_for_load)
    log.info("Loaded in %.1fs | nodes=%d edges=%d dangling=%d",
             time.perf_counter() - t0,
             gd["num_nodes"], gd["num_edges"], len(gd["dangling"]))

    nodes, in_links, out_degree, dangling = (
        gd["nodes"], gd["in_links"], gd["out_degree"], gd["dangling"],
    )
    N, E = gd["num_nodes"], gd["num_edges"]
    summary = []

    # ------ 1. Sequential baseline ------
    log.info("=== M2 / 1: Sequential baseline ===")
    seq_cfg = build_pagerank_config(graph_path=str(BIG_GRAPH_PATH), num_workers=1)
    t_seq = time.perf_counter()
    _, seq_elapsed, seq_iters, seq_deltas = custom_pagerank(
        nodes, in_links, out_degree, dangling, seq_cfg,
    )
    log.info("Sequential: %.1fs in %d iterations", seq_elapsed, seq_iters)
    _save_run("sequential", {
        "strategy": "sequential", "workers": 1, "elapsed_s": seq_elapsed,
        "iterations": seq_iters, "num_nodes": N, "num_edges": E,
        "deltas": seq_deltas,
    })
    summary.append({"name": "sequential", "strategy": "sequential",
                    "workers": 1, "elapsed_s": seq_elapsed,
                    "iterations": seq_iters, "speedup": 1.0})

    # ------ Parallel runs: launch Ray once, vary workers ------
    for workers in (4, 8):
        ray.init(num_cpus=workers, ignore_reinit_error=True,
                 logging_level=logging.WARNING, include_dashboard=False)
        try:
            partitions = partition_nodes(N, workers)
            cfg = build_pagerank_config(
                graph_path=str(BIG_GRAPH_PATH), num_workers=workers,
            )
            for strategy_name, runner in (
                ("centralized", run_pagerank_centralized),
                ("distributed", run_pagerank_distributed),
            ):
                run_name = f"{strategy_name}_w{workers}"
                log.info("=== M2 / %s ===", run_name)
                _, elapsed, iters, metrics = runner(
                    nodes, in_links, out_degree, dangling, partitions, cfg,
                )
                log.info("%s: %.1fs in %d iters (speedup %.2fx)",
                         run_name, elapsed, iters, seq_elapsed / elapsed)
                _save_run(run_name, {
                    "strategy": strategy_name, "workers": workers,
                    "elapsed_s": elapsed, "iterations": iters,
                    "num_nodes": N, "num_edges": E,
                    "speedup": seq_elapsed / elapsed,
                    "load_imbalance": metrics.get("load_imbalance", 0.0),
                    "peak_memory_mb": metrics.get("peak_memory_mb", 0.0),
                    "comm_kb_total": metrics.get("comm_kb_total", 0.0),
                    "iter_times": metrics.get("iter_times", []),
                    "iter_deltas": metrics.get("iter_deltas", []),
                })
                summary.append({
                    "name": run_name, "strategy": strategy_name,
                    "workers": workers, "elapsed_s": elapsed,
                    "iterations": iters,
                    "speedup": seq_elapsed / elapsed,
                })
        finally:
            ray.shutdown()

    with open(M2_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    for r in summary:
        log.info("  %-20s w=%d elapsed=%.1fs speedup=%.2fx",
                 r["name"], r["workers"], r["elapsed_s"], r["speedup"])


if __name__ == "__main__":
    main()
