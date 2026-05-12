# main_pagerank.py — Entry point for Milestone 2: Parallel PageRank Computation.

"""
Pipeline
  Phase 1  →  Load graph from Milestone 1 pickle file
  Phase 2a →  Custom sequential PageRank (baseline for speedup)
  Phase 2b →  NetworkX PageRank (correctness reference)
  Phase 3  →  Parallel PageRank — Centralised Aggregation
  Phase 4  →  Parallel PageRank — Distributed Reduction
  Phase 5  →  Validate correctness (parallel ≈ sequential)
  Phase 6  →  Full performance report with all metrics

"""

import argparse
import dataclasses
import logging
import sys

import ray

from pagerank_config import PageRankConfig
from graph_loader import load_graph, partition_nodes
from pagerank_sequential import networkx_pagerank, custom_pagerank, validate_scores
from pagerank_parallel import run_pagerank_centralized, run_pagerank_distributed
from performance_analysis import (
    PageRankResults,
    print_results,
    print_full_comparison,
    print_convergence_table,
    save_results,
    amdahl_speedup,
    estimate_parallel_fraction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main_pagerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2: Parallel PageRank Computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph",          default="output_config3/web_graph.pkl")
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--damping",        type=float, default=0.85)
    parser.add_argument("--max-iter",       type=int,   default=100)
    parser.add_argument("--threshold",      type=float, default=1e-6)
    parser.add_argument("--no-convergence", action="store_true")
    parser.add_argument("--strategy",       choices=["both","centralized","distributed"], default="both")
    parser.add_argument("--output-dir",     default="pagerank_output")
    parser.add_argument("--ray-address",    default=None)
    parser.add_argument("--skip-sequential",action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PageRankConfig(
        graph_path            = args.graph,
        num_workers           = args.workers,
        num_partitions        = args.workers,
        damping_factor        = args.damping,
        max_iterations        = args.max_iter,
        convergence_threshold = args.threshold,
        use_convergence       = not args.no_convergence,
        strategy              = args.strategy,
        output_dir            = args.output_dir,
    )

    logger.info("=" * 68)
    logger.info("Milestone 2 — Parallel PageRank Computation")
    logger.info("=" * 68)
    logger.info("Config: %s", dataclasses.asdict(config))


    # Phase 1: Load graph                                                 
    logger.info("Phase 1: Loading graph ...")
    gd         = load_graph(config)
    nodes      = gd["nodes"]
    in_links   = gd["in_links"]
    out_degree = gd["out_degree"]
    dangling   = gd["dangling"]
    G          = gd["graph"]
    N          = gd["num_nodes"]
    E          = gd["num_edges"]

    logger.info("Graph ready | nodes=%d  edges=%d  dangling=%d (%.1f%%)",
                N, E, len(dangling), 100*len(dangling)/N)

    partitions = partition_nodes(N, config.num_partitions)
    logger.info("Partitioned into %d chunks | avg size=%d nodes",
                len(partitions), N // max(len(partitions), 1))

    
    # Phase 2: Sequential baselines                                       
    seq_result = None
    seq_time   = None
    seq_scores = None

    if not args.skip_sequential:
        logger.info("Phase 2a: Custom sequential PageRank ...")
        seq_scores, seq_elapsed, seq_iters, seq_deltas = custom_pagerank(
            nodes, in_links, out_degree, dangling, config
        )
        seq_result = PageRankResults(
            strategy="Sequential (Custom)", num_workers=1,
            num_nodes=N, num_edges=E, scores=seq_scores,
            elapsed=seq_elapsed, num_iterations=seq_iters,
            damping_factor=config.damping_factor,
            convergence_threshold=config.convergence_threshold,
            metrics={}, iter_deltas=seq_deltas,
        )
        seq_time = seq_elapsed
        print_results(seq_result)

        logger.info("Phase 2b: NetworkX PageRank (reference) ...")
        nx_scores, nx_elapsed, _ = networkx_pagerank(G, config)
        logger.info("NetworkX done in %.3fs", nx_elapsed)

        logger.info("Validating custom sequential against NetworkX ...")
        if validate_scores(nx_scores, seq_scores, tolerance=1e-4):
            logger.info("✓ Custom sequential matches NetworkX — algorithm is correct.")
        else:
            logger.warning("✗ Custom sequential diverges from NetworkX!")

    
    # Phase 3 & 4: Parallel strategies                                   
    logger.info("Initialising Ray ...")
    ray_kw = {"num_cpus": config.num_workers}
    if args.ray_address:
        ray_kw["address"] = args.ray_address
    ray.init(**ray_kw, ignore_reinit_error=True)

    par_c = None
    par_d = None

    if args.strategy in ("both", "centralized"):
        logger.info("Phase 3: Parallel PageRank — Centralised Aggregation ...")
        sc, tc, ic, mc = run_pagerank_centralized(
            nodes, in_links, out_degree, dangling, partitions, config
        )
        par_c = PageRankResults(
            strategy="Parallel Centralised", num_workers=config.num_workers,
            num_nodes=N, num_edges=E, scores=sc,
            elapsed=tc, num_iterations=ic,
            damping_factor=config.damping_factor,
            convergence_threshold=config.convergence_threshold,
            metrics=mc, iter_deltas=mc.get("iter_deltas", []),
        )
        print_results(par_c, seq_time)
        if seq_scores:
            validate_scores(seq_scores, sc, tolerance=1e-4)
        save_results(par_c, config.output_dir)

    if args.strategy in ("both", "distributed"):
        logger.info("Phase 4: Parallel PageRank — Distributed Reduction ...")
        sd, td, id_, md = run_pagerank_distributed(
            nodes, in_links, out_degree, dangling, partitions, config
        )
        par_d = PageRankResults(
            strategy="Parallel Distributed", num_workers=config.num_workers,
            num_nodes=N, num_edges=E, scores=sd,
            elapsed=td, num_iterations=id_,
            damping_factor=config.damping_factor,
            convergence_threshold=config.convergence_threshold,
            metrics=md, iter_deltas=md.get("iter_deltas", []),
        )
        print_results(par_d, seq_time)
        if seq_scores:
            validate_scores(seq_scores, sd, tolerance=1e-4)
        save_results(par_d, config.output_dir)

   
    # Phase 6: Full comparison                                           
    if seq_result and par_c and par_d:
        print_full_comparison(seq_result, par_c, par_d)
        print_convergence_table(seq_result, par_c, par_d)

        # Amdahl summary
        print("  AMDAHL SCALABILITY PROJECTION")
        print("  " + "-" * 50)
        p = estimate_parallel_fraction(seq_time, par_c.elapsed, config.num_workers)
        print(f"  Estimated parallel fraction: {p:.4f} ({p*100:.1f}%)")
        print(f"  {'Workers':>8}  {'Theoretical Speedup':>20}")
        for n in [1, 2, 4, 8, 16, 32]:
            th = amdahl_speedup(p, n)
            print(f"  {n:>8}  {th:>20.4f}x")
        print()

    ray.shutdown()
    logger.info("Ray shut down. Milestone 2 complete.")


if __name__ == "__main__":
    main()