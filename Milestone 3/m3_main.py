
# m3_main.py: Single CLI entry point for Milestone 3.
#
# Subcommands:
#
#   crawl-incremental   Run one (or all) incremental crawl steps to produce
#                       graph_v1.pkl, graph_v2.pkl, ... from M1 Config 3.
#
#   pagerank            Run a single PageRank computation on a chosen
#                       snapshot, using a specified partitioning strategy and
#                       update strategy. Produces metrics under output/.
#
#   evaluate            Run the full evaluation matrix (Group A / B / C / all).
#                       Produces a tree of metrics.json files.
#
#   plot                Aggregate every metrics.json under output/ and produce
#                       all figures.
#
# Examples (run from inside Milestone 3/):
#   python m3_main.py crawl-incremental --all
#   python m3_main.py crawl-incremental --version 1
#   python m3_main.py pagerank --snapshot snapshots/graph_v1.pkl \
#                              --partition edge_balanced --update warm
#   python m3_main.py evaluate --group all
#   python m3_main.py plot


import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Optional

import ray

from m3_config import M3Config
from partitioning_strategies import VALID_STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("m3_main")



# Shared CLI argument helpers

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=4,
                        help="Ray worker count.")
    parser.add_argument("--partitions", type=int, default=None,
                        help="Partition count (defaults to --workers).")
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--no-convergence", action="store_true",
                        help="Run fixed --max-iter iterations instead.")
    parser.add_argument("--ray-address", default=None,
                        help="Connect to an existing Ray cluster.")
    parser.add_argument("--snapshot-dir", default="snapshots")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="output")


def _config_from_args(args: argparse.Namespace) -> M3Config:
    cfg = M3Config(
        snapshot_dir=getattr(args, "snapshot_dir", "snapshots"),
        cache_dir=getattr(args, "cache_dir", "cache"),
        damping_factor=getattr(args, "damping", 0.85),
        max_iterations=getattr(args, "max_iter", 100),
        convergence_threshold=getattr(args, "threshold", 1e-6),
        use_convergence=not getattr(args, "no_convergence", False),
        num_workers=getattr(args, "workers", 4),
        num_partitions=getattr(args, "partitions", None) or getattr(args, "workers", 4),
        partition_strategy=getattr(args, "partition", "range"),
        update_strategy=getattr(args, "update", "full"),
        execution_strategy="centralized",
        output_dir=getattr(args, "output_dir", "output"),
    )
    return cfg



# Subcommand: crawl-incremental

def cmd_crawl_incremental(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    cfg.crawl_index = args.crawl_index
    cfg.target_domain = args.domain
    cfg.restrict_to_domain = args.domain
    cfg.crawl_batch_size = args.batch_size

    from incremental_crawler import (
        ensure_v0_snapshot,
        run_full_growth_plan,
        run_incremental_step,
    )

    ensure_v0_snapshot(cfg)

    if args.all:
        results = run_full_growth_plan(cfg)
        out = {"steps": results}
    elif args.version is not None:
        if args.version <= 0:
            raise SystemExit("--version must be >= 1.")
        if args.version > len(cfg.growth_plan):
            raise SystemExit(
                f"--version {args.version} exceeds growth_plan length "
                f"({len(cfg.growth_plan)})."
            )
        ray.init(num_cpus=cfg.num_workers, ignore_reinit_error=True,
                 logging_level=logging.WARNING)
        from_v = args.version - 1
        to_v = args.version
        prev_path = os.path.join(cfg.snapshot_dir, f"graph_v{from_v}.pkl")
        cdx_slice = tuple(cfg.growth_plan[from_v])
        result = run_incremental_step(
            cfg=cfg,
            from_version=from_v,
            to_version=to_v,
            cdx_slice=cdx_slice,
            prev_snapshot_path=prev_path,
            ray_already_initialised=True,
        )
        ray.shutdown()
        out = {"step": result}
    else:
        raise SystemExit("Specify either --all or --version <N>.")

    summary_path = os.path.join(cfg.snapshot_dir, "crawl_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Crawl summary written to %s", summary_path)



# Subcommand: pagerank

def cmd_pagerank(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)

    from incremental_pagerank import (
        run_sequential, run_full_recomputation,
        run_warm_start, run_localised_update,
    )

    snapshot = args.snapshot
    if not os.path.exists(snapshot):
        raise SystemExit(f"Snapshot not found: {snapshot}")

    out_subdir = args.run_name or f"{cfg.update_strategy}_{cfg.partition_strategy}"
    out_dir = os.path.join(cfg.output_dir, out_subdir)

    prev_scores = None
    if args.prev_scores:
        with open(args.prev_scores, "r", encoding="utf-8") as f:
            payload = json.load(f)
        prev_scores = (payload if isinstance(payload, dict)
                       and all(isinstance(v, (int, float)) for v in payload.values())
                       else payload.get("scores", payload))

    delta = None
    if args.delta:
        with open(args.delta, "r", encoding="utf-8") as f:
            delta = json.load(f)

    # Sequential is the only path that does not need Ray.
    if cfg.update_strategy != "sequential":
        ray_kw = {"num_cpus": cfg.num_workers}
        if args.ray_address:
            ray_kw["address"] = args.ray_address
        ray.init(**ray_kw, ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    if cfg.update_strategy == "sequential":
        run_sequential(snapshot, cfg, output_dir=out_dir, run_name=out_subdir)
    elif cfg.update_strategy == "full":
        run_full_recomputation(snapshot, cfg, output_dir=out_dir,
                               run_name=out_subdir)
    elif cfg.update_strategy == "warm":
        if prev_scores is None:
            raise SystemExit("--prev-scores is required for --update warm")
        run_warm_start(snapshot, prev_scores, cfg, output_dir=out_dir,
                       run_name=out_subdir)
    elif cfg.update_strategy == "localised":
        if prev_scores is None or delta is None:
            raise SystemExit(
                "--prev-scores and --delta are both required for --update localised"
            )
        run_localised_update(snapshot, prev_scores, delta, cfg,
                             output_dir=out_dir, run_name=out_subdir)
    else:
        raise SystemExit(f"Unknown update strategy: {cfg.update_strategy}")

    if cfg.update_strategy != "sequential":
        ray.shutdown()



# Subcommand: evaluate

def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    from evaluation_runner import run_evaluation
    run_evaluation(cfg, group=args.group, quick=args.quick)



# Subcommand: plot

def cmd_plot(args: argparse.Namespace) -> None:
    from plot_results import generate_all_figures
    generate_all_figures(input_dir=args.input_dir, output_dir=args.figures_dir)



# Argument parser construction

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Milestone 3 - Incremental Updates and System Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # crawl-incremental
    pc = sub.add_parser("crawl-incremental",
                        help="Grow the graph by crawling a new CDX slice.")
    _add_common_args(pc)
    pc.add_argument("--all", action="store_true",
                    help="Run all steps in the growth plan.")
    pc.add_argument("--version", type=int, default=None,
                    help="Run only the step that produces graph_v<N>.")
    pc.add_argument("--domain", default="en.wikipedia.org")
    pc.add_argument("--crawl-index", default="CC-MAIN-2024-10")
    pc.add_argument("--batch-size", type=int, default=25)
    pc.set_defaults(func=cmd_crawl_incremental)

    # pagerank
    pp = sub.add_parser("pagerank", help="Run PageRank on a snapshot.")
    _add_common_args(pp)
    pp.add_argument("--snapshot", required=True,
                    help="Path to graph_vN.pkl (or any DiGraph pickle).")
    pp.add_argument("--update", choices=["sequential", "full", "warm", "localised"],
                    default="full")
    pp.add_argument("--partition", choices=list(VALID_STRATEGIES), default="range")
    pp.add_argument("--prev-scores", default=None,
                    help="Path to previous run's scores_full.json (warm/localised).")
    pp.add_argument("--delta", default=None,
                    help="Path to delta_v(K-1)_vK.json (localised only).")
    pp.add_argument("--run-name", default=None,
                    help="Custom subfolder name under --output-dir.")
    pp.set_defaults(func=cmd_pagerank)

    # evaluate
    pe = sub.add_parser("evaluate", help="Run the structured evaluation matrix.")
    _add_common_args(pe)
    pe.add_argument("--group", choices=["A", "B", "C", "all"], default="all",
                    help="A=seq vs parallel, B=partitioning, C=static vs incremental.")
    pe.add_argument("--quick", action="store_true",
                    help="Run a reduced sub-matrix (faster smoke test).")
    pe.set_defaults(func=cmd_evaluate)

    # plot
    pl = sub.add_parser("plot", help="Generate figures from saved metrics.")
    pl.add_argument("--input-dir", default="output",
                    help="Directory containing metrics.json files.")
    pl.add_argument("--figures-dir", default="output/figures",
                    help="Where to write generated figures.")
    pl.set_defaults(func=cmd_plot)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
