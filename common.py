"""
common.py
=========

Shared boilerplate for the synthetic-large-graph driver scripts:
  - sys.path / PYTHONPATH wiring so the unchanged Milestone 2 and
    Milestone 3 modules can be imported by both driver and Ray workers.
  - Factory helpers to build PageRankConfig (M2) and M3Config (M3) with
    the experiment-appropriate parameters.

We do NOT modify any code in Milestone 1, 2, or 3. We only IMPORT from them.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PROJECT = _ROOT.parent
_M1 = _PROJECT / "Milestone 1"
_M2 = _PROJECT / "Milestone 2"
_M3 = _PROJECT / "Milestone 3"


def setup_paths() -> None:
    """Insert M1/M2/M3 and the study folder onto sys.path AND PYTHONPATH."""
    for p in (_M1, _M2, _M3, _ROOT):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_M1), str(_M2), str(_M3), str(_ROOT)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("synthetic_study")


def build_m3_config(
    num_workers: int = 4,
    partition_strategy: str = "range",
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    affected_hop_radius: int = 2,
):
    """
    Build an M3Config for the synthetic study. We deliberately leave the
    crawl/snapshot/cache fields at their defaults because we never call
    the M3 incremental crawler - we provide our own snapshots.
    """
    setup_paths()
    from m3_config import M3Config  # type: ignore
    return M3Config(
        num_workers=num_workers,
        num_partitions=num_workers,
        partition_strategy=partition_strategy,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        affected_hop_radius=affected_hop_radius,
    )


def build_pagerank_config(
    graph_path: str,
    num_workers: int = 4,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    strategy: str = "centralized",
    output_dir: str = "output/m2_run",
):
    """Build an M2 PageRankConfig pointing at our synthetic snapshot."""
    setup_paths()
    from pagerank_config import PageRankConfig  # type: ignore
    return PageRankConfig(
        graph_path=graph_path,
        num_workers=num_workers,
        num_partitions=num_workers,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        strategy=strategy,
        output_dir=output_dir,
    )


# Convenience paths
SNAPSHOT_DIR = _ROOT / "snapshots"
OUTPUT_DIR = _ROOT / "output"
BIG_GRAPH_PATH = SNAPSHOT_DIR / "graph_big.pkl"   # 2M nodes - M2 + M3 A/B
GROUPC_V0 = SNAPSHOT_DIR / "groupc_v0.pkl"        # 800K nodes - M3 C base
GROUPC_V1 = SNAPSHOT_DIR / "groupc_v1.pkl"        # 900K nodes - M3 C +12.5%
GROUPC_V2 = SNAPSHOT_DIR / "groupc_v2.pkl"        # 1M nodes - M3 C +11%
GROUPC_DELTA_01 = SNAPSHOT_DIR / "groupc_delta_v0_v1.json"
GROUPC_DELTA_12 = SNAPSHOT_DIR / "groupc_delta_v1_v2.json"
