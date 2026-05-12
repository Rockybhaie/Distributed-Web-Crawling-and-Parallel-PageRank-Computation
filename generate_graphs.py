"""
generate_graphs.py
==================

Materialises every synthetic snapshot the study needs:

  snapshots/graph_big.pkl       2,000,000 nodes  (M2 + M3 Group A/B)
  snapshots/groupc_v0.pkl         800,000 nodes  (M3 Group C base)
  snapshots/groupc_v1.pkl         900,000 nodes  (M3 Group C +12.5%)
  snapshots/groupc_v2.pkl       1,000,000 nodes  (M3 Group C +11%)
  snapshots/groupc_delta_v0_v1.json   delta dict for incremental_pagerank
  snapshots/groupc_delta_v1_v2.json

Idempotent: skips any snapshot that already exists.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sibling module imports work when running with `python generate_graphs.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SNAPSHOT_DIR, BIG_GRAPH_PATH,
    GROUPC_V0, GROUPC_V1, GROUPC_V2,
    GROUPC_DELTA_01, GROUPC_DELTA_12,
    configure_logging,
)
from synthetic_generator import (
    GenConfig, grow_graph, save_snapshot, load_snapshot,
    compute_delta, save_delta,
)

log = configure_logging()


def _ensure_big_graph() -> None:
    if BIG_GRAPH_PATH.exists():
        log.info("graph_big.pkl already exists, skipping.")
        return
    cfg = GenConfig(target_nodes=2_000_000, edges_per_node=4,
                    num_communities=64, p_cross_community=0.10,
                    p_dangling=0.05, seed=20260512)
    log.info("Building big graph (2,000,000 nodes)... this takes ~10-15 min.")
    t0 = time.perf_counter()
    G, meta = grow_graph(None, cfg)
    log.info("Built in %.1fs, saving...", time.perf_counter() - t0)
    save_snapshot(G, BIG_GRAPH_PATH, meta=meta)


def _ensure_groupc_series() -> None:
    """v0 -> v1 -> v2 progression so we have deltas for incremental PR."""
    targets = [(GROUPC_V0, 800_000), (GROUPC_V1, 900_000), (GROUPC_V2, 1_000_000)]

    G = None
    for path, n in targets:
        if path.exists():
            log.info("%s already exists, loading to continue chain.", path.name)
            G = load_snapshot(path)
            continue
        cfg = GenConfig(target_nodes=n, edges_per_node=4,
                        num_communities=64, p_cross_community=0.10,
                        p_dangling=0.05, seed=20260512)
        log.info("Growing toward %d nodes -> %s", n, path.name)
        G, meta = grow_graph(G, cfg)
        save_snapshot(G, path, meta=meta)

    # Compute deltas.
    if not GROUPC_DELTA_01.exists():
        log.info("Computing delta v0 -> v1 ...")
        d = compute_delta(load_snapshot(GROUPC_V0), load_snapshot(GROUPC_V1))
        save_delta(d, GROUPC_DELTA_01)
    if not GROUPC_DELTA_12.exists():
        log.info("Computing delta v1 -> v2 ...")
        d = compute_delta(load_snapshot(GROUPC_V1), load_snapshot(GROUPC_V2))
        save_delta(d, GROUPC_DELTA_12)


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_groupc_series()  # smaller first - failure here is cheap
    _ensure_big_graph()
    log.info("All snapshots ready.")


if __name__ == "__main__":
    main()
