"""
synthetic_generator.py
======================

Generates a directed, clustered, scale-free graph that mimics the structural
properties of the CommonCrawl web graph used by Milestones 1-3, but at a
size large enough that parallel PageRank actually shows speedup over the
sequential baseline.

Why this generator?
-------------------
The real Milestone 1 graph has 37K nodes / 67K edges. At that scale,
Ray's serialisation + barrier costs dominate the work-per-iteration, so
parallel < sequential (we observed ~0.30x). To expose the parallel regime
we need: (a) ~1M-2M nodes, (b) realistic in-degree power-law (a few hub
pages, long tail), and (c) some clustering (real web pages link more
within their own community than across).

Model used: per-community Barabasi-Albert preferential-attachment growth.
We split the nodes into K communities. When a new node is added it picks
its community uniformly at random, then attaches `m` out-edges via
preferential attachment. With probability `p_cross` (default 0.1) an edge
crosses into a different random community; otherwise it stays within the
community. This produces:
  - in-degree distribution that follows a power law (BA hallmark),
  - clustering coefficient > 0 (community structure),
  - directed edges with non-trivial dangling node fraction.

The generator works incrementally: you can grow an existing snapshot from
N to N' by passing it in via `grow_graph()`. We use this both for the
final 2M graph (built in one call) and for the M3 Group C v0/v1/v2 series
which needs deltas between snapshots.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GenConfig:
    """Parameters for synthetic graph generation / growth."""
    target_nodes: int = 2_000_000     # final node count
    edges_per_node: int = 4           # m in Barabasi-Albert (out-degree on creation)
    num_communities: int = 64         # number of clusters
    p_cross_community: float = 0.10   # probability an edge crosses community
    p_dangling: float = 0.05          # probability a new node is dangling (no out-edges)
    seed: int = 20260512


def _community_of(node_id: int, num_communities: int) -> int:
    """Deterministic community assignment by hashing the node id."""
    # Simple but well-mixing: multiplicative hash mod K.
    return (node_id * 2654435761) % num_communities


def generate_graph(
    cfg: GenConfig,
    log_every: int = 100_000,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Build a fresh synthetic graph of `cfg.target_nodes` nodes.

    Returns:
        G : nx.DiGraph with integer-indexed node URLs of the form
            "syn://node/{i}" so the M2/M3 pipelines (which key on URL
            strings) work unchanged.
        meta : dict with generation metadata.
    """
    return grow_graph(None, cfg, log_every=log_every)


def grow_graph(
    G: Optional[nx.DiGraph],
    cfg: GenConfig,
    log_every: int = 100_000,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Grow an existing DiGraph up to `cfg.target_nodes` nodes (or generate
    a fresh one if G is None). Uses Barabasi-Albert preferential
    attachment with community structure.

    Edges are sampled by drawing each end-point from a list of
    "attachment slots" -- a multiset where node id i appears once per
    in-link it has plus one (so brand-new nodes get a starting weight
    of 1, preventing zero-probability sinks). This is exactly the
    standard BA construction; we just bias toward intra-community picks.
    """
    rng = random.Random(cfg.seed)
    K = cfg.num_communities
    m = cfg.edges_per_node

    if G is None:
        G = nx.DiGraph()
        seed_nodes = max(m + 1, 2 * K)
        # Seed clique-ish graph so preferential attachment has something to bite into.
        for i in range(seed_nodes):
            G.add_node(_url_of(i))
        # Sprinkle a small number of edges so each community is connected.
        for i in range(seed_nodes):
            for _ in range(m):
                j = rng.randrange(seed_nodes)
                if j != i:
                    G.add_edge(_url_of(i), _url_of(j))
        start_id = seed_nodes
    else:
        start_id = G.number_of_nodes()

    target = cfg.target_nodes
    if start_id >= target:
        logger.info("Graph already at %d >= target %d, no growth needed.",
                    start_id, target)
        meta = _meta(G, cfg, grown_from=start_id)
        return G, meta

    # Build attachment-slot lists per community. slots[c] is a Python
    # list where every appearance of a node id biases its selection
    # probability proportional to (1 + in_degree). We populate it from
    # the current graph (O(E)) then append on the fly during growth.
    slots: List[List[int]] = [[] for _ in range(K)]
    for url in G.nodes():
        nid = _id_of(url)
        c = _community_of(nid, K)
        slots[c].append(nid)  # base weight of 1
    for src, dst in G.edges():
        dst_id = _id_of(dst)
        c_dst = _community_of(dst_id, K)
        slots[c_dst].append(dst_id)  # in-link gives one extra slot

    t0 = time.perf_counter()
    last_log = t0
    for new_id in range(start_id, target):
        new_url = _url_of(new_id)
        G.add_node(new_url)
        c_new = _community_of(new_id, K)
        slots[c_new].append(new_id)  # base weight

        # Decide if this node is dangling (no out-edges).
        if rng.random() < cfg.p_dangling:
            pass  # no out-edges
        else:
            # Sample m unique targets via preferential attachment.
            picked: Set[int] = set()
            attempts = 0
            while len(picked) < m and attempts < m * 6:
                attempts += 1
                # Pick community: usually own, sometimes cross.
                if rng.random() < cfg.p_cross_community:
                    c = rng.randrange(K)
                else:
                    c = c_new
                bucket = slots[c]
                if not bucket:
                    c = c_new
                    bucket = slots[c_new]
                tgt = bucket[rng.randrange(len(bucket))]
                if tgt == new_id or tgt in picked:
                    continue
                picked.add(tgt)
            for tgt in picked:
                G.add_edge(new_url, _url_of(tgt))
                c_tgt = _community_of(tgt, K)
                slots[c_tgt].append(tgt)  # in-link weight bump

        if (new_id + 1) % log_every == 0:
            now = time.perf_counter()
            rate = log_every / max(now - last_log, 1e-9)
            last_log = now
            logger.info("  generated %d / %d nodes (%.0f nodes/sec, edges so far=%d)",
                        new_id + 1, target, rate, G.number_of_edges())

    elapsed = time.perf_counter() - t0
    logger.info("Generation complete: %d nodes, %d edges in %.1fs",
                G.number_of_nodes(), G.number_of_edges(), elapsed)
    meta = _meta(G, cfg, grown_from=start_id, gen_time_s=elapsed)
    return G, meta


def save_snapshot(
    G: nx.DiGraph,
    out_pkl: Path,
    meta: Optional[Dict] = None,
) -> None:
    """Save the graph as a pickle (matches M1/M3 format) plus a sidecar JSON."""
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    if meta is not None:
        with open(out_pkl.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    logger.info("Saved %s (%d nodes / %d edges)",
                out_pkl, G.number_of_nodes(), G.number_of_edges())


def load_snapshot(pkl_path: Path) -> nx.DiGraph:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_delta(prev_G: nx.DiGraph, new_G: nx.DiGraph) -> Dict:
    """
    Compute the delta between two snapshots in the exact dict shape that
    Milestone 3's incremental_pagerank.compute_dirty_set() expects:
      - new_source_nodes: nodes that exist in new_G but not prev_G AND have out-edges
      - new_target_nodes: nodes that exist in new_G but not prev_G AND have in-edges
      - affected_existing_sources: existing nodes that gained a new out-edge
      - new_edges: list of (src, dst) tuples
    """
    prev_nodes = set(prev_G.nodes())
    new_nodes_set = set(new_G.nodes()) - prev_nodes
    prev_edges = set(prev_G.edges())
    new_edges_set = set(new_G.edges()) - prev_edges

    new_source_nodes: List[str] = []
    new_target_nodes: List[str] = []
    for u in new_nodes_set:
        if new_G.out_degree(u) > 0:
            new_source_nodes.append(u)
        if new_G.in_degree(u) > 0:
            new_target_nodes.append(u)

    affected_existing_sources: Set[str] = set()
    for src, dst in new_edges_set:
        if src in prev_nodes:
            affected_existing_sources.add(src)

    return {
        "new_source_nodes": new_source_nodes,
        "new_target_nodes": new_target_nodes,
        "affected_existing_sources": sorted(affected_existing_sources),
        "new_edges": [list(e) for e in new_edges_set],
        "summary": {
            "prev_nodes": len(prev_nodes),
            "new_nodes": len(new_nodes_set),
            "prev_edges": len(prev_edges),
            "new_edges": len(new_edges_set),
        },
    }


def save_delta(delta: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(delta, f)
    logger.info("Saved delta %s (new_nodes=%d, new_edges=%d)",
                out_json, delta["summary"]["new_nodes"],
                delta["summary"]["new_edges"])


def _url_of(node_id: int) -> str:
    return f"syn://node/{node_id}"


def _id_of(url: str) -> int:
    return int(url.rsplit("/", 1)[1])


def _meta(G: nx.DiGraph, cfg: GenConfig, **extra) -> Dict:
    return {
        "config": asdict(cfg),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        **extra,
    }
