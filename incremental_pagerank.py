
# incremental_pagerank.py: PageRank update strategies for a growing graph.
#
# Three strategies are implemented so the Milestone 3 evaluation can compare
# the cost-vs-accuracy trade-off the spec asks for:
#
#   1. run_full_recomputation(snapshot_path, partition_strategy)
#        Cold restart from r = 1/N. Reuses Milestone 2's parallel
#        infrastructure. This is the GROUND-TRUTH reference.
#
#   2. run_warm_start(snapshot_path, prev_scores, partition_strategy)
#        Starts iteration from prev_scores (padded with 1/N for new nodes,
#        rescaled so sum(r) == 1). Otherwise identical to the parallel
#        PageRank loop. Should converge in fewer iterations.
#
#   3. run_localised_update(snapshot_path, prev_scores, delta,
#                           partition_strategy, hop_radius)
#        Identifies a "dirty" set of nodes whose ranks are most likely
#        to change:
#            new nodes
#          + existing nodes that received a new in-edge
#          + their k-hop successor closure (k = hop_radius)
#        Iterates over only the dirty set, leaving non-dirty ranks at
#        their warm-start value. This trades accuracy for speed.
#
# All three strategies share the same Ray-based parallel inner loop
# (_run_parallel_pagerank) and use the partitioning strategy from
# partitioning_strategies.py. Every run produces a RuntimeStats object
# (instrumentation.py) containing all metrics.


import logging
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import ray
import networkx as nx

from instrumentation import (
    RuntimeStats,
    Stopwatch,
    estimate_serialised_bytes,
)
from partitioning_strategies import get_partitions, STRATEGY_RANGE
from m3_config import M3Config

# Import Milestone 2 modules. Mirror PYTHONPATH so Ray workers can import too.
_M2_DIR = (Path(__file__).resolve().parent.parent / "Milestone 2")
_M3_DIR = Path(__file__).resolve().parent
for _dir in (_M2_DIR, _M3_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

_existing_pp = os.environ.get("PYTHONPATH", "")
_pp_parts = [p for p in (str(_M2_DIR), str(_M3_DIR), _existing_pp) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(_pp_parts)

from pagerank_parallel import compute_partial_ranks      # noqa: E402

logger = logging.getLogger(__name__)



# Graph-loading helpers (URL-keyed -> integer-indexed structures).

def load_graph_for_pagerank(snapshot_path: str) -> Dict:
    """
    Load a snapshot pickle and return integer-indexed structures suitable
    for parallel PageRank, mirroring graph_loader.load_graph from M2 but
    independent of the M2 config object.
    """
    with open(snapshot_path, "rb") as f:
        G: nx.DiGraph = pickle.load(f)

    nodes: List[str] = list(G.nodes())
    url_to_id: Dict[str, int] = {url: i for i, url in enumerate(nodes)}
    num_nodes = len(nodes)

    in_links: List[List[int]] = [[] for _ in range(num_nodes)]
    out_degree: List[int] = [0] * num_nodes
    for src_url, dst_url in G.edges():
        s = url_to_id[src_url]
        t = url_to_id[dst_url]
        in_links[t].append(s)
        out_degree[s] += 1
    dangling: Set[int] = {i for i in range(num_nodes) if out_degree[i] == 0}

    return {
        "graph":      G,
        "nodes":      nodes,
        "url_to_id":  url_to_id,
        "in_links":   in_links,
        "out_degree": out_degree,
        "dangling":   dangling,
        "num_nodes":  num_nodes,
        "num_edges":  G.number_of_edges(),
    }



# Initial-rank construction (cold / warm).

def initial_ranks_uniform(num_nodes: int) -> List[float]:
    if num_nodes <= 0:
        return []
    return [1.0 / num_nodes] * num_nodes


def initial_ranks_warm(
    nodes: List[str],
    prev_scores: Dict[str, float],
) -> List[float]:
    """
    Build an initial rank vector for a graph that has grown since
    `prev_scores` was computed:
      - existing URLs keep their previous rank
      - new URLs get 1/N
      - the whole vector is rescaled so sum == 1
    """
    n = len(nodes)
    if n == 0:
        return []
    base = 1.0 / n
    ranks = [prev_scores.get(url, base) for url in nodes]
    s = sum(ranks)
    if s > 0:
        ranks = [r / s for r in ranks]
    return ranks



# Dirty-set computation for localised updates.

def compute_dirty_set(
    delta: Dict,
    url_to_id: Dict[str, int],
    in_links: List[List[int]],
    out_links: List[List[int]],
    hop_radius: int,
) -> Set[int]:
    """
    Compute the set of node ids whose ranks should be recomputed in the
    localised-update strategy.

    Initial seed:
        - all new source nodes
        - all new target nodes
        - all existing sources that received a new in-edge

    Then expand by hop_radius hops via SUCCESSORS (forward propagation),
    because rank changes propagate from a node to its out-neighbours
    over PageRank iterations.
    """
    seed: Set[int] = set()

    for url in delta.get("new_source_nodes", []):
        if url in url_to_id:
            seed.add(url_to_id[url])
    for url in delta.get("new_target_nodes", []):
        if url in url_to_id:
            seed.add(url_to_id[url])
    for url in delta.get("affected_existing_sources", []):
        if url in url_to_id:
            seed.add(url_to_id[url])

    # Also flag the destinations of new edges - their in-link sums changed.
    for src_url, dst_url in delta.get("new_edges", []):
        if dst_url in url_to_id:
            seed.add(url_to_id[dst_url])
        if src_url in url_to_id:
            seed.add(url_to_id[src_url])

    if hop_radius <= 0 or not seed:
        return seed

    visited: Set[int] = set(seed)
    frontier = deque(seed)
    depth_marker = -1
    frontier.append(depth_marker)
    depth = 0

    while frontier and depth < hop_radius:
        u = frontier.popleft()
        if u == depth_marker:
            depth += 1
            if frontier:
                frontier.append(depth_marker)
            continue
        for v in out_links[u]:
            if v not in visited:
                visited.add(v)
                frontier.append(v)
    return visited


def build_out_links(
    in_links: List[List[int]],
    num_nodes: int,
) -> List[List[int]]:
    """Reverse the in-link index to get an out-link adjacency."""
    out_links: List[List[int]] = [[] for _ in range(num_nodes)]
    for u in range(num_nodes):
        for v in in_links[u]:
            out_links[v].append(u)
    return out_links



# Core parallel inner loop (used by all three strategies).

def _run_parallel_pagerank(
    nodes: List[str],
    in_links: List[List[int]],
    out_degree: List[int],
    dangling: Set[int],
    initial_ranks: List[float],
    update_node_set: Optional[Set[int]],
    cfg: M3Config,
    stats: RuntimeStats,
) -> Tuple[List[float], int]:
    """
    Generic Ray-parallel PageRank loop.

    update_node_set:
      - None  -> recompute every node every iteration (full / warm).
      - set() -> recompute only nodes in this set every iteration (localised).
                 Non-listed nodes keep their initial rank.
    """
    n = len(nodes)
    d = cfg.damping_factor
    ranks = list(initial_ranks)

    # Decide which nodes participate in this run.
    if update_node_set is None:
        active = list(range(n))
    else:
        active = sorted(update_node_set)
    num_active = len(active)

    # Partition the active set with the configured strategy.
    # For non-range strategies that work over the full node set, we partition
    # the active list itself by re-mapping ids.
    partitions: List[List[int]]
    if update_node_set is None:
        partitions = get_partitions(
            cfg.partition_strategy, n, cfg.num_partitions, in_links=in_links,
        )
    else:
        # Partition only the active subset.
        if cfg.partition_strategy == STRATEGY_RANGE:
            chunk = max(1, num_active // cfg.num_partitions)
            partitions = []
            for i in range(cfg.num_partitions):
                start = i * chunk
                end = start + chunk if i < cfg.num_partitions - 1 else num_active
                partitions.append(active[start:end])
        elif cfg.partition_strategy == "hash":
            partitions = [[] for _ in range(cfg.num_partitions)]
            for u in active:
                partitions[u % cfg.num_partitions].append(u)
        else:
            # edge_balanced: greedy-LPT on |in_links[u]| for u in active
            import heapq
            work = {u: len(in_links[u]) + 1 for u in active}
            order = sorted(active, key=lambda u: -work[u])
            heap = [(0, i) for i in range(cfg.num_partitions)]
            heapq.heapify(heap)
            partitions = [[] for _ in range(cfg.num_partitions)]
            for u in order:
                load, idx = heapq.heappop(heap)
                partitions[idx].append(u)
                heapq.heappush(heap, (load + work[u], idx))
            for p in partitions:
                p.sort()

    logger.info(
        "Partitioned %d active nodes into %d chunks "
        "(strategy=%s sizes=%s)",
        num_active, len(partitions), cfg.partition_strategy,
        [len(p) for p in partitions],
    )

    # Place large read-only structures in Ray's object store ONCE.
    in_links_ref = ray.put(in_links)
    out_degree_ref = ray.put(out_degree)

    # Per-iteration loop.
    for it in range(1, cfg.max_iterations + 1):
        iter_start = time.perf_counter()
        with Stopwatch() as sw_pre:
            dangling_sum = sum(ranks[i] for i in dangling)
            ranks_ref = ray.put(ranks)

        # Estimate communication volume for this iteration:
        #   - rank vector broadcast (ray.put + worker reads)
        #   - returned partial-rank lists (one per partition)
        rank_vector_bytes = estimate_serialised_bytes(ranks)
        comm_bytes = rank_vector_bytes * (1 + len(partitions))  # broadcast + returns

        # Dispatch tasks.
        with Stopwatch() as sw_dispatch:
            futures = [
                compute_partial_ranks.remote(
                    p, ranks_ref, in_links_ref, out_degree_ref,
                    dangling_sum, n, d,
                )
                for p in partitions if p
            ]

        # Synchronisation barrier.
        with Stopwatch() as sw_barrier:
            results = ray.get(futures) if futures else []

        # Merge results on the driver.
        with Stopwatch() as sw_merge:
            new_ranks = list(ranks)  # carry forward non-active ranks
            task_times: List[float] = []
            worker_mems: List[float] = []
            for node_ids, partial_ranks, task_dur, mem_mb in results:
                for nid, r in zip(node_ids, partial_ranks):
                    new_ranks[nid] = r
                task_times.append(task_dur)
                worker_mems.append(mem_mb)

            delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
            ranks = new_ranks

        iter_wall = time.perf_counter() - iter_start
        compute_time = sw_pre.elapsed + sw_dispatch.elapsed + sw_merge.elapsed

        stats.record_iteration(
            iteration=it,
            wall_time_s=iter_wall,
            barrier_wait_s=sw_barrier.elapsed,
            compute_time_s=compute_time,
            delta=delta,
            comm_bytes=comm_bytes,
            worker_task_times_s=task_times,
            worker_peak_rss_mb=worker_mems,
            dirty_node_count=(num_active if update_node_set is not None else None),
        )

        logger.info(
            "  iter %3d | delta=%.2e | wall=%.4fs | barrier=%.4fs | "
            "tasks: min=%.4fs max=%.4fs",
            it, delta, iter_wall, sw_barrier.elapsed,
            min(task_times) if task_times else 0.0,
            max(task_times) if task_times else 0.0,
        )

        if cfg.use_convergence and delta < cfg.convergence_threshold:
            logger.info("Converged at iteration %d (delta=%.2e).", it, delta)
            break

    return ranks, it



# Public entry points: full / warm / localised.

def run_full_recomputation(
    snapshot_path: str,
    cfg: M3Config,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], RuntimeStats]:
    g = load_graph_for_pagerank(snapshot_path)
    nodes = g["nodes"]
    n = g["num_nodes"]
    logger.info("FULL recomputation | nodes=%d edges=%d strategy=%s workers=%d",
                n, g["num_edges"], cfg.partition_strategy, cfg.num_workers)

    metadata = {
        "update_strategy": "full",
        "partition_strategy": cfg.partition_strategy,
        "num_workers": cfg.num_workers,
        "num_nodes": n,
        "num_edges": g["num_edges"],
        "damping_factor": cfg.damping_factor,
        "convergence_threshold": cfg.convergence_threshold,
        "snapshot_path": snapshot_path,
    }
    stats = RuntimeStats(run_name or f"full_{cfg.partition_strategy}",
                         run_metadata=metadata)

    initial = initial_ranks_uniform(n)
    final_ranks, num_iter = _run_parallel_pagerank(
        nodes, g["in_links"], g["out_degree"], g["dangling"],
        initial, None, cfg, stats,
    )
    stats.finish()
    stats.set_extra("num_iterations", num_iter)
    stats.set_extra("rank_sum", sum(final_ranks))

    scores = {nodes[i]: final_ranks[i] for i in range(n)}
    if output_dir:
        stats.save(output_dir)
        _save_scores(scores, output_dir)
    return scores, stats


def run_warm_start(
    snapshot_path: str,
    prev_scores: Dict[str, float],
    cfg: M3Config,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], RuntimeStats]:
    g = load_graph_for_pagerank(snapshot_path)
    nodes = g["nodes"]
    n = g["num_nodes"]
    logger.info("WARM-START | nodes=%d edges=%d strategy=%s workers=%d "
                "(prev_scores=%d)", n, g["num_edges"],
                cfg.partition_strategy, cfg.num_workers, len(prev_scores))

    metadata = {
        "update_strategy": "warm",
        "partition_strategy": cfg.partition_strategy,
        "num_workers": cfg.num_workers,
        "num_nodes": n,
        "num_edges": g["num_edges"],
        "prev_scores_count": len(prev_scores),
        "damping_factor": cfg.damping_factor,
        "convergence_threshold": cfg.convergence_threshold,
        "snapshot_path": snapshot_path,
    }
    stats = RuntimeStats(run_name or f"warm_{cfg.partition_strategy}",
                         run_metadata=metadata)

    initial = initial_ranks_warm(nodes, prev_scores)
    final_ranks, num_iter = _run_parallel_pagerank(
        nodes, g["in_links"], g["out_degree"], g["dangling"],
        initial, None, cfg, stats,
    )
    stats.finish()
    stats.set_extra("num_iterations", num_iter)
    stats.set_extra("rank_sum", sum(final_ranks))

    scores = {nodes[i]: final_ranks[i] for i in range(n)}
    if output_dir:
        stats.save(output_dir)
        _save_scores(scores, output_dir)
    return scores, stats


def run_sequential(
    snapshot_path: str,
    cfg: M3Config,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], RuntimeStats]:
    """
    Single-threaded PageRank using the same iterative formula as the
    parallel implementations. Used as the speedup baseline in Group A.
    Records identical RuntimeStats so cross-strategy comparison is direct.
    """
    g = load_graph_for_pagerank(snapshot_path)
    nodes = g["nodes"]
    in_links = g["in_links"]
    out_degree = g["out_degree"]
    dangling = g["dangling"]
    n = g["num_nodes"]
    d = cfg.damping_factor

    logger.info("SEQUENTIAL | nodes=%d edges=%d", n, g["num_edges"])

    metadata = {
        "update_strategy": "sequential",
        "partition_strategy": "n/a",
        "num_workers": 1,
        "num_nodes": n,
        "num_edges": g["num_edges"],
        "damping_factor": d,
        "convergence_threshold": cfg.convergence_threshold,
        "snapshot_path": snapshot_path,
    }
    stats = RuntimeStats(run_name or "sequential", run_metadata=metadata)

    ranks = initial_ranks_uniform(n)

    for it in range(1, cfg.max_iterations + 1):
        iter_start = time.perf_counter()
        with Stopwatch() as sw_compute:
            dangling_sum = sum(ranks[i] for i in dangling)
            new_ranks = [0.0] * n
            for u in range(n):
                base = (1.0 - d) / n
                dangling_contrib = d * dangling_sum / n
                link_contrib = d * sum(
                    ranks[v] / out_degree[v]
                    for v in in_links[u] if out_degree[v] > 0
                )
                new_ranks[u] = base + dangling_contrib + link_contrib
            delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
            ranks = new_ranks
        iter_wall = time.perf_counter() - iter_start

        stats.record_iteration(
            iteration=it,
            wall_time_s=iter_wall,
            barrier_wait_s=0.0,
            compute_time_s=sw_compute.elapsed,
            delta=delta,
            comm_bytes=0,
            worker_task_times_s=[sw_compute.elapsed],
            worker_peak_rss_mb=[],
            dirty_node_count=None,
        )
        logger.info("  iter %3d | delta=%.2e | wall=%.4fs", it, delta, iter_wall)
        if cfg.use_convergence and delta < cfg.convergence_threshold:
            logger.info("Converged at iteration %d (delta=%.2e).", it, delta)
            break

    stats.finish()
    stats.set_extra("num_iterations", it)
    stats.set_extra("rank_sum", sum(ranks))

    scores = {nodes[i]: ranks[i] for i in range(n)}
    if output_dir:
        stats.save(output_dir)
        _save_scores(scores, output_dir)
    return scores, stats


def run_localised_update(
    snapshot_path: str,
    prev_scores: Dict[str, float],
    delta: Dict,
    cfg: M3Config,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], RuntimeStats]:
    g = load_graph_for_pagerank(snapshot_path)
    nodes = g["nodes"]
    n = g["num_nodes"]
    in_links = g["in_links"]
    out_links = build_out_links(in_links, n)

    dirty = compute_dirty_set(
        delta, g["url_to_id"], in_links, out_links,
        hop_radius=cfg.affected_hop_radius,
    )
    logger.info(
        "LOCALISED | nodes=%d (dirty=%d, %.1f%%) edges=%d strategy=%s workers=%d "
        "k_hops=%d", n, len(dirty), 100.0 * len(dirty) / n if n else 0.0,
        g["num_edges"], cfg.partition_strategy, cfg.num_workers,
        cfg.affected_hop_radius,
    )

    metadata = {
        "update_strategy": "localised",
        "partition_strategy": cfg.partition_strategy,
        "num_workers": cfg.num_workers,
        "num_nodes": n,
        "num_edges": g["num_edges"],
        "dirty_node_count": len(dirty),
        "dirty_pct": 100.0 * len(dirty) / n if n else 0.0,
        "hop_radius": cfg.affected_hop_radius,
        "damping_factor": cfg.damping_factor,
        "convergence_threshold": cfg.convergence_threshold,
        "snapshot_path": snapshot_path,
    }
    stats = RuntimeStats(run_name or f"localised_{cfg.partition_strategy}",
                         run_metadata=metadata)

    initial = initial_ranks_warm(nodes, prev_scores)
    final_ranks, num_iter = _run_parallel_pagerank(
        nodes, in_links, g["out_degree"], g["dangling"],
        initial, dirty, cfg, stats,
    )

    # Re-normalise so sum == 1 (localised updates can drift).
    s = sum(final_ranks)
    if s > 0:
        final_ranks = [r / s for r in final_ranks]

    stats.finish()
    stats.set_extra("num_iterations", num_iter)
    stats.set_extra("rank_sum", sum(final_ranks))

    scores = {nodes[i]: final_ranks[i] for i in range(n)}
    if output_dir:
        stats.save(output_dir)
        _save_scores(scores, output_dir)
    return scores, stats


# Accuracy comparison helper (used by Group C of the evaluation).

def compare_to_reference(
    test_scores: Dict[str, float],
    reference_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute simple accuracy metrics between two PageRank score sets sharing
    the same node URLs (the universe is reference_scores).
    """
    if not reference_scores:
        return {"max_abs_diff": 0.0, "l1_diff": 0.0, "l2_diff": 0.0,
                "missing_nodes": 0}
    diffs: List[float] = []
    missing = 0
    for url, ref in reference_scores.items():
        if url not in test_scores:
            missing += 1
            continue
        diffs.append(abs(test_scores[url] - ref))
    if not diffs:
        return {"max_abs_diff": float("inf"), "l1_diff": float("inf"),
                "l2_diff": float("inf"), "missing_nodes": missing}
    max_abs = max(diffs)
    l1 = sum(diffs)
    l2 = (sum(x * x for x in diffs)) ** 0.5
    return {
        "max_abs_diff": max_abs,
        "l1_diff": l1,
        "l2_diff": l2,
        "missing_nodes": missing,
    }



# Internal: persist scores to disk.

def _save_scores(scores: Dict[str, float], output_dir: str) -> str:
    import json
    os.makedirs(output_dir, exist_ok=True)
    top1k = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000])
    path = os.path.join(output_dir, "scores_top1000.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rank_sum": sum(scores.values()),
                   "node_count": len(scores),
                   "top_1000": top1k}, f, indent=2)
    full_path = os.path.join(output_dir, "scores_full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(scores, f)
    return path
