
# partitioning_strategies.py: Multiple partitioning strategies for Milestone 3.
#
# Milestone 2 only used range-based partitioning. The Milestone 3 spec requires
# comparing "different partitioning strategies", so this module provides three:
#
#   1. partition_range(num_nodes, k)
#        Contiguous node-id ranges. Same as Milestone 2.
#        - Pros: simple, deterministic, equal partition sizes.
#        - Cons: ignores per-node work (in-link count varies a lot).
#
#   2. partition_hash(num_nodes, k)
#        node_id % k. Good random load balance, breaks adjacency locality.
#        - Pros: probabilistic balance of node-id-correlated workloads.
#        - Cons: destroys locality, increases cross-partition edges.
#
#   3. partition_edge_balanced(in_links, k)
#        Greedy LPT (longest-processing-time) packing on per-node work
#        estimate (= len(in_links[u]) + 1). Each node is assigned to the
#        currently-lightest partition.
#        - Pros: minimises load imbalance for the actual PageRank cost,
#          which is dominated by in-link aggregation:
#              for u: r_u = base + d * sum(r_v / out(v) for v in in_links[u])
#          so the work for node u is O(|in_links[u]|).
#        - Cons: more setup time, partitions are not contiguous.
#
# Each function returns a List[List[int]] - a list of partitions, each being a
# list of node ids. This signature is a drop-in replacement for the
# graph_loader.partition_nodes() function from Milestone 2.


import heapq
import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

# Names exposed via the CLI / config.
STRATEGY_RANGE = "range"
STRATEGY_HASH = "hash"
STRATEGY_EDGE_BALANCED = "edge_balanced"

VALID_STRATEGIES = (STRATEGY_RANGE, STRATEGY_HASH, STRATEGY_EDGE_BALANCED)



# 1. Range-based (same as Milestone 2)

def partition_range(num_nodes: int, k: int) -> List[List[int]]:
    if k <= 0:
        raise ValueError("Number of partitions must be > 0")
    if num_nodes <= 0:
        return [[] for _ in range(k)]

    chunk_size = max(1, num_nodes // k)
    partitions: List[List[int]] = []
    for i in range(k):
        start = i * chunk_size
        end = start + chunk_size if i < k - 1 else num_nodes
        partitions.append(list(range(start, end)))
    return partitions



# 2. Hash-based (round-robin on node_id mod k)

def partition_hash(num_nodes: int, k: int) -> List[List[int]]:
    if k <= 0:
        raise ValueError("Number of partitions must be > 0")
    partitions: List[List[int]] = [[] for _ in range(k)]
    for u in range(num_nodes):
        partitions[u % k].append(u)
    return partitions



# 3. Edge-balanced (greedy LPT packing on in-link counts)

def partition_edge_balanced(
    in_links: List[List[int]],
    k: int,
) -> List[List[int]]:
    """
    Pack nodes into k partitions so that the total per-node work
    (proxy: |in_links[u]| + 1) is roughly equal across partitions.

    Uses the classic Longest-Processing-Time (LPT) heuristic:
      - Sort nodes by descending estimated work.
      - Maintain a min-heap keyed by current partition load.
      - Assign each node to the currently-lightest partition.

    Output: list of k node-id lists. Within each list, ids are sorted
    ascending so iteration order remains deterministic.
    """
    if k <= 0:
        raise ValueError("Number of partitions must be > 0")
    n = len(in_links)
    if n == 0:
        return [[] for _ in range(k)]

    # Per-node work estimate. The "+ 1" prevents zero-work nodes (no in-edges)
    # from being treated as free, since they still pay loop / dispatch cost.
    work = [len(adj) + 1 for adj in in_links]

    # Sort node ids by descending work (LPT rule).
    order = sorted(range(n), key=lambda u: -work[u])

    # Min-heap keyed by (current_load, partition_index).
    heap = [(0, i) for i in range(k)]
    heapq.heapify(heap)
    buckets: List[List[int]] = [[] for _ in range(k)]

    for u in order:
        load, idx = heapq.heappop(heap)
        buckets[idx].append(u)
        heapq.heappush(heap, (load + work[u], idx))

    # Sort each bucket so downstream iteration order is deterministic.
    for b in buckets:
        b.sort()
    return buckets



# Convenience dispatcher

def get_partitions(
    strategy: str,
    num_nodes: int,
    num_partitions: int,
    in_links: List[List[int]] = None,
) -> List[List[int]]:
    s = strategy.lower().strip()
    if s == STRATEGY_RANGE:
        return partition_range(num_nodes, num_partitions)
    if s == STRATEGY_HASH:
        return partition_hash(num_nodes, num_partitions)
    if s == STRATEGY_EDGE_BALANCED:
        if in_links is None:
            raise ValueError(
                "edge_balanced partitioning requires in_links argument."
            )
        return partition_edge_balanced(in_links, num_partitions)
    raise ValueError(
        f"Unknown partitioning strategy '{strategy}'. "
        f"Valid: {VALID_STRATEGIES}"
    )



# Diagnostics: report how well-balanced a partitioning actually is.

def partition_diagnostics(
    partitions: List[List[int]],
    in_links: List[List[int]] = None,
) -> Dict[str, float]:
    sizes = [len(p) for p in partitions]
    diag: Dict[str, float] = {
        "num_partitions": len(partitions),
        "size_min": min(sizes) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "size_avg": (sum(sizes) / len(sizes)) if sizes else 0,
        "size_imbalance": (
            (max(sizes) - min(sizes)) / max(sizes) if sizes and max(sizes) > 0 else 0
        ),
    }
    if in_links is not None:
        loads = [
            sum(len(in_links[u]) + 1 for u in p) for p in partitions
        ]
        diag.update({
            "work_min": min(loads) if loads else 0,
            "work_max": max(loads) if loads else 0,
            "work_avg": (sum(loads) / len(loads)) if loads else 0,
            "work_imbalance": (
                (max(loads) - min(loads)) / max(loads) if loads and max(loads) > 0 else 0
            ),
        })
    return diag
