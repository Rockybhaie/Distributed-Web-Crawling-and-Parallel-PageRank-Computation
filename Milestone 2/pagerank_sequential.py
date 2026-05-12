# pagerank_sequential.py — Sequential PageRank baseline for Milestone 2.
# This file contains two implementations of PageRank:
# 1. networkx_pagerank()  — uses NetworkX's built-in PageRank function as a gold-standard reference.
# 2. custom_pagerank()    — our own iterative implementation, which will be parallelised in the next steps. 


import logging
import time
from typing import Dict, List, Set, Tuple

import networkx as nx

from pagerank_config import PageRankConfig

logger = logging.getLogger(__name__)


def networkx_pagerank(
    G:      nx.DiGraph,
    config: PageRankConfig,
) -> Tuple[Dict[str, float], float, int]:
    """
    Run PageRank using NetworkX's built-in scipy implementation.
    Used as a reference to validate correctness of our custom version.
    """
    logger.info(
        "Running NetworkX PageRank | nodes=%d  edges=%d  d=%.2f",
        G.number_of_nodes(), G.number_of_edges(), config.damping_factor,
    )
    t0     = time.perf_counter()
    scores = nx.pagerank(
        G,
        alpha    = config.damping_factor,
        tol      = config.convergence_threshold,
        max_iter = config.max_iterations,
    )
    elapsed = time.perf_counter() - t0
    logger.info("NetworkX PageRank done | time=%.3fs  top_node=%s",
                elapsed, max(scores, key=scores.get))
    return scores, elapsed, -1


def custom_pagerank(
    nodes:      List[str],
    in_links:   List[List[int]],
    out_degree: List[int],
    dangling:   Set[int],
    config:     PageRankConfig,
) -> Tuple[Dict[str, float], float, int, List[float]]:
    
    N = len(nodes)
    d = config.damping_factor

    logger.info(
        "Running custom sequential PageRank | nodes=%d  d=%.2f  max_iter=%d",
        N, d, config.max_iterations,
    )

    ranks  = [1.0 / N] * N
    deltas = []

    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        dangling_sum = sum(ranks[i] for i in dangling)
        new_ranks    = [0.0] * N

        for u in range(N):
            base             = (1.0 - d) / N
            dangling_contrib = d * dangling_sum / N
            link_contrib     = d * sum(
                ranks[v] / out_degree[v]
                for v in in_links[u]
                if out_degree[v] > 0
            )
            new_ranks[u] = base + dangling_contrib + link_contrib

        delta  = sum(abs(new_ranks[i] - ranks[i]) for i in range(N))
        ranks  = new_ranks
        deltas.append(delta)

        logger.info("  Iteration %3d | delta=%.2e  (threshold=%.2e)",
                    iteration + 1, delta, config.convergence_threshold)

        if config.use_convergence and delta < config.convergence_threshold:
            logger.info("Converged at iteration %d | delta=%.2e",
                        iteration + 1, delta)
            break

    elapsed        = time.perf_counter() - t0
    num_iterations = iteration + 1
    scores         = {nodes[i]: ranks[i] for i in range(N)}

    logger.info("Custom sequential PageRank done | time=%.3fs  iterations=%d",
                elapsed, num_iterations)

    return scores, elapsed, num_iterations, deltas


def validate_scores(
    scores_ref:  Dict[str, float],
    scores_test: Dict[str, float],
    tolerance:   float = 1e-4,
) -> bool:
    # Check that two PageRank score sets agree within tolerance.
    max_diff  = 0.0
    worst     = None
    for node in scores_ref:
        if node not in scores_test:
            logger.warning("Node missing from test scores: %s", node)
            return False
        diff = abs(scores_ref[node] - scores_test[node])
        if diff > max_diff:
            max_diff = diff
            worst    = node
    passed = max_diff <= tolerance
    logger.info("Validation %s | max_diff=%.2e  worst_node=%s",
                "PASSED" if passed else "FAILED", max_diff, worst)
    return passed