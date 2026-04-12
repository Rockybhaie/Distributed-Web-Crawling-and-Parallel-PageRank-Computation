# pagerank_parallel.py — Parallel PageRank using Ray (Milestone 2 Core File).

"""
Performance Metrics Collected Per Run
- Per-iteration wall-clock time
- Per-task duration (min/max/avg) for load imbalance analysis
- Memory usage per worker (RSS in MB via psutil)
- Communication volume per iteration (bytes)
- Convergence delta per iteration

Two Execution Strategies
A. Centralised Aggregation  — workers return partial ranks to driver
B. Distributed Reduction    — workers write directly to shared RankActor
"""

import logging
import os
import time
from typing import Dict, List, Set, Tuple


import ray

from pagerank_config import PageRankConfig

logger = logging.getLogger(__name__)



#  Strategy B Actor                                                             

@ray.remote
class RankActor:
    # Shared Actor for Distributed Reduction strategy.

    def __init__(self, initial_ranks: List[float]) -> None:
        self._ranks       = list(initial_ranks)
        self._accumulator = [0.0] * len(initial_ranks)

    def add_partial_ranks(self, node_ids: List[int], partial_ranks: List[float]) -> None:
        for node_id, rank in zip(node_ids, partial_ranks):
            self._accumulator[node_id] = rank

    def finalise_iteration(self) -> Tuple[List[float], float]:
        max_delta = max(
            abs(self._accumulator[i] - self._ranks[i])
            for i in range(len(self._ranks))
        )
        self._ranks       = list(self._accumulator)
        self._accumulator = [0.0] * len(self._ranks)
        return self._ranks, max_delta

    def get_ranks(self) -> List[float]:
        return self._ranks



#  Core Ray Task                                                               

@ray.remote
def compute_partial_ranks(
    node_partition: List[int],
    current_ranks:  List[float],
    in_links:       List[List[int]],
    out_degree:     List[int],
    dangling_sum:   float,
    num_nodes:      int,
    damping_factor: float,
) -> Tuple[List[int], List[float], float, float]:
    
    # Memory before
    try:
        import psutil
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        mem_before = 0.0

    task_start = time.perf_counter()

    d         = damping_factor
    N         = num_nodes
    new_ranks = []

    for u in node_partition:
        base             = (1.0 - d) / N
        dangling_contrib = d * dangling_sum / N
        link_contrib     = d * sum(
            current_ranks[v] / out_degree[v]
            for v in in_links[u]
            if out_degree[v] > 0
        )
        new_ranks.append(base + dangling_contrib + link_contrib)

    task_duration = time.perf_counter() - task_start

    # Memory after
    try:
        import psutil
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        mem_used  = max(mem_after - mem_before, 0.0)
    except Exception:
        mem_used = 0.0

    return node_partition, new_ranks, task_duration, mem_used



#  Strategy A — Centralised Aggregation                                         

def run_pagerank_centralized(
    nodes:      List[str],
    in_links:   List[List[int]],
    out_degree: List[int],
    dangling:   Set[int],
    partitions: List[List[int]],
    config:     PageRankConfig,
) -> Tuple[Dict[str, float], float, int, Dict]:
    
    N = len(nodes)
    d = config.damping_factor

    logger.info(
        "Parallel PageRank [Centralised] | nodes=%d  workers=%d  partitions=%d",
        N, config.num_workers, len(partitions),
    )

    # Store large read-only structures in Ray object store once
    in_links_ref   = ray.put(in_links)
    out_degree_ref = ray.put(out_degree)

    ranks = [1.0 / N] * N

    iter_times:      List[float]       = []
    iter_deltas:     List[float]       = []
    all_task_times:  List[float]       = []
    worker_memories: List[float]       = []

    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        iter_start   = time.perf_counter()
        dangling_sum = sum(ranks[i] for i in dangling)
        ranks_ref    = ray.put(ranks)

        # Dispatch all partition tasks concurrently
        futures = [
            compute_partial_ranks.remote(
                partition, ranks_ref, in_links_ref, out_degree_ref,
                dangling_sum, N, d,
            )
            for partition in partitions
        ]

        # ---- SYNCHRONISATION BARRIER ----
        results = ray.get(futures)

        # Centralised merge on driver
        new_ranks = [0.0] * N
        this_task_times = []
        for node_ids, partial_ranks, task_dur, mem_mb in results:
            for node_id, rank in zip(node_ids, partial_ranks):
                new_ranks[node_id] = rank
            this_task_times.append(task_dur)
            worker_memories.append(mem_mb)

        delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(N))
        ranks = new_ranks

        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        iter_deltas.append(delta)
        all_task_times.extend(this_task_times)

        logger.info(
            "  [Centralised] Iteration %3d | delta=%.2e | iter_time=%.4fs | "
            "task min=%.4fs max=%.4fs",
            iteration + 1, delta, iter_elapsed,
            min(this_task_times), max(this_task_times),
        )

        if config.use_convergence and delta < config.convergence_threshold:
            logger.info("[Centralised] Converged at iteration %d | delta=%.2e",
                        iteration + 1, delta)
            break

    elapsed        = time.perf_counter() - t0
    num_iterations = iteration + 1

    metrics = _build_metrics(
        iter_times, iter_deltas, all_task_times,
        worker_memories, num_iterations, N,
    )

    scores = {nodes[i]: ranks[i] for i in range(N)}
    logger.info("[Centralised] Done | time=%.3fs  iterations=%d  "
                "load_imbalance=%.1f%%  comm=%.1fKB",
                elapsed, num_iterations,
                metrics["load_imbalance"] * 100, metrics["comm_kb_total"])

    return scores, elapsed, num_iterations, metrics



#  Strategy B — Distributed Reduction                                           

def run_pagerank_distributed(
    nodes:      List[str],
    in_links:   List[List[int]],
    out_degree: List[int],
    dangling:   Set[int],
    partitions: List[List[int]],
    config:     PageRankConfig,
) -> Tuple[Dict[str, float], float, int, Dict]:
    
    N = len(nodes)
    d = config.damping_factor

    logger.info(
        "Parallel PageRank [Distributed] | nodes=%d  workers=%d  partitions=%d",
        N, config.num_workers, len(partitions),
    )

    in_links_ref   = ray.put(in_links)
    out_degree_ref = ray.put(out_degree)

    actor         = RankActor.remote([1.0 / N] * N)
    iter_times:      List[float] = []
    iter_deltas:     List[float] = []
    all_task_times:  List[float] = []
    worker_memories: List[float] = []

    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        iter_start    = time.perf_counter()
        current_ranks = ray.get(actor.get_ranks.remote())
        dangling_sum  = sum(current_ranks[i] for i in dangling)
        ranks_ref     = ray.put(current_ranks)

        futures = [
            _distributed_worker.remote(
                partition, ranks_ref, in_links_ref, out_degree_ref,
                dangling_sum, N, d, actor,
            )
            for partition in partitions
        ]

        worker_results = ray.get(futures)
        new_ranks, delta = ray.get(actor.finalise_iteration.remote())

        iter_elapsed = time.perf_counter() - iter_start
        this_task_times = [r[0] for r in worker_results]
        iter_times.append(iter_elapsed)
        iter_deltas.append(delta)
        all_task_times.extend(this_task_times)
        worker_memories.extend([r[1] for r in worker_results])

        logger.info(
            "  [Distributed] Iteration %3d | delta=%.2e | iter_time=%.4fs",
            iteration + 1, delta, iter_elapsed,
        )

        if config.use_convergence and delta < config.convergence_threshold:
            logger.info("[Distributed] Converged at iteration %d | delta=%.2e",
                        iteration + 1, delta)
            break

    elapsed        = time.perf_counter() - t0
    num_iterations = iteration + 1

    metrics = _build_metrics(
        iter_times, iter_deltas, all_task_times,
        worker_memories, num_iterations, N,
    )

    final_ranks = ray.get(actor.get_ranks.remote())
    scores      = {nodes[i]: final_ranks[i] for i in range(N)}

    logger.info("[Distributed] Done | time=%.3fs  iterations=%d  "
                "load_imbalance=%.1f%%",
                elapsed, num_iterations, metrics["load_imbalance"] * 100)

    return scores, elapsed, num_iterations, metrics


@ray.remote
def _distributed_worker(
    node_partition: List[int],
    current_ranks:  List[float],
    in_links:       List[List[int]],
    out_degree:     List[int],
    dangling_sum:   float,
    num_nodes:      int,
    damping_factor: float,
    actor:          RankActor,
) -> Tuple[float, float]:
    node_ids, partial_ranks, task_dur, mem_mb = ray.get(
        compute_partial_ranks.remote(
            node_partition, current_ranks, in_links, out_degree,
            dangling_sum, num_nodes, damping_factor,
        )
    )
    actor.add_partial_ranks.remote(node_ids, partial_ranks)
    return task_dur, mem_mb



#  Internal helpers                                                              

def _build_metrics(
    iter_times:      List[float],
    iter_deltas:     List[float],
    all_task_times:  List[float],
    worker_memories: List[float],
    num_iterations:  int,
    num_nodes:       int,
) -> Dict:
    safe_max = max(all_task_times) if all_task_times else 1.0
    safe_min = min(all_task_times) if all_task_times else 0.0

    return {
        "iter_times":       iter_times,
        "iter_deltas":      iter_deltas,
        "all_task_times":   all_task_times,
        "worker_memories":  worker_memories,
        "avg_iter_time":    sum(iter_times) / len(iter_times) if iter_times else 0,
        "min_task_time":    safe_min,
        "max_task_time":    safe_max,
        "avg_task_time":    sum(all_task_times) / len(all_task_times) if all_task_times else 0,
        "load_imbalance":   (safe_max - safe_min) / safe_max if safe_max > 0 else 0,
        "peak_memory_mb":   max(worker_memories) if worker_memories else 0,
        "comm_bytes_total": num_nodes * 8 * num_iterations,
        "comm_kb_total":    num_nodes * 8 * num_iterations / 1024,
    }