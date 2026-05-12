# performance_analysis.py — Comprehensive performance metrics for Milestone 2.

"""
Metrics computed and reported:
    1.  Execution time (sequential vs parallel)
    2.  Speedup  = T_sequential / T_parallel
    3.  Parallel efficiency = Speedup / num_workers
    4.  Amdahl's Law theoretical speedup
    5.  Communication overhead (bytes transferred)
    6.  Load imbalance (max task time vs min task time across workers)
    7.  Per-iteration timing breakdown
    8.  Convergence rate (delta per iteration)
    9.  Rank distribution analysis (Gini coefficient)
    10. Peak memory usage per worker
    11. Top-ranked pages
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)



#  Results container                                                             

class PageRankResults:
    # Holds all data from one PageRank run.

    def __init__(
        self,
        strategy:              str,
        num_workers:           int,
        num_nodes:             int,
        num_edges:             int,
        scores:                Dict[str, float],
        elapsed:               float,
        num_iterations:        int,
        damping_factor:        float,
        convergence_threshold: float,
        metrics:               Optional[Dict] = None,
        iter_deltas:           Optional[List[float]] = None,
    ):
        self.strategy              = strategy
        self.num_workers           = num_workers
        self.num_nodes             = num_nodes
        self.num_edges             = num_edges
        self.scores                = scores
        self.elapsed               = elapsed
        self.num_iterations        = num_iterations
        self.damping_factor        = damping_factor
        self.convergence_threshold = convergence_threshold
        self.metrics               = metrics or {}
        self.iter_deltas           = iter_deltas or []

    def speedup(self, sequential_time: float) -> float:
        if self.elapsed == 0:
            return float("inf")
        return sequential_time / self.elapsed

    def efficiency(self, sequential_time: float) -> float:
        return self.speedup(sequential_time) / max(self.num_workers, 1)

    def top_pages(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def score_sum(self) -> float:
        return sum(self.scores.values())

    def gini_coefficient(self) -> float:
        
        values = sorted(self.scores.values())
        n      = len(values)
        if n == 0:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        cumulative = 0.0
        gini_sum   = 0.0
        for i, v in enumerate(values):
            cumulative += v
            gini_sum   += cumulative
        # Standard Gini formula
        return (2.0 * gini_sum / (n * total)) - (n + 1) / n



#  Amdahl's Law                                                                 

def amdahl_speedup(parallel_fraction: float, num_workers: int) -> float:
    
    serial_fraction = 1.0 - parallel_fraction
    return 1.0 / (serial_fraction + parallel_fraction / num_workers)


def estimate_parallel_fraction(seq_time: float, par_time: float, num_workers: int) -> float:
    
    # Back-calculate the parallel fraction from observed speedup using Amdahl's Law.
    # Useful for characterising how parallelisable the workload actually is.
    
    observed_speedup = seq_time / par_time if par_time > 0 else 1.0
    # Solve Amdahl for p: S = 1/((1-p) + p/n)
    # => p = (1/S - 1) / (1/n - 1)
    if num_workers <= 1:
        return 0.0
    try:
        p = (1.0 / observed_speedup - 1.0) / (1.0 / num_workers - 1.0)
        return max(0.0, min(1.0, p))
    except ZeroDivisionError:
        return 0.0



#  Print helpers                                                                 
def print_results(results: PageRankResults, seq_time: Optional[float] = None) -> None:
    m = results.metrics
    print("\n" + "=" * 68)
    print(f"  PageRank Results — {results.strategy}")
    print("=" * 68)
    print(f"  Graph:            {results.num_nodes:>10,} nodes  |  {results.num_edges:>10,} edges")
    print(f"  Workers:          {results.num_workers:>10}")
    print(f"  Iterations:       {results.num_iterations:>10}")
    print(f"  Damping factor:   {results.damping_factor:>10.2f}")
    print(f"  Execution time:   {results.elapsed:>10.4f}s")
    print(f"  Rank sum:         {results.score_sum():>10.6f}  (should be ~1.0)")
    print(f"  Gini coefficient: {results.gini_coefficient():>10.4f}  (0=equal, 1=concentrated)")

    if seq_time is not None:
        sp  = results.speedup(seq_time)
        eff = results.efficiency(seq_time)
        p   = estimate_parallel_fraction(seq_time, results.elapsed, results.num_workers)
        print(f"  Speedup:          {sp:>10.4f}x")
        print(f"  Efficiency:       {eff:>10.4f}  (1.0 = perfect linear speedup)")
        print(f"  Parallel fraction:{p:>10.4f}  (Amdahl estimate)")

    if m:
        print(f"\n  Per-iteration metrics:")
        print(f"    Avg iter time:  {m.get('avg_iter_time', 0):>10.4f}s")
        print(f"    Min task time:  {m.get('min_task_time', 0):>10.4f}s")
        print(f"    Max task time:  {m.get('max_task_time', 0):>10.4f}s")
        print(f"    Avg task time:  {m.get('avg_task_time', 0):>10.4f}s")
        imb = m.get('load_imbalance', 0) * 100
        print(f"    Load imbalance: {imb:>10.1f}%  (0% = perfect balance)")
        print(f"    Peak memory:    {m.get('peak_memory_mb', 0):>10.2f} MB/worker")
        print(f"    Comm volume:    {m.get('comm_kb_total', 0):>10.1f} KB total")

    if results.iter_deltas:
        print(f"\n  Convergence (delta per iteration):")
        for i, delta in enumerate(results.iter_deltas, 1):
            bar = "█" * min(int(math.log10(1.0 / delta + 1) * 4), 30) if delta > 0 else ""
            print(f"    Iter {i:>3}: {delta:.4e}  {bar}")

    print(f"\n  Top 10 Pages by PageRank:")
    print("  " + "-" * 60)
    for rank, (url, score) in enumerate(results.top_pages(10), 1):
        short = url.replace("https://en.wikipedia.org/wiki/", "")[:45]
        print(f"  {rank:>2}. {score:.6f}  {short}")
    print("=" * 68 + "\n")


def print_full_comparison(
    seq:         PageRankResults,
    centralized: PageRankResults,
    distributed: PageRankResults,
) -> None:
    # Print the complete side-by-side performance comparison table.
    seq_t = seq.elapsed

    print("\n" + "=" * 80)
    print("  FULL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"  {'Strategy':<28} {'Time':>8} {'Speedup':>9} {'Efficiency':>11} "
          f"{'Iters':>6} {'CommKB':>8} {'Imbal%':>7}")
    print("  " + "-" * 80)

    rows = [
        (seq.strategy,         seq.elapsed,         1.0,                        1.0,                        seq.num_iterations,         seq.metrics.get('comm_kb_total',0),  0.0),
        (centralized.strategy, centralized.elapsed,  centralized.speedup(seq_t), centralized.efficiency(seq_t), centralized.num_iterations, centralized.metrics.get('comm_kb_total',0), centralized.metrics.get('load_imbalance',0)*100),
        (distributed.strategy, distributed.elapsed,  distributed.speedup(seq_t), distributed.efficiency(seq_t), distributed.num_iterations, distributed.metrics.get('comm_kb_total',0), distributed.metrics.get('load_imbalance',0)*100),
    ]

    for name, t, sp, eff, iters, comm, imb in rows:
        print(f"  {name:<28} {t:>8.4f} {sp:>8.3f}x {eff:>10.4f} "
              f"{iters:>6} {comm:>8.1f} {imb:>6.1f}%")

    print("=" * 80)

    # Amdahl analysis
    print("\n  AMDAHL'S LAW ANALYSIS")
    print("  " + "-" * 60)
    p_c = estimate_parallel_fraction(seq_t, centralized.elapsed, centralized.num_workers)
    p_d = estimate_parallel_fraction(seq_t, distributed.elapsed, distributed.num_workers)
    print(f"  Centralised — estimated parallel fraction: {p_c:.4f} ({p_c*100:.1f}%)")
    print(f"  Distributed — estimated parallel fraction: {p_d:.4f} ({p_d*100:.1f}%)")
    print()
    print(f"  Theoretical max speedup (Amdahl) at p={p_c:.2f}:")
    for n in [1, 2, 4, 8, 16]:
        th = amdahl_speedup(p_c, n)
        print(f"    {n:>3} workers → {th:.3f}x theoretical")

    # Communication analysis
    print("\n  COMMUNICATION OVERHEAD ANALYSIS")
    print("  " + "-" * 60)
    n_nodes = centralized.num_nodes
    n_iters = centralized.num_iterations
    bytes_per_iter = n_nodes * 8
    total_kb       = bytes_per_iter * n_iters / 1024
    print(f"  Centralised strategy:")
    print(f"    Rank vector size: {n_nodes:,} floats × 8 bytes = {bytes_per_iter/1024:.1f} KB")
    print(f"    Iterations:       {n_iters}")
    print(f"    Total comm:       {total_kb:.1f} KB (driver ↔ {centralized.num_workers} workers)")
    print(f"  Distributed strategy:")
    chunk = n_nodes // distributed.num_workers
    print(f"    Chunk size/worker: {chunk:,} floats × 8 bytes = {chunk*8/1024:.1f} KB")
    print(f"    Actor overhead:    additional IPC per write call")

    # Load imbalance
    print("\n  LOAD IMBALANCE ANALYSIS")
    print("  " + "-" * 60)
    c_imb = centralized.metrics.get('load_imbalance', 0) * 100
    d_imb = distributed.metrics.get('load_imbalance', 0) * 100
    print(f"  Centralised  — load imbalance: {c_imb:.1f}%")
    print(f"  Distributed  — load imbalance: {d_imb:.1f}%")
    print(f"  (0% = all workers finish at same time, 100% = extreme imbalance)")

    # Rank distribution
    print("\n  RANK DISTRIBUTION (Gini Coefficient)")
    print("  " + "-" * 60)
    print(f"  Sequential:   {seq.gini_coefficient():.4f}")
    print(f"  Centralised:  {centralized.gini_coefficient():.4f}")
    print(f"  Distributed:  {distributed.gini_coefficient():.4f}")
    print(f"  (All should be ~equal — confirms algorithm correctness)")
    print("=" * 80 + "\n")


def print_convergence_table(
    seq:         PageRankResults,
    centralized: PageRankResults,
    distributed: PageRankResults,
) -> None:
    # Print convergence delta per iteration for all strategies.
    print("\n  CONVERGENCE RATE COMPARISON")
    print("  " + "-" * 55)
    max_iters = max(len(seq.iter_deltas),
                    len(centralized.metrics.get('iter_deltas', [])),
                    len(distributed.metrics.get('iter_deltas', [])))

    c_deltas = centralized.metrics.get('iter_deltas', [])
    d_deltas = distributed.metrics.get('iter_deltas', [])

    print(f"  {'Iter':>4}  {'Sequential':>12}  {'Centralised':>12}  {'Distributed':>12}")
    print("  " + "-" * 55)
    for i in range(max_iters):
        s = f"{seq.iter_deltas[i]:.4e}"   if i < len(seq.iter_deltas) else "  converged"
        c = f"{c_deltas[i]:.4e}"          if i < len(c_deltas)        else "  converged"
        d = f"{d_deltas[i]:.4e}"          if i < len(d_deltas)        else "  converged"
        print(f"  {i+1:>4}  {s:>12}  {c:>12}  {d:>12}")
    print()



#  Save                                                                          

def save_results(results: PageRankResults, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    top_scores = dict(
        sorted(results.scores.items(), key=lambda x: x[1], reverse=True)[:1000]
    )
    output = {
        "strategy":              results.strategy,
        "num_workers":           results.num_workers,
        "num_nodes":             results.num_nodes,
        "num_edges":             results.num_edges,
        "elapsed_seconds":       results.elapsed,
        "num_iterations":        results.num_iterations,
        "damping_factor":        results.damping_factor,
        "convergence_threshold": results.convergence_threshold,
        "rank_sum":              results.score_sum(),
        "gini_coefficient":      results.gini_coefficient(),
        "metrics":               {k: v for k, v in results.metrics.items()
                                  if not isinstance(v, list)},
        "iter_deltas":           results.iter_deltas,
        "top_1000_scores":       top_scores,
    }
    safe = results.strategy.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"pagerank_{safe}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results -> %s", path)
    return path