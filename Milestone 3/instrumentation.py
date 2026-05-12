
# instrumentation.py: Unified runtime statistics collector for Milestone 3.
#
# Every parallel/sequential PageRank run records metrics through this module so
# that all evaluation outputs share a common schema. The schema is what the
# Milestone 3 evaluation report ultimately consumes.
#
# Captured metrics (per run):
#   - Per-iteration wall-clock time
#   - Per-task duration (min / max / avg / std) for load imbalance
#   - Synchronisation barrier wait time per iteration
#   - Communication volume per iteration (bytes serialised through Ray)
#   - Per-worker peak resident memory (RSS, MB) via psutil
#   - Driver-side peak memory (MB)
#   - Convergence delta per iteration
#   - Number of "dirty" / affected nodes per iteration (for incremental runs)
#
# Outputs (per run subfolder):
#   metrics.json         - full structured dump
#   metrics_summary.csv  - one-row summary, easy to aggregate
#   iteration_log.csv    - per-iteration timeseries


import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)



# Communication-volume helper.

# We approximate Ray inter-process traffic by summing the number of bytes a
# given Python object would occupy when pickled. This matches what Ray's object
# store actually serialises and is good enough for cross-strategy comparisons.
def estimate_serialised_bytes(obj: Any) -> int:
    try:
        import pickle
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0



# Memory snapshot helper.

def driver_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0



# Per-iteration record.

@dataclass
class IterationRecord:
    iteration: int
    wall_time_s: float
    barrier_wait_s: float
    compute_time_s: float
    delta: float
    comm_bytes: int
    driver_rss_mb: float
    worker_task_times_s: List[float] = field(default_factory=list)
    worker_peak_rss_mb: List[float] = field(default_factory=list)
    dirty_node_count: Optional[int] = None  # None for non-incremental runs

    @property
    def task_min_s(self) -> float:
        return min(self.worker_task_times_s) if self.worker_task_times_s else 0.0

    @property
    def task_max_s(self) -> float:
        return max(self.worker_task_times_s) if self.worker_task_times_s else 0.0

    @property
    def task_avg_s(self) -> float:
        if not self.worker_task_times_s:
            return 0.0
        return sum(self.worker_task_times_s) / len(self.worker_task_times_s)

    @property
    def task_std_s(self) -> float:
        if len(self.worker_task_times_s) < 2:
            return 0.0
        return statistics.stdev(self.worker_task_times_s)

    @property
    def load_imbalance(self) -> float:
        # (max - min) / max, in [0, 1]; 0 = perfectly balanced.
        if self.task_max_s <= 0:
            return 0.0
        return (self.task_max_s - self.task_min_s) / self.task_max_s



# Whole-run aggregator.

class RuntimeStats:

    def __init__(
        self,
        run_name: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_name: str = run_name
        self.run_metadata: Dict[str, Any] = dict(run_metadata or {})
        self.iterations: List[IterationRecord] = []
        self.t_start: float = time.perf_counter()
        self.t_end: Optional[float] = None
        self.driver_rss_baseline_mb: float = driver_rss_mb()
        self.driver_rss_peak_mb: float = self.driver_rss_baseline_mb
        # Free-form extras (e.g. accuracy_max_diff for incremental runs).
        self.extras: Dict[str, Any] = {}

    
    # Iteration recording
    
    def record_iteration(
        self,
        iteration: int,
        wall_time_s: float,
        barrier_wait_s: float,
        compute_time_s: float,
        delta: float,
        comm_bytes: int,
        worker_task_times_s: Optional[List[float]] = None,
        worker_peak_rss_mb: Optional[List[float]] = None,
        dirty_node_count: Optional[int] = None,
    ) -> None:
        rss = driver_rss_mb()
        if rss > self.driver_rss_peak_mb:
            self.driver_rss_peak_mb = rss
        rec = IterationRecord(
            iteration=iteration,
            wall_time_s=wall_time_s,
            barrier_wait_s=barrier_wait_s,
            compute_time_s=compute_time_s,
            delta=delta,
            comm_bytes=comm_bytes,
            driver_rss_mb=rss,
            worker_task_times_s=list(worker_task_times_s or []),
            worker_peak_rss_mb=list(worker_peak_rss_mb or []),
            dirty_node_count=dirty_node_count,
        )
        self.iterations.append(rec)

    def finish(self) -> None:
        self.t_end = time.perf_counter()

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value

    
    # Derived aggregates
    
    @property
    def total_wall_time_s(self) -> float:
        end = self.t_end if self.t_end is not None else time.perf_counter()
        return end - self.t_start

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def total_comm_bytes(self) -> int:
        return sum(r.comm_bytes for r in self.iterations)

    @property
    def total_barrier_wait_s(self) -> float:
        return sum(r.barrier_wait_s for r in self.iterations)

    @property
    def total_compute_time_s(self) -> float:
        return sum(r.compute_time_s for r in self.iterations)

    @property
    def avg_iteration_time_s(self) -> float:
        n = self.num_iterations
        return (sum(r.wall_time_s for r in self.iterations) / n) if n > 0 else 0.0

    @property
    def avg_load_imbalance(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(r.load_imbalance for r in self.iterations) / len(self.iterations)

    @property
    def peak_worker_rss_mb(self) -> float:
        peak = 0.0
        for r in self.iterations:
            if r.worker_peak_rss_mb:
                m = max(r.worker_peak_rss_mb)
                if m > peak:
                    peak = m
        return peak

    
    # Summary dict (used as the canonical metrics format)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "metadata": self.run_metadata,
            "total_wall_time_s": self.total_wall_time_s,
            "num_iterations": self.num_iterations,
            "avg_iteration_time_s": self.avg_iteration_time_s,
            "total_compute_time_s": self.total_compute_time_s,
            "total_barrier_wait_s": self.total_barrier_wait_s,
            "total_comm_bytes": self.total_comm_bytes,
            "total_comm_kb": self.total_comm_bytes / 1024.0,
            "avg_load_imbalance": self.avg_load_imbalance,
            "driver_rss_baseline_mb": self.driver_rss_baseline_mb,
            "driver_rss_peak_mb": self.driver_rss_peak_mb,
            "peak_worker_rss_mb": self.peak_worker_rss_mb,
            "final_delta": self.iterations[-1].delta if self.iterations else None,
            "extras": self.extras,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        return {
            **self.to_summary_dict(),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "wall_time_s": r.wall_time_s,
                    "barrier_wait_s": r.barrier_wait_s,
                    "compute_time_s": r.compute_time_s,
                    "delta": r.delta,
                    "comm_bytes": r.comm_bytes,
                    "driver_rss_mb": r.driver_rss_mb,
                    "worker_task_times_s": r.worker_task_times_s,
                    "worker_peak_rss_mb": r.worker_peak_rss_mb,
                    "task_min_s": r.task_min_s,
                    "task_max_s": r.task_max_s,
                    "task_avg_s": r.task_avg_s,
                    "task_std_s": r.task_std_s,
                    "load_imbalance": r.load_imbalance,
                    "dirty_node_count": r.dirty_node_count,
                }
                for r in self.iterations
            ],
        }

   
    # Persistence
    
    def save(self, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # 1. Full structured dump
        json_path = os.path.join(output_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_full_dict(), f, indent=2, default=str)
        paths["metrics_json"] = json_path

        # 2. One-row summary (for cross-run aggregation)
        summary = self.to_summary_dict()
        flat = _flatten_dict(summary)
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(flat.keys()))
            writer.writerow([_csv_safe(v) for v in flat.values()])
        paths["metrics_summary_csv"] = csv_path

        # 3. Per-iteration timeseries
        log_path = os.path.join(output_dir, "iteration_log.csv")
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "wall_time_s", "barrier_wait_s",
                "compute_time_s", "delta", "comm_bytes", "driver_rss_mb",
                "task_min_s", "task_max_s", "task_avg_s", "task_std_s",
                "load_imbalance", "dirty_node_count",
            ])
            for r in self.iterations:
                writer.writerow([
                    r.iteration, r.wall_time_s, r.barrier_wait_s,
                    r.compute_time_s, r.delta, r.comm_bytes, r.driver_rss_mb,
                    r.task_min_s, r.task_max_s, r.task_avg_s, r.task_std_s,
                    r.load_imbalance, r.dirty_node_count,
                ])
        paths["iteration_log_csv"] = log_path

        logger.info("Saved metrics for run '%s' to %s", self.run_name, output_dir)
        return paths



# Lightweight context manager: time a block precisely.

class Stopwatch:
    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start



# Internal helpers

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _csv_safe(v: Any) -> str:
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=str)
    if v is None:
        return ""
    return str(v)
