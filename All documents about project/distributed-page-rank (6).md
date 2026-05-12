# Distributed Web Crawling and Parallel PageRank Computation

## Project Overview

This project focuses on designing and evaluating a parallel and distributed system for large-scale graph processing. Students will implement a distributed web crawler to construct a directed graph of web pages and apply a parallel PageRank algorithm to compute node importance scores. The emphasis of the project is on **parallel execution models, data partitioning strategies, synchronization, and performance analysis**, rather than on web engineering or visualization polish.

The system will be implemented using Ray as a task-based parallel runtime and will execute within a shared-memory or single-cluster environment.

---

## Milestone 1: Parallel Web Crawling and Graph Construction

This milestone establishes a parallel data ingestion pipeline that builds the web graph required for PageRank computation.

### Objectives

* Expose students to task parallelism and shared-state coordination
* Generate a realistic but bounded graph workload for downstream computation

### Key Tasks

* Design a parallel crawling strategy where URL fetch tasks are executed concurrently using Ray workers.
* Implement link extraction to construct a directed graph where nodes represent pages and edges represent hyperlinks.
* Ensure duplicate URLs are filtered using a shared or distributed data structure.
* Limit crawl scope using depth or domain constraints to ensure deterministic and reproducible workloads.
* Store the resulting graph in a format suitable for parallel PageRank computation (e.g., adjacency lists or edge lists).

**PDC focus:** task parallelism, shared-state coordination, workload distribution.

---

## Milestone 2: Parallel PageRank Computation

This milestone is the core computational component of the project. Students will implement and analyze a parallel PageRank algorithm operating over the crawled web graph.

### Objectives

* Apply data parallelism to an iterative graph algorithm
* Study synchronization and communication costs
* Evaluate scalability and convergence behavior

### Key Tasks

* Implement the PageRank algorithm using iterative relaxation.
* Partition the graph across workers (e.g., by node ranges or subgraphs).
* Execute PageRank iterations in parallel, synchronizing rank updates at iteration boundaries.
* Compare different execution strategies:

  * Centralized aggregation vs distributed reduction
  * Fixed iteration count vs convergence-based termination
* Measure performance metrics including execution time, speedup, and communication overhead.

**PDC focus:** data partitioning, synchronization barriers, iterative parallel algorithms.

---

## Milestone 3: Incremental Updates and System Evaluation

This milestone extends the system to handle incremental graph growth and emphasizes system-level evaluation.

### Objectives

* Analyze how dynamic data affects parallel algorithms
* Evaluate system behavior under changing workloads

### Key Tasks

* Extend the crawler to introduce new pages after initial PageRank convergence.
* Update PageRank scores incrementally or through partial recomputation.
* Analyze trade-offs between recomputation cost and convergence accuracy.
* Instrument the system to collect runtime statistics such as task duration, communication volume, and memory usage.
* Produce a structured evaluation comparing:

  * Sequential vs parallel PageRank
  * Different partitioning strategies
  * Static vs incremental computation

**PDC focus:** dynamic workloads, recomputation strategies, performance analysis.

---

## Expected Outcomes

By completing this project, students will demonstrate:

* Practical understanding of task-based parallelism
* Ability to parallelize iterative graph algorithms
* Insight into synchronization and communication trade-offs
* Competence in evaluating and reasoning about parallel system performance