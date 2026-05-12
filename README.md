# Distributed Web Crawling and Parallel PageRank Computation

**Course:** Parallel and Distributed Computing
**Group members:** Abdullah Irfan (29266), Ameer Hamza (28308), Muhammad Khizer (28991), Rameez Hoda (29057)
**Tech stack:** Python 3.11+, Ray, NetworkX, NumPy, SciPy, BeautifulSoup, CommonCrawl

This project builds an end-to-end parallel system for large-scale graph processing, in three milestones plus a post-hoc scaling study:

| Milestone | Folder | Focus |
|---|---|---|
| 1 | `Milestone 1/` | Parallel web crawling with Ray, building a directed web graph from CommonCrawl. |
| 2 | `Milestone 2/` | Parallel PageRank with two strategies (centralised aggregation / distributed reduction). |
| 3 | `Milestone 3/` | Incremental graph growth + system-wide evaluation (sequential vs parallel, partitioning strategies, static vs incremental). |
| Extra | `Synthetic Large Graph Study/` | Scaling study: every M2 and M3 experiment re-run on a 2M-node synthetic graph (54x larger) to characterise behaviour out of the small-graph Amdahl regime. |

Reports for Milestones 1 and 2 are in `All documents about project/` as
PDFs. The Milestone 3 report is provided as LaTeX source at
`Milestone 3/report/Milestone3_Report.tex` and is compiled to PDF via
Overleaf or a local TeX install (see Step 6 below or
`Milestone 3/report/README.txt`).

---

## Quick Start (the entire project on a fresh machine)

### 1. Clone and set up the environment

```powershell
# From the repo root
python -m venv venv
venv\Scripts\Activate.ps1                    # Windows PowerShell
# (or venv\Scripts\activate.bat / source venv/bin/activate)
```

### 2. Install dependencies (Milestone 3's requirements file is a superset)

```powershell
pip install -r "Milestone 3/requirements_m3.txt"
```

### 3. (Optional) Re-run Milestone 1 to build the seed graph

The `Milestone 1/All config outputs/Config3/web_graph.pkl` file is already shipped in this repo (37,074 nodes / 67,667 edges). If you want to rebuild it from scratch:

```powershell
cd "Milestone 1"
python main.py --max-records 500 --num-workers 4 --output-dir output_config3
```

### 4. (Optional) Re-run Milestone 2's evaluation suite

```powershell
cd "Milestone 2"
python main_pagerank.py --graph output_config3/web_graph.pkl
```

The 14 individual test commands are listed in `Milestone 2/How to run.txt`.

### 5. Run Milestone 3 (incremental updates + full evaluation)

```powershell
cd "Milestone 3"

# (a) Bootstrap v0 + run incremental crawls v1, v2, v3 (one-time setup).
python m3_main.py crawl-incremental --all --workers 4

# (b) Run all three evaluation groups.
python m3_main.py evaluate --group all --workers 4

# (c) Generate every figure as PNG + PDF.
python m3_main.py plot
```

### 6. Compile the Milestone 3 report (Overleaf-friendly)

The LaTeX source is at `Milestone 3/report/Milestone3_Report.tex` with
all figures already copied into `Milestone 3/report/figures/`. Either
upload the `report/` folder as a zip to https://www.overleaf.com/ and
hit Recompile, or run `pdflatex Milestone3_Report.tex` (twice) inside
the `report/` folder if you have a local TeX install. Full instructions
in `Milestone 3/report/README.txt`.

Detailed instructions, per-step explanations, and troubleshooting are in `Milestone 3/How to run.txt`.

---

## Repository Layout

```
Project Final Final Final/
|-- All documents about project/
|   |-- distributed-page-rank (6).md      Project spec
|   |-- Milestone1_Report.pdf
|   `-- Milestone2_Report.pdf
|
|-- Milestone 1/                          Parallel web crawler
|   |-- main.py
|   |-- config.py
|   |-- commoncrawl_fetcher.py            CommonCrawl CDX + WARC retrieval
|   |-- ray_workers.py                    GraphActor + parallel batch task
|   |-- graph_builder.py                  Output serialisation
|   |-- check_output.py
|   |-- requirements.txt
|   |-- All config outputs/               Reference graph snapshots
|   `-- How to run.txt
|
|-- Milestone 2/                          Parallel PageRank
|   |-- main_pagerank.py
|   |-- pagerank_config.py
|   |-- graph_loader.py                   URL -> integer index, partitioning
|   |-- pagerank_sequential.py            Custom + NetworkX baselines
|   |-- pagerank_parallel.py              Centralised + distributed (RankActor)
|   |-- performance_analysis.py           Metrics, Amdahl, JSON export
|   |-- All outputs/                      14-test evaluation results
|   |-- requirements_m2.txt
|   `-- How to run.txt
|
|-- Milestone 3/                          Incremental updates + evaluation
|   |-- m3_main.py                        CLI entry point
|   |-- m3_config.py
|   |-- instrumentation.py                Unified runtime-stats collector
|   |-- partitioning_strategies.py        range / hash / edge_balanced
|   |-- incremental_crawler.py            Versioned snapshot growth
|   |-- incremental_pagerank.py           full / warm / localised + sequential
|   |-- evaluation_runner.py              Groups A / B / C
|   |-- plot_results.py                   All figures
|   |-- snapshots/                        graph_v0..v3 + delta files
|   |-- cache/                            CommonCrawl per-URL cache
|   |-- output/                           Run metrics + figures
|   |-- report/                           LaTeX source + figures for the M3 report
|   |   |-- Milestone3_Report.tex
|   |   |-- figures/
|   |   `-- README.txt                    How to compile the report
|   |-- requirements_m3.txt
|   `-- How to run.txt
|
|-- Synthetic Large Graph Study/          Scaling study on a 2M-node synthetic graph
|   |-- synthetic_generator.py            Clustered Barabasi-Albert generator
|   |-- generate_graphs.py                Builds graph_big (2M) + v0/v1/v2 chain
|   |-- run_m2.py                         Re-runs M2 strategies at 2M nodes
|   |-- run_m3_a.py / run_m3_b.py / run_m3_c.py   Re-runs M3 Groups A / B / C
|   |-- make_plots.py                     Regenerates every figure from metrics
|   |-- report/SyntheticStudy.tex         Companion LaTeX report (5 pages)
|   |-- snapshots/                        (gitignored) 2M-node graph pickles
|   |-- output/                           (gitignored) per-run metrics
|   `-- README.md                         Study-specific instructions
|
|-- venv/                                 (gitignored) Python virtual environment
|-- .gitignore
`-- README.md                             You are here
```

---

## What Each Milestone Adds

### Milestone 1 - Parallel Web Crawling

- Parallel crawl of 500 Wikipedia pages from CommonCrawl using Ray task parallelism.
- A Ray Actor (`GraphActor`) provides shared, race-free graph state across workers.
- Output: directed graph with **37,074 nodes / 67,667 edges** (Config 3, the seed graph used by M2 and M3).
- Four configurations evaluated for worker scaling, batch size, and workload size.

### Milestone 2 - Parallel PageRank

- Custom iterative PageRank + NetworkX reference baseline.
- Two parallel execution strategies:
  - **Centralised aggregation**: workers return partial ranks; driver merges.
  - **Distributed reduction**: workers write to a shared `RankActor`.
- 14-test evaluation grid covering worker scaling, strategy comparison, termination policy, graph-size sensitivity, and damping-factor sensitivity.
- Convergence in 4 iterations on the dangling-heavy Config 3 graph.

### Synthetic Large Graph Study (post-hoc scaling appendix)

The Milestone 1 CommonCrawl graph (37K nodes) is too small for parallel
speedup to emerge -- Ray's per-iteration overhead dominates. This study
re-runs the entire M2 and M3 evaluation unchanged on a synthetic
2{,}000{,}000-node / 7.6M-edge graph produced by a clustered
Barabasi-Albert generator, plus an 800K-to-1M version chain for Group C.

Headline results:
- **Warm-start incremental PageRank: 2.10x speedup** over full recomputation at v1, max abs rank diff 1.4e-9 (essentially exact). *This is the >1x speedup result the system design predicts.*
- Pure parallel scaling ratio improves 3.1x vs M1 (0.30x -> 0.93x), but stays below 1.0x because Ray's broadcast of the 16 MB rank vector dominates at this scale.
- Partitioning strategies now differentiate meaningfully: `hash` and `edge_balanced` drop load imbalance from 85.5% (`range`) to ~23% -- a 3.6x improvement.

See `Synthetic Large Graph Study/README.md` and the LaTeX report at
`Synthetic Large Graph Study/report/SyntheticStudy.tex` for the full
analysis (including a per-iteration decomposition explaining the 0.93x
ceiling).

### Milestone 3 - Incremental Updates and System Evaluation

- **Versioned incremental crawler** that grows the graph in three stages (v1 = +5%, v2 = +10%, v3 = +25%) by pulling additional CommonCrawl CDX slices.
- **Per-URL on-disk cache**: every WARC fetch is cached so re-runs require zero network access.
- **Three partitioning strategies** for parallel PageRank:
  - `range` (M2 baseline) - contiguous node-id ranges.
  - `hash` - `node_id % k`, breaking adjacency locality.
  - `edge_balanced` - greedy LPT packing on `|in_links[u]|`, minimising load imbalance.
- **Three update strategies** for incremental PageRank:
  - `full` - cold restart from `1/N` (ground truth).
  - `warm` - warm-start from previous scores.
  - `localised` - only recompute the dirty set (new nodes + their k-hop predecessors / successors).
- **Unified instrumentation layer** capturing per-iteration wall time, barrier-wait time, communication volume, per-worker peak RSS, driver memory, convergence delta, and dirty-set size.
- **Structured evaluation matrix** with three groups:
  - **Group A**: sequential vs parallel scaling (1 / 2 / 4 workers).
  - **Group B**: partitioning strategy comparison.
  - **Group C**: static vs incremental cost-vs-accuracy trade-off.
- **Reproducible figures**: PNG + PDF for the report, all generated from saved metrics by a single `plot` command.

---

## Reading Order for Graders

1. `README.md` (you are here) - high-level overview.
2. `All documents about project/Milestone1_Report.pdf` - data ingestion design.
3. `All documents about project/Milestone2_Report.pdf` - parallel PageRank design and 14-test eval.
4. `Milestone 3/report/Milestone3_Report.tex` (or compiled PDF) - dynamic workloads + final evaluation.
5. `Synthetic Large Graph Study/report/SyntheticStudy.tex` - scaling study on a 2M-node synthetic graph (54x larger than M1). Demonstrates the **2.10x warm-start speedup** and decomposes the parallel bottleneck at scale.
6. `Milestone 3/How to run.txt` - end-to-end run instructions.

---

## Troubleshooting

- **CommonCrawl unavailable**: the M3 cache layer means experiments are fully reproducible after the first successful crawl. If you are running on a fresh machine and CC is down, wait a few minutes and re-run; only uncached URLs hit the network.
- **Out-of-memory on small machines**: lower `--workers` (e.g. `--workers 2`) - the 37K-node graph is not memory-bound, but each Ray worker process adds overhead.
- **Ray "address already in use"**: `ray stop` (kills any leftover Ray instance) then re-run.

For per-milestone details, see each milestone folder's own `How to run.txt`.
