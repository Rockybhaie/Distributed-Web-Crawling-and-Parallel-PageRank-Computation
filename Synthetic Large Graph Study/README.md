# Synthetic Large-Graph Scaling Study

Companion appendix to the project: re-runs Milestone 2 and the three
Milestone 3 evaluation groups on a **2,000,000-node synthetic web graph**
(plus a smaller 800K-to-1M chain for Group C) to demonstrate the parallel
speedup that the Milestone 1 / 37K-node CommonCrawl graph was too small
to expose.

**Nothing in `Milestone 1/`, `Milestone 2/`, or `Milestone 3/` is
modified.** This folder only *imports* from them.

## Layout

| File | Purpose |
|---|---|
| `synthetic_generator.py` | Clustered Barabasi-Albert generator + delta computation. |
| `generate_graphs.py` | Materialises `graph_big.pkl` (2M) and `groupc_v{0,1,2}.pkl`. |
| `common.py` | Shared `sys.path` wiring, `M3Config` / `PageRankConfig` factories. |
| `run_m2.py` | Milestone 2 strategies (centralised, distributed) at w=4 and w=8. |
| `run_m3_a.py` | Group A: sequential vs parallel range w=1/2/4/8. |
| `run_m3_b.py` | Group B: range / hash / edge_balanced at w=8. |
| `run_m3_c.py` | Group C: full / warm / localised on v0->v1->v2. |
| `make_plots.py` | Generates report figures from `output/*/summary.json`. |
| `report/SyntheticStudy.tex` | LaTeX source (compile this) - all tables and figures inlined. |
| `snapshots/` | Generated graph snapshots (`.pkl`) and delta JSONs. |
| `output/` | Per-run metrics + summaries. |

## Reproduce

```powershell
cd "Synthetic Large Graph Study"
..\venv\Scripts\python.exe generate_graphs.py   # ~2-3 min
..\venv\Scripts\python.exe run_m3_a.py          # ~15 min (headline)
..\venv\Scripts\python.exe run_m3_b.py          # ~10 min
..\venv\Scripts\python.exe run_m3_c.py          # ~15 min
..\venv\Scripts\python.exe run_m2.py            # ~15 min
..\venv\Scripts\python.exe make_plots.py
# Then compile report\SyntheticStudy.tex with Overleaf or local pdflatex.
```

The drivers are idempotent: snapshots that already exist are reused.

## Dependencies

Uses the same `venv` as the rest of the project. No new pip installs
required; everything is already in `Milestone 3/requirements_m3.txt`
(networkx, ray, numpy, psutil, matplotlib).
