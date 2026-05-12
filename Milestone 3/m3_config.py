
# m3_config.py : Central configuration for Milestone 3:
#                Incremental Updates and System Evaluation.
#
# Extends the design pattern of Milestone 1's CrawlConfig and Milestone 2's
# PageRankConfig. Every tunable parameter for incremental crawling, partitioning
# strategies, incremental PageRank, and the evaluation runner lives here so
# experiments are fully reproducible from a single config object.


from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class M3Config:

   
    # Snapshot / incremental-crawl settings
    

    # Path to the initial graph (Milestone 1 Config 3 output).
    # This is the "v0" baseline graph that Milestone 3 grows from.
    initial_graph_path: str = "../Milestone 1/All config outputs/Config3/web_graph.pkl"

    # Where versioned snapshots are written:
    #   snapshots/graph_v0.pkl, graph_v1.pkl, ...
    #   snapshots/delta_v0_v1.json (which nodes/edges are new)
    snapshot_dir: str = "snapshots"

    # Where every successful CommonCrawl fetch is cached. After the first
    # successful crawl, every subsequent run reads from cache (no network).
    cache_dir: str = "cache"

    # CommonCrawl settings (mirrors Milestone 1 CrawlConfig).
    crawl_index: str = "CC-MAIN-2024-10"
    target_domain: str = "en.wikipedia.org"
    restrict_to_domain: Optional[str] = "en.wikipedia.org"
    max_links_per_page: int = 150
    request_timeout: int = 30
    max_retries: int = 5
    user_agent: str = (
        "PDC-Course-Crawler-M3/1.0 "
        "(academic project; uses CommonCrawl data only - no live crawling)"
    )

    # Incremental crawl plan: each entry is (start_offset, count).
    # Records 0..500 are already in the v0 graph (Milestone 1 Config 3).
    # We append by pulling the next slice from the same CC index.
    growth_plan: List[tuple] = field(default_factory=lambda: [
        (500, 100),   # v0 -> v1: +100 records (~5% growth)
        (600, 200),   # v1 -> v2: +200 records (~10% growth)
        (800, 500),   # v2 -> v3: +500 records (~25% growth)
    ])

    # Ray task batch size during incremental crawl (same role as M1).
    crawl_batch_size: int = 25

    
    # PageRank algorithm parameters
  

    damping_factor: float = 0.85
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_convergence: bool = True


    # Parallelism settings


    num_workers: int = 4
    num_partitions: int = 4

    
    # Partitioning strategy
   

    # One of: "range", "hash", "edge_balanced".
    partition_strategy: str = "range"

    
    # Incremental PageRank update strategy
    

    # One of:
    #   "full"      - cold restart from 1/N (M2 baseline)
    #   "warm"      - warm-start from previous ranks; full iterations
    #   "localised" - only update affected nodes + their k-hop predecessors
    update_strategy: str = "full"

    # k-hop predecessor radius for localised updates.
    affected_hop_radius: int = 2

    
    # Execution strategy (carried over from Milestone 2)
    

    # "centralized" or "distributed".
    execution_strategy: str = "centralized"

   
    # Output settings
    

    output_dir: str = "output"
    save_scores: bool = True
    save_metrics: bool = True
