
# Entry point for Milestone 1: Parallel Web Crawling and Graph Construction.

# Pipeline overview
# Phase 0  →  (Optional) parse CLI arguments and override config defaults.
# Phase 1  →  Query CommonCrawl CDX API to get WARC record metadata.
# Phase 2  →  Initialise Ray and run parallel WARC fetching + link extraction.
# Phase 3  →  Pull completed adjacency list from the shared Ray Actor.
# Phase 4  →  Save graph to disk (JSON, CSV, pickle) and print a summary.
# Phase 5  →  Shut down Ray cleanly.

# ways to call
    # Run with defaults (500 records, en.wikipedia.org, 4 workers)
    # python main.py

    # Custom settings
    # python main.py --domain en.wikipedia.org --max-records 2000 --num-workers 8

    # Skip live network calls (for development / CI) using a mock
    # python main.py --dry-run


import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import ray

from config import CrawlConfig
from commoncrawl_fetcher import fetch_cdx_records
from ray_workers import run_parallel_crawl
from graph_builder import save_graph, build_networkx_graph, print_summary


#  Logging setup                                                                

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")



#  CLI argument parsing                                                         

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Milestone 1: Parallel Web Crawling using CommonCrawl + Ray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        default="en.wikipedia.org",
        help="Target domain to sample from CommonCrawl.",
    )
    parser.add_argument(
        "--crawl-index",
        default="CC-MAIN-2024-10",
        help="CommonCrawl snapshot identifier (see index.commoncrawl.org).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="Maximum CDX records (= pages) to fetch from CommonCrawl.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="CDX records per Ray task (controls task granularity).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Degree of parallelism (Ray worker slots).",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=150,
        help="Maximum outgoing links to extract per page.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for saving graph files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip network calls; use a small synthetic graph for testing.",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help=(
            "Address of an existing Ray cluster (e.g. 'ray://head-node:10001'). "
            "Leave blank to start a local Ray instance."
        ),
    )
    return parser.parse_args()



#  Dry-run / mock mode (for local testing without network access)               

def mock_cdx_records(n: int = 20) -> list:
    """
    Return a list of fake CDX records for development/testing.
    When --dry-run is passed the real CommonCrawl calls are skipped and
    this synthetic data is used instead so you can test the Ray pipeline
    and graph-saving code offline.
    """
    records = []
    for i in range(n):
        records.append({
            "url":      f"https://en.wikipedia.org/wiki/Page_{i}",
            "filename": "FAKE",   # won't be fetched in dry-run
            "offset":   0,
            "length":   0,
            "status":   "200",
            "mime":     "text/html",
        })
    return records


@ray.remote
def mock_process_batch(batch, actor, config_dict):
    """
    Dry-run version of process_batch.
    Instead of downloading WARC files, we synthesise a random graph.
    """
    import random
    from config import CrawlConfig

    config = CrawlConfig(**config_dict)
    stats = {"fetched": 0, "failed": 0, "links": 0}
    all_urls = [f"https://en.wikipedia.org/wiki/Page_{i}" for i in range(100)]

    for record in batch:
        source = record["url"]
        # Randomly pick 5-15 links to simulate the graph
        k = random.randint(5, min(15, config.max_links_per_page))
        targets = random.sample(all_urls, k)
        actor.add_edges.remote(source, targets)
        stats["fetched"] += 1
        stats["links"] += k

    return stats



#  Main pipeline                                                                

def main() -> None:
    args = parse_args()

    
    # Build config from CLI args                                          
    config = CrawlConfig(
        crawl_index        = args.crawl_index,
        target_domain      = args.domain,
        max_records        = args.max_records,
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
        max_links_per_page = args.max_links,
        restrict_to_domain = args.domain,
        output_dir         = args.output_dir,
    )

    logger.info("=" * 60)
    logger.info("Milestone 1 — Parallel Web Crawling & Graph Construction")
    logger.info("=" * 60)
    logger.info("Config: %s", dataclasses.asdict(config))

    
    # Phase 1 — Fetch CDX index records from CommonCrawl                 
    if args.dry_run:
        logger.info("[DRY RUN] Using synthetic CDX records — no network calls.")
        cdx_records = mock_cdx_records(args.max_records)
    else:
        logger.info("Phase 1: Querying CommonCrawl CDX API …")
        t0 = time.perf_counter()
        cdx_records = fetch_cdx_records(config)
        logger.info(
            "Phase 1 done in %.1fs | %d records retrieved.",
            time.perf_counter() - t0, len(cdx_records),
        )

    if not cdx_records:
        logger.error(
            "No CDX records found for domain '%s' in index '%s'. "
            "Try a different --crawl-index value.",
            config.target_domain, config.crawl_index,
        )
        sys.exit(1)

    
    # Phase 2 — Initialise Ray and run the parallel crawl                
    logger.info("Phase 2: Initialising Ray …")
    ray_init_kwargs = {"num_cpus": args.num_workers}
    if args.ray_address:
        # Connect to an existing distributed cluster
        ray_init_kwargs["address"] = args.ray_address
        logger.info("Connecting to Ray cluster at %s", args.ray_address)
    else:
        logger.info("Starting local Ray instance with %d CPUs.", args.num_workers)

    ray.init(**ray_init_kwargs, ignore_reinit_error=True)

    logger.info("Phase 2: Dispatching parallel Ray tasks …")
    t0 = time.perf_counter()

    if args.dry_run:
        # Use the mock worker so we can test offline
        from ray_workers import GraphActor, _chunk
        import ray as _ray

        actor = GraphActor.remote()
        config_dict = dataclasses.asdict(config)
        batches  = _chunk(cdx_records, config.batch_size)
        futures  = [mock_process_batch.remote(b, actor, config_dict) for b in batches]
        _ray.get(futures)
    else:
        actor = run_parallel_crawl(cdx_records, config)

    elapsed = time.perf_counter() - t0
    logger.info("Phase 2 done in %.1fs.", elapsed)

   
    # Phase 3 — Pull graph from Actor (single network call to Actor)     
    logger.info("Phase 3: Retrieving graph from Ray Actor …")
    adjacency = ray.get(actor.get_graph.remote())


    # Phase 4 — Save and summarise                                       
    logger.info("Phase 4: Saving graph to '%s' …", config.output_dir)
    paths = save_graph(adjacency, config.output_dir, config.graph_filename)

    logger.info("Saved files:")
    for fmt, path in paths.items():
        size_kb = Path(path).stat().st_size / 1024
        logger.info("  %-8s → %s  (%.1f KB)", fmt, path, size_kb)

    G = build_networkx_graph(adjacency)
    print_summary(adjacency, G)

 
    # Phase 5 — Shutdown Ray                                              
    ray.shutdown()
    logger.info("Ray shut down. Milestone 1 complete.")


if __name__ == "__main__":
    main()
