
# Ray tasks and Actors for parallel graph construction.



import logging
from typing import List, Dict, Optional

import ray

from commoncrawl_fetcher import fetch_warc_record, extract_links
from config import CrawlConfig

logger = logging.getLogger(__name__)



#  Actor — shared mutable state across all Ray workers                         
# Centralised, thread-safe store for the graph being built. 

# Why an Actor instead of a plain dict?
# Ray tasks run in separate processes; they cannot share a Python dict
# directly.  An Actor is the Ray way to provide shared mutable state:
# it lives in one process and all other workers communicate with it
# via asynchronous remote method calls.

# Responsibilities
# * Keep track of which URLs have already been processed (deduplication).
# * Accumulate directed edges  source_url → [target_url, …].
# * Expose statistics for progress reporting.


@ray.remote
class GraphActor:
    

    def __init__(self) -> None:
        # adjacency list: source URL → list of destination URLs
        self._graph: Dict[str, List[str]] = {}
        # set of URLs we have already added as source nodes
        self._visited: set = set()

    
    #  Write operations (called by worker tasks)                          
    def add_edges(self, source: str, targets: List[str]) -> None:
        """
        Register outgoing edges from ``source`` to each URL in ``targets``.
        This is the primary write method called by every Ray worker task
        after it has successfully parsed a page.
        """
        if source not in self._graph:
            self._graph[source] = []
        # Extend (not replace) to allow incremental updates in Milestone 3
        self._graph[source].extend(targets)
        self._visited.add(source)

   
    #  Read operations (called by the driver / main.py)                  
    
    # Return True if this URL is already a source node in the graph.
    def is_visited(self, url: str) -> bool:
        return url in self._visited


    # Return the complete adjacency list (used when saving the graph).
    def get_graph(self) -> Dict[str, List[str]]:
        
        return self._graph
    


    # Return current graph statistics for progress reporting.
    def get_stats(self) -> Dict[str, int]:
        return {
            "nodes":        len(self._graph),
            "edges":        sum(len(v) for v in self._graph.values()),
            "visited_urls": len(self._visited),
        }



#  Remote task — runs on a Ray worker process                                   

@ray.remote
def process_batch(
    batch: List[Dict],
    actor: GraphActor,          # Ray Actor handle — not a local Python object
    config_dict: Dict,
) -> Dict[str, int]:
    """
    Ray task: fetch and parse a batch of WARC records, then write edges to
    the shared GraphActor.
    Parameters
    batch : list[dict]
        A slice of CDX records (each describing one crawled page).
    actor : GraphActor
        Handle to the shared Ray Actor.  Method calls on this handle are
        sent as remote messages to the Actor process.
    config_dict : dict
        CrawlConfig serialised as a plain dict so Ray can serialise it
        without pickling issues.

    Returns
    dict
        Per-batch statistics (successful fetches, links found, …).
    """
    # Re-inflate the config inside the worker process
    config = CrawlConfig(**config_dict)

    stats = {"fetched": 0, "failed": 0, "links": 0}

    for cdx_record in batch:
        source_url = cdx_record.get("url", "")
        if not source_url:
            stats["failed"] += 1
            continue

        
        # Step 1: Download the specific WARC byte range for this page      
        
        html = fetch_warc_record(cdx_record, config)
        if html is None:
            stats["failed"] += 1
            continue

        
        # Step 2: Extract all outgoing hyperlinks from the HTML            
        targets = extract_links(source_url, html, config)
        stats["fetched"] += 1
        stats["links"]   += len(targets)

        
        # Step 3: Write edges to the shared Actor                          
        # (this remote call is non-blocking from the task's perspective;   
        #  Ray queues it internally)                                        
        
        if targets:
            # .remote() sends the call asynchronously to the Actor process
            actor.add_edges.remote(source_url, targets)

    return stats



#  Orchestration helper — called by main.py                                    

def run_parallel_crawl(
    cdx_records: List[Dict],
    config: CrawlConfig,
) -> GraphActor:
    """
    Orchestrate the full parallel crawl using Ray.
    Steps
    1.  Create one shared GraphActor (lives for the duration of the crawl).
    2.  Partition CDX records into batches.
    3.  Dispatch one Ray *task* per batch (tasks run concurrently).
    4.  Wait (synchronisation barrier) until all tasks have finished.
    5.  Return the Actor handle so main.py can extract the graph.

    Parameters
    cdx_records : list[dict]
        Records from the CommonCrawl CDX API.
    config : CrawlConfig
        Project-wide configuration.

    Returns
    GraphActor
        The Actor handle containing the completed graph.
    """
    logger.info(
        "Starting parallel crawl | records=%d  batch_size=%d  workers=%d",
        len(cdx_records), config.batch_size, config.num_workers,
    )

    
    # 1. Create the shared Actor (one per crawl session)                  
    actor = GraphActor.remote()

   
    # 2. Partition CDX records into equally-sized batches                 
    batches = _chunk(cdx_records, config.batch_size)
    logger.info("Partitioned records into %d batches.", len(batches))

    # Serialise config to a plain dict so Ray can send it to remote tasks
    config_dict = {k: v for k, v in config.__dict__.items()}

    
    # 3. Dispatch one Ray task per batch                                  
    #    .remote() returns a Future immediately; the actual work happens  
    #    concurrently on Ray's worker processes.                           
     
    futures = []
    for i, batch in enumerate(batches):
        future = process_batch.remote(batch, actor, config_dict)
        futures.append(future)
        logger.debug("Dispatched task for batch %d/%d", i + 1, len(batches))

        # Throttle dispatch to avoid overwhelming the Actor with a flood
        # of simultaneous add_edges calls at startup
        if len(futures) - i > config.num_workers * 3:
            # Let some tasks finish before dispatching more
            done, futures = ray.wait(futures, num_returns=len(futures) // 2)
            _log_batch_stats(ray.get(done))

    
    # 4. Synchronisation barrier — wait for ALL tasks to complete         
    all_stats = ray.get(futures)
    _log_batch_stats(all_stats)

    
    # 5. Report final graph stats from the Actor                          
    final_stats = ray.get(actor.get_stats.remote())
    logger.info(
        "Crawl complete | nodes=%d  edges=%d  visited=%d",
        final_stats["nodes"],
        final_stats["edges"],
        final_stats["visited_urls"],
    )

    return actor



#  Private helpers                                                              

def _chunk(lst: List, size: int) -> List[List]:
    """Split a list into consecutive chunks of at most *size* elements."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _log_batch_stats(stats_list: List[Dict]) -> None:
    """Aggregate and log stats returned by a group of completed tasks."""
    totals = {"fetched": 0, "failed": 0, "links": 0}
    for s in stats_list:
        for k in totals:
            totals[k] += s.get(k, 0)
    logger.info(
        "Batch group done | fetched=%d  failed=%d  links_found=%d",
        totals["fetched"], totals["failed"], totals["links"],
    )
