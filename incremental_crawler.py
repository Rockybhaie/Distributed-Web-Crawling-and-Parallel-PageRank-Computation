
# incremental_crawler.py: Versioned incremental crawler for Milestone 3.
#
# Extends the Milestone 1 crawler so a previously-converged graph (M1 Config 3,
# 37,074 nodes / 67,667 edges) can be GROWN by appending newly-crawled
# CommonCrawl pages, producing versioned snapshots:
#
#     graph_v0.pkl     <- starting graph (loaded from Milestone 1 output)
#     graph_v1.pkl     <- after first incremental crawl (+~5% growth)
#     graph_v2.pkl     <- after second                  (+~10%)
#     ...
#
#     delta_v0_v1.json <- {"new_source_nodes": [...],
#                          "new_target_nodes": [...],
#                          "new_edges": [[src, dst], ...],
#                          "affected_existing_sources": [...]}
#
# The delta files are inputs to incremental_pagerank.py - they tell the update
# strategies exactly which nodes/edges changed.
#
# How this differs from the Milestone 1 crawler:
#   1. The GraphActor is SEEDED with the existing v(K-1) adjacency list.
#   2. CDX records come from a *slice* of the same crawl_index so the incremental
#      crawl reaches new pages that were not in Config 3.
#   3. Every successful WARC fetch + extracted-link list is cached to
#      cache/<sha1(url)>.json. Re-runs read the cache (no network).


import dataclasses
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ray

from m3_config import M3Config


# Bring in the Milestone 1 modules. We do NOT duplicate any code from M1.
#
# We extend BOTH sys.path (for the driver) AND the PYTHONPATH environment
# variable (so spawned Ray worker processes can import config.py and
# commoncrawl_fetcher.py too).

_M1_DIR = (Path(__file__).resolve().parent.parent / "Milestone 1")
_M3_DIR = Path(__file__).resolve().parent
for _dir in (_M1_DIR, _M3_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

_existing_pp = os.environ.get("PYTHONPATH", "")
_pp_parts = [p for p in (str(_M1_DIR), str(_M3_DIR), _existing_pp) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(_pp_parts)

from config import CrawlConfig                                       # noqa: E402
from commoncrawl_fetcher import (                                    # noqa: E402
    fetch_cdx_records,
    fetch_warc_record,
    extract_links,
)
from ray_workers import GraphActor                                   # noqa: E402

logger = logging.getLogger(__name__)



# Cache layer

def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cache_path(cache_dir: str, url: str) -> str:
    return os.path.join(cache_dir, f"{_cache_key(url)}.json")


def _read_cache(cache_dir: str, url: str) -> Optional[List[str]]:
    path = _cache_path(cache_dir, url)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("links", []))
    except Exception:
        return None


def _write_cache(cache_dir: str, url: str, links: List[str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    tmp = _cache_path(cache_dir, url) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"url": url, "links": list(links)}, f, ensure_ascii=False)
    os.replace(tmp, _cache_path(cache_dir, url))



# CDX retrieval with overall-cache (so we don't re-query the index every run).

def _cdx_index_cache_path(cache_dir: str, crawl_index: str, domain: str, total: int) -> str:
    key = hashlib.sha1(f"{crawl_index}|{domain}|{total}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"cdx_index_{key}.json")


def fetch_cdx_records_total(
    cfg: M3Config,
    total: int,
) -> List[Dict[str, Any]]:
    """
    Fetch up to `total` CDX records (status=200, html) for the configured
    domain and crawl index. Caches the full list to disk so repeated runs
    do not re-hit the CDX API.
    """
    cache_path = _cdx_index_cache_path(cfg.cache_dir, cfg.crawl_index,
                                       cfg.target_domain, total)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            logger.info("CDX index loaded from cache (%d records).", len(records))
            return records
        except Exception:
            logger.warning("CDX cache at %s unreadable - re-fetching.", cache_path)

    # Wrap M1's fetcher with a CrawlConfig that requests up to `total` records.
    m1_cfg = CrawlConfig(
        crawl_index=cfg.crawl_index,
        target_domain=cfg.target_domain,
        max_records=total,
        max_links_per_page=cfg.max_links_per_page,
        restrict_to_domain=cfg.restrict_to_domain,
        request_timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        user_agent=cfg.user_agent,
    )
    logger.info("Querying CommonCrawl CDX API | total=%d", total)
    records = fetch_cdx_records(m1_cfg)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    logger.info("Cached %d CDX records to %s", len(records), cache_path)
    return records



# Cache-aware Ray batch task

@ray.remote
def cached_process_batch(
    batch: List[Dict[str, Any]],
    actor: GraphActor,
    m1_config_dict: Dict[str, Any],
    cache_dir: str,
    skip_existing_sources: List[str],
) -> Dict[str, int]:
    """
    Like Milestone 1's process_batch, but:
      - Reads the per-URL link-list cache before any HTTP fetch.
      - Skips URLs that are already source nodes in the seeded graph.
      - Writes successful fetches back to the cache.

    The actor is the *same* GraphActor type defined in Milestone 1; it has
    already been seeded with the previous snapshot's adjacency.
    """
    m1_cfg = CrawlConfig(**m1_config_dict)
    skip = set(skip_existing_sources or [])

    stats = {"fetched": 0, "failed": 0, "links": 0,
             "cache_hits": 0, "cache_misses": 0, "skipped": 0}

    for record in batch:
        url = record.get("url", "")
        if not url:
            stats["failed"] += 1
            continue
        if url in skip:
            stats["skipped"] += 1
            continue

        cached = _read_cache(cache_dir, url)
        if cached is not None:
            stats["cache_hits"] += 1
            if cached:
                actor.add_edges.remote(url, cached)
                stats["links"] += len(cached)
            stats["fetched"] += 1
            continue

        stats["cache_misses"] += 1
        html = fetch_warc_record(record, m1_cfg)
        if html is None:
            stats["failed"] += 1
            continue

        targets = extract_links(url, html, m1_cfg)
        _write_cache(cache_dir, url, targets)

        if targets:
            actor.add_edges.remote(url, targets)
        stats["fetched"] += 1
        stats["links"] += len(targets)

    return stats



# Snapshot helpers (load / save)

def load_snapshot_adjacency(path: str) -> Dict[str, List[str]]:
    """
    Load any of the formats Milestone 1 saves and return an adjacency dict.
    Supports .pkl (NetworkX DiGraph) and .json (already an adjacency list).
    """
    if path.endswith(".pkl"):
        import networkx as nx  # local import to keep top imports light
        with open(path, "rb") as f:
            G = pickle.load(f)
        adj: Dict[str, List[str]] = {}
        for src in G.nodes():
            outs = list(G.successors(src))
            if outs:
                adj[src] = outs
        return adj
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported snapshot extension: {path}")


def save_snapshot(
    adjacency: Dict[str, List[str]],
    snapshot_dir: str,
    version: int,
) -> Dict[str, str]:
    """
    Save a snapshot in the same three formats as Milestone 1 (pkl, json, csv)
    so M2/M3 PageRank can load it interchangeably with M1's outputs.
    """
    import csv
    import networkx as nx
    os.makedirs(snapshot_dir, exist_ok=True)
    base = os.path.join(snapshot_dir, f"graph_v{version}")

    # 1. Build NetworkX graph
    G = nx.DiGraph()
    for src, dsts in adjacency.items():
        G.add_node(src)
        for dst in dsts:
            G.add_edge(src, dst)

    # 2. Pickle
    pkl_path = base + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. JSON adjacency
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(adjacency, f, ensure_ascii=False)

    # 4. CSV edges
    csv_path = base + "_edges.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for src, dsts in adjacency.items():
            for dst in dsts:
                w.writerow([src, dst])

    logger.info(
        "Saved snapshot v%d | nodes=%d edges=%d -> %s{.pkl,.json,_edges.csv}",
        version, G.number_of_nodes(), G.number_of_edges(), base,
    )
    return {"pickle": pkl_path, "json": json_path, "csv": csv_path}



# Delta computation

def compute_delta(
    prev_adj: Dict[str, List[str]],
    new_adj: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Compute which nodes / edges are new in `new_adj` vs `prev_adj`.

    Returns:
      new_source_nodes:   sources present in new_adj but not in prev_adj
      new_target_nodes:   target URLs that appear in new_adj's adjacency lists
                          but were not present in prev_adj as either source
                          or target
      new_edges:          edges in new_adj that were not in prev_adj
      affected_existing_sources: existing sources whose out-link list grew
                                 (these need PageRank rescore)
    """
    prev_sources: Set[str] = set(prev_adj.keys())
    prev_targets: Set[str] = set()
    prev_edges: Set[Tuple[str, str]] = set()
    for s, ts in prev_adj.items():
        for t in ts:
            prev_targets.add(t)
            prev_edges.add((s, t))
    prev_all_nodes = prev_sources | prev_targets

    new_sources: Set[str] = set(new_adj.keys())
    new_targets: Set[str] = set()
    new_edges_set: Set[Tuple[str, str]] = set()
    for s, ts in new_adj.items():
        for t in ts:
            new_targets.add(t)
            new_edges_set.add((s, t))
    new_all_nodes = new_sources | new_targets

    new_source_nodes = sorted(new_sources - prev_sources)
    new_target_nodes = sorted(new_all_nodes - prev_all_nodes - set(new_source_nodes))
    new_edges_list = sorted(new_edges_set - prev_edges)

    affected_existing: Set[str] = set()
    for s, t in new_edges_list:
        if s in prev_sources:
            affected_existing.add(s)

    return {
        "new_source_nodes": new_source_nodes,
        "new_target_nodes": new_target_nodes,
        "new_edges": [list(e) for e in new_edges_list],
        "affected_existing_sources": sorted(affected_existing),
        "summary": {
            "prev_node_count": len(prev_all_nodes),
            "new_node_count": len(new_all_nodes),
            "added_nodes": len(new_all_nodes) - len(prev_all_nodes),
            "prev_edge_count": len(prev_edges),
            "new_edge_count": len(new_edges_set),
            "added_edges": len(new_edges_set) - len(prev_edges),
        },
    }


def save_delta(delta: Dict[str, Any], snapshot_dir: str,
               from_version: int, to_version: int) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"delta_v{from_version}_v{to_version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(delta, f, ensure_ascii=False)
    logger.info("Saved delta v%d->v%d | +nodes=%d +edges=%d -> %s",
                from_version, to_version,
                delta["summary"]["added_nodes"],
                delta["summary"]["added_edges"], path)
    return path



# Seeding the GraphActor with a previous snapshot

def _seed_actor(actor: GraphActor, adjacency: Dict[str, List[str]],
                chunk_size: int = 256) -> None:
    """
    Bulk-seed the GraphActor using its existing add_edges API. Sending edges
    in chunks keeps individual remote-call payloads small and avoids actor
    queue blowups for very large adjacency lists.
    """
    futures = []
    for src, dsts in adjacency.items():
        if not dsts:
            continue
        for i in range(0, len(dsts), chunk_size):
            chunk = dsts[i:i + chunk_size]
            futures.append(actor.add_edges.remote(src, chunk))
    if futures:
        ray.get(futures)



# Main entry: run one incremental crawl step

def run_incremental_step(
    cfg: M3Config,
    from_version: int,
    to_version: int,
    cdx_slice: Tuple[int, int],
    prev_snapshot_path: str,
    ray_already_initialised: bool = False,
) -> Dict[str, Any]:
    """
    Run a single incremental crawl step:
      1. Load the v(from_version) graph adjacency.
      2. Initialise Ray (if needed) and create a fresh GraphActor.
      3. Seed the actor with the existing adjacency.
      4. Fetch CDX records and slice [start:start+count] for this step,
         filter out URLs already present as source nodes.
      5. Dispatch cached_process_batch tasks across Ray workers.
      6. Pull the merged adjacency out of the actor.
      7. Save graph_v{to_version}.{pkl,json,csv} and delta_v{from}_v{to}.json.
    """
    slice_start, slice_count = cdx_slice
    logger.info("=" * 70)
    logger.info("Incremental step v%d -> v%d | slice=[%d:%d) | workers=%d",
                from_version, to_version, slice_start,
                slice_start + slice_count, cfg.num_workers)
    logger.info("=" * 70)

    # 1. Load previous adjacency
    prev_adj = load_snapshot_adjacency(prev_snapshot_path)
    logger.info("Loaded v%d snapshot | sources=%d", from_version, len(prev_adj))

    # 2. Ray
    if not ray_already_initialised:
        ray.init(num_cpus=cfg.num_workers, ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    actor = GraphActor.remote()

    # 3. Seed
    t0 = time.perf_counter()
    _seed_actor(actor, prev_adj)
    logger.info("Seeded GraphActor in %.2fs", time.perf_counter() - t0)

    # 4. CDX retrieval + slicing
    total_needed = slice_start + slice_count
    cdx_records = fetch_cdx_records_total(cfg, total_needed)
    if len(cdx_records) < total_needed:
        logger.warning(
            "CDX returned only %d records (wanted %d). Slice may be short.",
            len(cdx_records), total_needed,
        )
    new_records = cdx_records[slice_start:slice_start + slice_count]
    logger.info("Selected %d CDX records for this step.", len(new_records))

    # 5. Dispatch parallel fetch tasks
    m1_cfg = CrawlConfig(
        crawl_index=cfg.crawl_index,
        target_domain=cfg.target_domain,
        max_records=cfg.max_records if hasattr(cfg, "max_records") else slice_count,
        max_links_per_page=cfg.max_links_per_page,
        restrict_to_domain=cfg.restrict_to_domain,
        request_timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        user_agent=cfg.user_agent,
        batch_size=cfg.crawl_batch_size,
        num_workers=cfg.num_workers,
    )
    m1_cfg_dict = dataclasses.asdict(m1_cfg)
    skip_sources = list(prev_adj.keys())

    batches = [
        new_records[i:i + cfg.crawl_batch_size]
        for i in range(0, len(new_records), cfg.crawl_batch_size)
    ]
    logger.info("Partitioned into %d batches.", len(batches))

    t0 = time.perf_counter()
    futures = [
        cached_process_batch.remote(b, actor, m1_cfg_dict,
                                    cfg.cache_dir, skip_sources)
        for b in batches
    ]
    all_stats = ray.get(futures) if futures else []
    crawl_time = time.perf_counter() - t0

    totals = {"fetched": 0, "failed": 0, "links": 0,
              "cache_hits": 0, "cache_misses": 0, "skipped": 0}
    for s in all_stats:
        for k in totals:
            totals[k] += s.get(k, 0)
    logger.info(
        "Crawl step complete in %.1fs | fetched=%d failed=%d links=%d "
        "cache_hits=%d cache_misses=%d skipped=%d",
        crawl_time,
        totals["fetched"], totals["failed"], totals["links"],
        totals["cache_hits"], totals["cache_misses"], totals["skipped"],
    )

    # 6. Pull merged adjacency
    new_adj: Dict[str, List[str]] = ray.get(actor.get_graph.remote())
    logger.info("Pulled new adjacency | sources=%d", len(new_adj))

    # 7. Save snapshot + delta
    snap_paths = save_snapshot(new_adj, cfg.snapshot_dir, to_version)
    delta = compute_delta(prev_adj, new_adj)
    delta_path = save_delta(delta, cfg.snapshot_dir, from_version, to_version)

    return {
        "from_version": from_version,
        "to_version": to_version,
        "snapshot_paths": snap_paths,
        "delta_path": delta_path,
        "delta_summary": delta["summary"],
        "crawl_stats": totals,
        "crawl_time_s": crawl_time,
    }



# Bootstrap helper: copy the existing M1 Config 3 graph as v0 if not present.

def ensure_v0_snapshot(cfg: M3Config) -> str:
    """
    Make sure snapshots/graph_v0.pkl exists. If not, copy/derive it from the
    initial_graph_path defined in the config (M1 Config 3 by default).
    """
    v0_path = os.path.join(cfg.snapshot_dir, "graph_v0.pkl")
    if os.path.exists(v0_path):
        logger.info("v0 snapshot already present at %s", v0_path)
        return v0_path

    src = cfg.initial_graph_path
    if not os.path.isabs(src):
        src = str((Path(__file__).resolve().parent / src).resolve())
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Initial graph not found at {src}. "
            f"Update M3Config.initial_graph_path."
        )

    logger.info("Materialising v0 from %s", src)
    adj = load_snapshot_adjacency(src)
    save_snapshot(adj, cfg.snapshot_dir, 0)
    return v0_path



# Run the entire growth plan in sequence

def run_full_growth_plan(cfg: M3Config) -> List[Dict[str, Any]]:
    ensure_v0_snapshot(cfg)
    ray.init(num_cpus=cfg.num_workers, ignore_reinit_error=True,
             logging_level=logging.WARNING)

    results: List[Dict[str, Any]] = []
    prev_path = os.path.join(cfg.snapshot_dir, "graph_v0.pkl")
    for i, slice_spec in enumerate(cfg.growth_plan):
        from_v = i
        to_v = i + 1
        result = run_incremental_step(
            cfg=cfg,
            from_version=from_v,
            to_version=to_v,
            cdx_slice=tuple(slice_spec),
            prev_snapshot_path=prev_path,
            ray_already_initialised=True,
        )
        results.append(result)
        prev_path = result["snapshot_paths"]["pickle"]

    ray.shutdown()
    return results
