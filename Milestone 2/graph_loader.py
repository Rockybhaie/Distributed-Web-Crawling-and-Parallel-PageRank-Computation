# graph_loader.py — Load and prepare the web graph from Milestone 1 outputs.

"""
This module bridges Milestone 1 and Milestone 2.  It loads the graph files
produced by graph_builder.py and converts them into the data structures that
the PageRank workers need.

The PageRank algorithm works on:
    - A list of all node IDs (URLs mapped to integer indices for efficiency)
    - An in-link index: for each node, which nodes link TO it
    - An out-degree map: for each node, how many outgoing links it has
    - A dangling node set: nodes with zero outgoing edges

"""

import logging
import os
import pickle
from typing import Dict, List, Set, Tuple

import networkx as nx

from pagerank_config import PageRankConfig

logger = logging.getLogger(__name__)



#  Main loader function -- Load the graph from Milestone 1 output and prepare it for PageRank.
"""
    Returns
    dict with keys:
        nodes       : list of all URL strings (index = node ID)
        url_to_id   : dict mapping URL string -> integer index
        in_links    : list of lists — in_links[i] = [j, k, ...] means j and k link to i
        out_degree  : list of ints — out_degree[i] = number of outgoing links from i
        dangling    : set of ints — node IDs with out_degree == 0
        num_nodes   : int — total number of nodes
        num_edges   : int — total number of directed edges
"""                                                        
def load_graph(config: PageRankConfig) -> Dict:
    
    
    # Step 1: Load the NetworkX DiGraph                                  
    G = _load_networkx_graph(config)

   
    # Step 2: Build integer-indexed data structures                       
    nodes = list(G.nodes())
    url_to_id = {url: i for i, url in enumerate(nodes)}
    num_nodes = len(nodes)

    logger.info(
        "Graph loaded | nodes=%d  edges=%d",
        G.number_of_nodes(), G.number_of_edges(),
    )

    # Build in-link index (who links TO each node)
    # in_links[i] = list of node IDs that have an edge pointing to node i
    in_links: List[List[int]] = [[] for _ in range(num_nodes)]

    # Build out-degree (how many links each node sends out)
    out_degree: List[int] = [0] * num_nodes

    for source_url, target_url in G.edges():
        src = url_to_id[source_url]
        tgt = url_to_id[target_url]
        in_links[tgt].append(src)
        out_degree[src] += 1

    # Dangling nodes: nodes with zero outgoing edges
    # These absorb rank without redistributing it — special handling needed
    dangling: Set[int] = {i for i in range(num_nodes) if out_degree[i] == 0}

    logger.info(
        "Data structures built | dangling_nodes=%d (%.1f%% of total)",
        len(dangling), 100.0 * len(dangling) / num_nodes if num_nodes > 0 else 0,
    )

    return {
        "nodes":      nodes,
        "url_to_id":  url_to_id,
        "in_links":   in_links,
        "out_degree": out_degree,
        "dangling":   dangling,
        "num_nodes":  num_nodes,
        "num_edges":  G.number_of_edges(),
        "graph":      G,   # keep the raw NetworkX graph for sequential baseline
    }


def _load_networkx_graph(config: PageRankConfig) -> nx.DiGraph:
    """Load the NetworkX DiGraph from the Milestone 1 pickle file."""
    if os.path.exists(config.graph_path):
        logger.info("Loading graph from pickle: %s", config.graph_path)
        with open(config.graph_path, "rb") as f:
            G = pickle.load(f)
        return G
    else:
        raise FileNotFoundError(
            f"Graph pickle not found at '{config.graph_path}'. "
            f"Please run Milestone 1 first and check the path in pagerank_config.py."
        )



#  Partition helper -- Partition the nodes into roughly equal chunks for parallel processing. 
# Each chunk is assigned to one Ray worker.                                                             

def partition_nodes(num_nodes: int, num_partitions: int) -> List[List[int]]:
    
    chunk_size = max(1, num_nodes // num_partitions)
    partitions = []
    for i in range(num_partitions):
        start = i * chunk_size
        end = start + chunk_size if i < num_partitions - 1 else num_nodes
        partitions.append(list(range(start, end)))
    return partitions