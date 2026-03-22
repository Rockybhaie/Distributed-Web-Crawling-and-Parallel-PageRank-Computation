
# Assemble, validate, and persist the web graph.

# After all Ray tasks finish, the driver calls functions here to:
# 1.  Pull the adjacency list from the GraphActor.
# 2.  Optionally build a NetworkX DiGraph for analysis / visualisation.
# 3.  Save the graph in three formats so Milestone 2 can choose freely:
            # JSON adjacency list  (web_graph.json)
            # CSV edge list        (web_graph_edges.csv)
            # Pickle / NetworkX    (web_graph.pkl)
# 4.  Print a brief structural summary.




import csv
import json
import logging
import os
import pickle
from typing import Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)



#  Public API                                                                   

# Convert a raw adjacency-list dict into a NetworkX DiGraph
def build_networkx_graph(adjacency: Dict[str, List[str]]) -> nx.DiGraph:
    
    G = nx.DiGraph()

    for source, targets in adjacency.items():
        G.add_node(source)
        for target in targets:
            # add_edge implicitly adds target as a node if it isn't already
            G.add_edge(source, target)

    logger.info(
        "NetworkX graph built | nodes=%d  edges=%d",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G



# Save the graph in JSON, CSV, and pickle formats.
def save_graph(
    adjacency: Dict[str, List[str]],
    output_dir: str,
    base_name: str = "web_graph",
) -> Dict[str, str]:
    
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    
    # 1.  JSON adjacency list                                             
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(adjacency, f, indent=2, ensure_ascii=False)
    paths["json"] = json_path
    logger.info("Saved JSON adjacency list → %s", json_path)


    # 2.  CSV edge list  (one row per directed edge: source,target)       
    csv_path = os.path.join(output_dir, f"{base_name}_edges.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])  # header
        for source, targets in adjacency.items():
            for target in targets:
                writer.writerow([source, target])
    paths["csv"] = csv_path
    logger.info("Saved CSV edge list      → %s", csv_path)


    # 3.  NetworkX DiGraph pickle                                         
    G = build_networkx_graph(adjacency)
    pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["pickle"] = pkl_path
    logger.info("Saved NetworkX pickle    → %s", pkl_path)

    return paths


def print_summary(adjacency: Dict[str, List[str]], G: Optional[nx.DiGraph] = None) -> None:
    
    num_nodes = len(adjacency)
    num_edges = sum(len(v) for v in adjacency.values())

    print("\n" + "=" * 60)
    print("  WEB GRAPH SUMMARY — Milestone 1")
    print("=" * 60)
    print(f"  Source nodes (pages crawled):  {num_nodes:>10,}")
    print(f"  Directed edges (hyperlinks):   {num_edges:>10,}")

    if num_nodes > 0:
        avg_out = num_edges / num_nodes
        print(f"  Avg out-degree per page:       {avg_out:>10.2f}")

    if G is not None and G.number_of_nodes() > 0:
        # NetworkX gives us richer stats
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        print(f"  Total nodes (incl. targets):   {G.number_of_nodes():>10,}")
        print(f"  Max in-degree:                 {max(in_degrees):>10,}")
        print(f"  Max out-degree:                {max(out_degrees):>10,}")

        # Check for dangling nodes (no outgoing edges) — important for
        # PageRank because they leak rank; Milestone 2 must handle them.
        dangling = sum(1 for d in out_degrees if d == 0)
        print(f"  Dangling nodes (out-deg = 0):  {dangling:>10,}")

        # Weakly connected components — gives a sense of graph connectivity
        num_wcc = nx.number_weakly_connected_components(G)
        print(f"  Weakly connected components:   {num_wcc:>10,}")

    print("=" * 60 + "\n")

# Load a previously saved JSON adjacency list.
def load_graph_json(path: str) -> Dict[str, List[str]]:
   
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load a previously saved NetworkX DiGraph pickle.
def load_graph_pickle(path: str) -> nx.DiGraph:
   
    with open(path, "rb") as f:
        return pickle.load(f)
