
# pagerank_config.py — Central configuration for Milestone 2: Parallel PageRank.

# All tunable parameters live here so nothing is hardcoded in the algorithm files.
# Follows the same pattern as config.py from Milestone 1.


from dataclasses import dataclass
from typing import Optional


@dataclass
class PageRankConfig:

    
    # Graph input settings (from Milestone 1 output)                      
    graph_path: str = "output_config3/web_graph.pkl"

    
    # PageRank algorithm parameters   
                                         
    # Damping factor (Google uses 0.85)
    # Probability that a random surfer follows a link (vs teleports)
    damping_factor: float = 0.85

    # Maximum number of iterations before stopping (fixed-iteration mode)
    max_iterations: int = 100

    # Convergence threshold — stop when max rank change < this value
    # Used in convergence-based termination mode
    convergence_threshold: float = 1e-6

    # Whether to use convergence-based termination (True) or
    # fixed iteration count (False)
    use_convergence: bool = True

    
    # Parallelism settings       
                                              
    # Number of Ray workers for parallel PageRank
    num_workers: int = 4

    # How many graph partitions to split the nodes into
    # Usually set equal to num_workers for best load balancing
    num_partitions: int = 4

    
    # Execution strategy                                                   

    # "centralized"  — all workers send partial sums to the driver
    # "distributed"  — workers exchange rank updates peer-to-peer via Actor
    strategy: str = "centralized"


    # Output settings                                                      
    output_dir: str = "pagerank_output"

    # Save the final PageRank scores to a JSON file
    save_scores: bool = True