
# Central configuration for the Distributed Web Crawler
# All tunable parameters live here so you never have to hunt through code files
# to change a setting.
# Design decesions explained more detailed in the report.


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrawlConfig:
    
    # CommonCrawl settings                                                 
    
    # The crawl snapshot to query.  Find available indices at:
    # https://index.commoncrawl.org/
    crawl_index: str = "CC-MAIN-2024-10"

    # Domain we want to use / to restrict the crawl. We are using Wikipedia.
    target_domain: str = "en.wikipedia.org"

    # Maximum number of CDX index records to retrieve from CommonCrawl. we can increase 
    # this number but for faster experimentation 500 is a reasonable number.
    max_records: int = 500

    # How many WARC records to bundle into a single Ray task.
    batch_size: int = 25

    
    # Crawl-scope constraints (ethical guardrails)                        
   
    # Cap the number of out-links stored per page to avoid star nodes
    # dominating the graph and to limit memory usage.
    max_links_per_page: int = 150

    # Only keep edges that stay within this domain.  Set to None to allow
    # cross-domain edges.
    restrict_to_domain: Optional[str] = "en.wikipedia.org"

    
    # Ray / parallelism settings                                           
    
    # Number of parallel Ray task slots to keep in-flight at once.
    # Rule of thumb: set to number of CPU cores − 1.
    num_workers: int = 4

    
    # HTTP request settings                                                
   
    request_timeout: int = 30   # seconds per HTTP request
    max_retries:     int = 3    # retries on transient failures

    # User-agent sent with every request so CommonCrawl can identify us.
    user_agent: str = (
        "PDC-Course-Crawler/1.0 "
        "(academic project; uses CommonCrawl data only - no live crawling)"
    )

    
    # Output settings                                                      
    
    output_dir:      str = "output"
    # Base filename for saved graph artefacts
    graph_filename:  str = "web_graph"
