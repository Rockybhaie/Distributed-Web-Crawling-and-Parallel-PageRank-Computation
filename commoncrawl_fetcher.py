
# Query CommonCrawl and download WARC records.



# Architecture overview
# 1.  fetch_cdx_records() queries the CommonCrawl CDX Index HTTP API and returns a list of record metadata dicts.
# 2.  fetch_warc_record()  →  for a single CDX record, downloads *only* the specific byte range of the WARC file that contains that page (using an HTTP Range request).
# 3.  extract_links()      →  parses the HTML in the WARC response record and returns all hyperlinks found on that page.


import gzip
import io
import logging
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator

from config import CrawlConfig

logger = logging.getLogger(__name__)

# Base URL for the CommonCrawl CDX Server API
CDX_API_BASE = "https://index.commoncrawl.org/{index}-index"

# Base URL from which WARC files are served
WARC_BASE_URL = "https://data.commoncrawl.org/"



#  Phase 1 — Fetch CDX index records                                           


def fetch_cdx_records(config: CrawlConfig) -> List[Dict]:
    
    api_url = CDX_API_BASE.format(index=config.crawl_index)

    params = {
        "url":     f"*.{config.target_domain}",  # wildcard for all sub-paths
        "output":  "json",
        "fl":      "url,filename,offset,length,status,mime",  # fields we need
        "limit":   config.max_records,
        # Only keep successful HTML responses — we cannot extract links from
        # PDFs, images, or error pages.
        "filter":  ["status:200", "mime:text/html"],
    }

    logger.info(
        "Querying CommonCrawl CDX API | index=%s domain=%s limit=%d",
        config.crawl_index, config.target_domain, config.max_records,
    )

    session = _make_session(config)

    for attempt in range(1, config.max_retries + 1):
        try:
            resp = session.get(api_url, params=params, timeout=config.request_timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning("CDX API attempt %d/%d failed: %s", attempt, config.max_retries, exc)
            if attempt == config.max_retries:
                raise
            time.sleep(2 ** attempt)  # exponential back-off

    # The CDX API returns one JSON object per line (JSON Lines format)
    records = []
    for line in resp.text.strip().splitlines():
        if not line:
            continue
        try:
            import json
            record = json.loads(line)
            # Only keep records that have all required fields
            if all(k in record for k in ("url", "filename", "offset", "length")):
                record["offset"] = int(record["offset"])
                record["length"] = int(record["length"])
                records.append(record)
        except (ValueError, KeyError) as exc:
            logger.debug("Skipping malformed CDX line: %s | error: %s", line[:80], exc)

    logger.info("Retrieved %d usable CDX records.", len(records))
    return records



#  Phase 2 — Fetch and parse a single WARC record                              
def fetch_warc_record(cdx_record: Dict, config: CrawlConfig) -> Optional[str]:
    
    warc_url = WARC_BASE_URL + cdx_record["filename"]
    byte_start = cdx_record["offset"]
    byte_end   = cdx_record["offset"] + cdx_record["length"] - 1

    headers = {
        "Range":      f"bytes={byte_start}-{byte_end}",
        "User-Agent": config.user_agent,
    }

    session = _make_session(config)

    for attempt in range(1, config.max_retries + 1):
        try:
            resp = session.get(
                warc_url, headers=headers, timeout=config.request_timeout
            )
            # 206 Partial Content is the correct status for a Range request
            if resp.status_code not in (200, 206):
                logger.debug(
                    "WARC fetch got HTTP %d for %s", resp.status_code, cdx_record["url"]
                )
                return None
            break
        except requests.RequestException as exc:
            logger.warning(
                "WARC fetch attempt %d/%d failed for %s: %s",
                attempt, config.max_retries, cdx_record["url"], exc,
            )
            if attempt == config.max_retries:
                return None
            time.sleep(2 ** attempt)

    return _parse_html_from_warc_bytes(resp.content)


def _parse_html_from_warc_bytes(raw_bytes: bytes) -> Optional[str]:
    
    try:
        stream = io.BytesIO(raw_bytes)
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                content = record.content_stream().read()
                # Try common encodings; fall back to latin-1 which never fails
                for enc in ("utf-8", "latin-1"):
                    try:
                        return content.decode(enc)
                    except UnicodeDecodeError:
                        continue
    except Exception as exc:
        logger.debug("Failed to parse WARC bytes: %s", exc)
    return None



#  Phase 3 — Extract hyperlinks from HTML                                      


def extract_links(source_url: str, html: str, config: CrawlConfig) -> List[str]:
    
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # Fall back to the built-in parser if lxml is unavailable
        soup = BeautifulSoup(html, "html.parser")

    links: List[str] = []
    seen: set = set()

    for tag in soup.find_all("a", href=True):
        raw_href = tag["href"].strip()
        if not raw_href or raw_href.startswith("#"):
            # Skip fragment-only links — they don't represent new pages
            continue

        # Resolve relative URLs against the source page's base URL
        absolute = urljoin(source_url, raw_href)
        normalised = _normalise_url(absolute)

        if normalised is None or normalised in seen:
            continue

        # Domain filter: only keep links within the target domain
        if config.restrict_to_domain:
            parsed = urlparse(normalised)
            if config.restrict_to_domain not in parsed.netloc:
                continue

        seen.add(normalised)
        links.append(normalised)

        if len(links) >= config.max_links_per_page:
            break

    return links



#Strip query strings and fragments, and return None for non-HTTP URLs
#(e.g. mailto:, javascript:, tel:).

def _normalise_url(url: str) -> Optional[str]:
    
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        # Rebuild without query or fragment to canonicalise
        return parsed._replace(query="", fragment="").geturl()
    except Exception:
        return None



#  Utility        
# Return a requests Session with our user-agent pre-set                                                             
def _make_session(config: CrawlConfig) -> requests.Session:
    
    session = requests.Session()
    session.headers["User-Agent"] = config.user_agent
    return session
