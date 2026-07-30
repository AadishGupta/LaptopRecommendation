"""
live_pricing.py - Live laptop market price lookup (Google Shopping via SerpApi)

Enriches the static Qdrant catalog price with a current market price pulled
from Google Shopping. Fails soft everywhere: no API key, no network, no
match, or a malformed response all just mean "no live price today" — the
catalog `price` field from Qdrant stays the source of truth for budget
filtering/ranking elsewhere in the pipeline. Live price is a display-layer
addition only, applied after retrieval/ranking is already done.

Setup:
    pip install requests
    export SERPAPI_KEY="your_key"      # https://serpapi.com — free tier: 100/mo
Without SERPAPI_KEY set, live pricing is silently disabled (catalog price
is used everywhere, no behavior change, no crash).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

LIVE_PRICING_ENABLED = bool(SERPAPI_KEY)
if not LIVE_PRICING_ENABLED:
    logger.warning("SERPAPI_KEY not set — live pricing disabled, catalog price will be used")

CACHE_TTL_SECONDS = 6 * 60 * 60  # 6h cache — SerpApi free tier is only 100 searches/month
REQUEST_TIMEOUT = 6

# laptop_name -> (fetched_at_epoch, result_dict_or_None)
_cache: Dict[str, tuple] = {}


def _extract_price_inr(price_str: str) -> Optional[int]:
    """'₹54,999.00' / 'Rs. 54,999' / '54999' -> 54999. None if unparsable."""
    if not price_str:
        return None
    digits = re.sub(r"[^\d]", "", str(price_str))
    return int(digits) if digits else None


_SPEC_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _clean_query_name(name: str) -> str:
    """
    Strip a trailing parenthetical spec block (e.g. "(9th Gen Core i7/ 32GB/
    1TB SSD/ Win10/ 6GB Graph)") before sending the name to Google Shopping.

    The catalog stores spec-heavy display names. Searching that raw string —
    slashes, spec shorthand, and all — doesn't look like anything a real
    product listing is titled, so it returns noisy/irrelevant top results.
    Searching just the bare model name matches how listings are actually
    titled, and is the fix for most of the bad-match cases.
    """
    if not name:
        return name
    cleaned = _SPEC_PAREN_RE.sub("", name).strip()
    return cleaned or name


def _title_matches(query_name: str, result_title: str, min_ratio: float = 0.45) -> bool:
    """
    Cheap relevance guard. SerpApi ranks shopping_results by its own
    relevance model, not ours — item[0] is not guaranteed to be the same
    product; it can be an accessory, a different generation/variant, or an
    unrelated listing entirely. Compare normalized token overlap between
    what we searched for and the candidate's title, and reject weak
    matches instead of trusting position 0 blindly.
    """
    if not result_title:
        return False
    q_tokens = set(re.findall(r"[a-z0-9]+", query_name.lower()))
    t_tokens = set(re.findall(r"[a-z0-9]+", result_title.lower()))
    if not q_tokens:
        return False
    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    return overlap >= min_ratio


def get_live_price(laptop_name: str, catalog_price: Optional[int] = None) -> Optional[dict]:
    """
    Look up laptop_name on Google Shopping (India) via SerpApi.
    Returns {"price": int, "source": str, "url": str} or None on any failure
    (no key, no network, no match, bad response). Cached per name for
    CACHE_TTL_SECONDS so repeated calls in one session don't burn quota.
    """
    if not LIVE_PRICING_ENABLED:
        return None
    if not laptop_name:
        logger.info("   [LIVE PRICE] skipped — no laptop name to look up")
        return None

    cached = _cache.get(laptop_name)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        logger.info(f"   [LIVE PRICE] cache hit for {laptop_name!r} -> {cached[1]}")
        return cached[1]

    query_name = _clean_query_name(laptop_name)
    logger.info(f"   [LIVE PRICE] querying Google Shopping for {query_name!r} (raw name: {laptop_name!r}) ...")
    result = None
    try:
        resp = requests.get(
            SERPAPI_URL,
            params={
                "engine": "google_shopping",
                "q": query_name,
                "gl": "in",
                "hl": "en",
                "api_key": SERPAPI_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("shopping_results", [])

        # Scan candidates in SerpApi's own ranked order, but don't trust
        # item[0] blindly — skip anything whose title doesn't plausibly
        # match the product we searched for, and (when we know the catalog
        # price) anything whose price is wildly out of line, since that's a
        # much stronger signal of a wrong-product match than a real deal.
        for top in items[:5]:
            title = top.get("title", "")
            if not _title_matches(query_name, title):
                logger.info(f"   [LIVE PRICE] skipping weak title match {title!r} for {query_name!r}")
                continue

            price_raw = top.get("extracted_price")
            price = (
                int(price_raw)
                if isinstance(price_raw, (int, float))
                else _extract_price_inr(top.get("price", ""))
            )
            if not price:
                continue

            if catalog_price:
                ratio = price / catalog_price
                if ratio < 0.4 or ratio > 2.5:
                    logger.info(
                        f"   [LIVE PRICE] rejecting implausible price ₹{price:,} for {query_name!r} "
                        f"(title={title!r}) vs catalog ₹{catalog_price:,} (ratio={ratio:.2f}x)"
                    )
                    continue

            result = {
                "price": price,
                "source": top.get("source", "Google Shopping"),
                "url": top.get("product_link") or top.get("link", ""),
            }
            break
    except Exception as exc:
        logger.warning(f"   [LIVE PRICE] lookup failed for {laptop_name!r}: {exc}")
        result = None

    if result:
        logger.info(f"   [LIVE PRICE] found for {laptop_name!r}: ₹{result['price']:,} via {result['source']}")
    else:
        logger.info(f"   [LIVE PRICE] no live price found for {laptop_name!r} — will show catalog price")

    _cache[laptop_name] = (time.time(), result)
    return result


def enrich_with_live_prices(laptops: List[dict]) -> List[dict]:
    """
    Adds a 'live_price' key (dict or None) to each laptop dict, in place, and
    returns the same list. Never touches 'price' — that stays the catalog
    value used for budget filtering/ranking upstream of this call.
    """
    if not LIVE_PRICING_ENABLED:
        logger.info("   [LIVE PRICE] disabled (SERPAPI_KEY not set) — showing catalog prices only")
        for lap in laptops:
            lap.setdefault("live_price", None)
        return laptops

    names = [lap.get("name", "?") for lap in laptops]
    logger.info(f"   [LIVE PRICE] enriching {len(laptops)} laptop(s): {names}")
    for lap in laptops:
        lap["live_price"] = get_live_price(lap.get("name", ""), catalog_price=lap.get("price"))
    hits = sum(1 for lap in laptops if lap.get("live_price"))
    logger.info(f"   [LIVE PRICE] done — {hits}/{len(laptops)} got a live price")
    return laptops