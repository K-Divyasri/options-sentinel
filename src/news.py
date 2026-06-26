"""Fetch recent news headlines for a ticker via yfinance (free, no API key).

yfinance changed its news data shape over time, so we defensively handle both
the newer 'content' nested format and the older flat format.
"""
from __future__ import annotations

from datetime import datetime

import yfinance as yf

from config import NEWS_LIMIT


def _parse(item: dict) -> dict:
    """Normalize one news item into a flat dict regardless of yfinance version."""
    if "content" in item:  # newer format
        c = item["content"]
        url = ""
        if isinstance(c.get("canonicalUrl"), dict):
            url = c["canonicalUrl"].get("url", "")
        elif isinstance(c.get("clickThroughUrl"), dict):
            url = c["clickThroughUrl"].get("url", "")
        provider = c.get("provider", {})
        return {
            "title": c.get("title", ""),
            "summary": c.get("summary", ""),
            "publisher": provider.get("displayName", "") if isinstance(provider, dict) else "",
            "link": url,
            "date": c.get("pubDate", ""),
        }

    ts = item.get("providerPublishTime")
    return {
        "title": item.get("title", ""),
        "summary": item.get("summary", ""),
        "publisher": item.get("publisher", ""),
        "link": item.get("link", ""),
        "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
    }


def get_news(symbol: str, limit: int = NEWS_LIMIT) -> list[dict]:
    """Return up to `limit` recent news items as flat dicts."""
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        return []
    items = [_parse(x) for x in raw[:limit]]
    return [x for x in items if x["title"]]
