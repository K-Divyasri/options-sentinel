"""Pull recent social sentiment from StockTwits — a stock-focused social network.

Why StockTwits instead of Reddit:
- NO API key and NO app registration (uses a public endpoint).
- Each message can carry an explicit user-tagged Bullish/Bearish label, which is a
  cleaner crowd signal than inferring mood from free text.

Caveats: this is an unofficial public endpoint, so it is rate-limited and can
occasionally block requests. We cache results upstream and fail gracefully
(returning an empty list), so the app still works on news text alone.
"""
from __future__ import annotations

import requests

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (options-sentinel; research/educational)"}


def social_available() -> bool:
    """StockTwits needs no credentials, so this is always available to try."""
    return True


def get_social_posts(symbol: str) -> list[dict]:
    """Return recent StockTwits messages for `symbol`.

    Each item: {text, tag, followers} where `tag` is the user's own
    'Bullish'/'Bearish' label (or None if they didn't tag it).
    """
    try:
        resp = requests.get(
            STOCKTWITS_URL.format(symbol=symbol), headers=HEADERS, timeout=15
        )
        if resp.status_code != 200:
            return []
        messages = resp.json().get("messages", [])
    except Exception:
        return []

    posts: list[dict] = []
    for m in messages:
        body = (m.get("body") or "").strip()
        if not body:
            continue
        tag = None
        sentiment = (m.get("entities") or {}).get("sentiment")
        if isinstance(sentiment, dict):
            tag = sentiment.get("basic")  # "Bullish" | "Bearish" | None
        user = m.get("user") or {}
        posts.append(
            {
                "text": body[:600],
                "tag": tag,
                "followers": int(user.get("followers", 0) or 0),
            }
        )
    return posts


def bull_bear_tally(posts: list[dict]) -> dict:
    """Count the explicit Bullish/Bearish tags users attached to their messages."""
    bull = sum(1 for p in posts if p.get("tag") == "Bullish")
    bear = sum(1 for p in posts if p.get("tag") == "Bearish")
    total = bull + bear
    return {
        "bull": bull,
        "bear": bear,
        "bull_pct": (bull / total * 100) if total else None,
    }
