"""Pull recent Reddit posts mentioning the ticker from finance subreddits.

This is OPTIONAL. If Reddit credentials aren't set in .env, the functions return
empty results and the rest of the app keeps working (crowd emotion then runs on
news text only). The free Reddit tier (~100 requests/min) is plenty here, but we
keep limits small and cache results to stay well under it.
"""
from __future__ import annotations

from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS,
    REDDIT_LOOKBACK,
    REDDIT_LIMIT_PER_SUB,
)


def reddit_available() -> bool:
    """True only if both Reddit credentials are present."""
    return bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)


def get_reddit_posts(symbol: str) -> list[dict]:
    """Return recent posts mentioning `symbol`; empty list if Reddit isn't set up."""
    if not reddit_available():
        return []

    try:
        import praw

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        reddit.read_only = True
    except Exception:
        return []

    posts: list[dict] = []
    for sub in REDDIT_SUBREDDITS:
        try:
            results = reddit.subreddit(sub).search(
                symbol, sort="new", time_filter=REDDIT_LOOKBACK,
                limit=REDDIT_LIMIT_PER_SUB,
            )
            for post in results:
                text = f"{post.title}. {post.selftext or ''}".strip()
                if len(text) > 15:
                    posts.append(
                        {"subreddit": sub, "score": int(post.score), "text": text[:600]}
                    )
        except Exception:
            continue
    return posts
