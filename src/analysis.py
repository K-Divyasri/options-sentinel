"""The analytical core: contrast HARD sentiment (put/call positioning) with SOFT
sentiment (crowd emotion) and surface when they DIVERGE.

This is the project's actual idea. The divergence — not either signal alone — is
the interesting, defensible insight. It is sentiment analysis, NOT financial advice.
"""
from __future__ import annotations

import numpy as np


def positioning_label(pc_volume: float | None) -> str:
    """Translate the put/call volume ratio into a positioning label.

    Higher put/call = more put activity = more bearish / hedged positioning.
    (Thresholds are conventional rules of thumb, easy to tune.)
    """
    if pc_volume is None or np.isnan(pc_volume):
        return "unknown"
    if pc_volume >= 1.1:
        return "bearish"   # heavy put activity
    if pc_volume <= 0.7:
        return "bullish"   # heavy call activity
    return "neutral"


def crowd_label(emotion, vader: float | None) -> str:
    """Reduce the crowd's emotion to a bullish/bearish/neutral mood."""
    if emotion is not None and emotion.bullish_bearish in {"bullish", "bearish", "neutral"}:
        return emotion.bullish_bearish
    if vader is None:
        return "unknown"
    if vader > 0.15:
        return "bullish"
    if vader < -0.15:
        return "bearish"
    return "neutral"


def divergence_read(positioning: str, crowd: str) -> dict:
    """Compare the two sentiment reads and describe their relationship."""
    if positioning == "unknown" or crowd == "unknown":
        return {
            "state": "incomplete",
            "headline": "Not enough data to compare hard vs soft sentiment.",
            "detail": "",
        }
    if positioning == crowd:
        return {
            "state": "aligned",
            "headline": f"Aligned: positioning and crowd mood are both {positioning}.",
            "detail": (
                "Hard money (options) and soft talk (social/news) agree. This is the "
                "expected, lower-information state."
            ),
        }
    return {
        "state": "divergent",
        "headline": (
            f"Divergence: options positioning looks {positioning} while the crowd "
            f"feels {crowd}."
        ),
        "detail": (
            "The bet and the mood disagree. Classic contrarian setups live here — for "
            "example an excited crowd while traders quietly buy downside protection. "
            "Treat this as a question to investigate, not a signal to act on."
        ),
    }


def build_dossier(symbol, options, market_pc, narrative, emotion, vader, reddit_n) -> dict:
    """Assemble every section into one dossier dict for the UI."""
    pc_vol = options.get("put_call_volume") if options.get("available") else None
    positioning = positioning_label(pc_vol)
    crowd = crowd_label(emotion, vader)
    return {
        "symbol": symbol,
        "positioning_label": positioning,
        "crowd_label": crowd,
        "divergence": divergence_read(positioning, crowd),
        "options": options,
        "market_pc": market_pc,
        "narrative": narrative,
        "emotion": emotion,
        "vader": vader,
        "reddit_n": reddit_n,
    }
