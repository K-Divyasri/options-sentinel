"""Download CBOE's free daily put/call ratio archive and place today's value
in historical context.

Returning a raw number ("today's ratio is 0.9") is weak. A reviewer wants the
*context*: is 0.9 high or low for this market? So we also compute a percentile
and a z-score against the trailing ~year. That statistical framing is the
quant habit worth showing.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from config import CBOE_PC_URL


def get_market_put_call(url: str = CBOE_PC_URL, window: int = 252) -> dict:
    """Return latest market put/call ratio + percentile + z-score vs `window` days."""
    try:
        raw = requests.get(url, timeout=20).text
    except Exception as exc:
        return {"available": False, "reason": f"Download failed: {exc}"}

    # The CBOE file has a couple of preamble lines before the real header row.
    lines = raw.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.lower().startswith("trade_date") or ln.lower().startswith("date")),
        None,
    )
    if header_idx is None:
        return {"available": False, "reason": "Unexpected CBOE file format."}

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    ratio_col = next((c for c in df.columns if "ratio" in c or c == "p/c"), df.columns[-1])
    date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    series = pd.to_numeric(df[ratio_col], errors="coerce").dropna()
    if series.empty:
        return {"available": False, "reason": "No ratio values parsed."}

    latest = float(series.iloc[-1])
    recent = series.tail(window)
    percentile = float((recent < latest).mean() * 100)
    zscore = float((latest - recent.mean()) / recent.std()) if recent.std() else np.nan

    return {
        "available": True,
        "latest": latest,
        "percentile": percentile,        # 0-100: where today sits vs the last ~year
        "zscore": zscore,
        "as_of": str(df[date_col].iloc[-1].date()),
    }
