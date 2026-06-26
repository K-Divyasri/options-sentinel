"""Pull a ticker's option chain and compute positioning metrics.

Metrics produced:
- put/call ratio by VOLUME (today's bets) and by OPEN INTEREST (standing bets)
- an implied-volatility skew proxy (are puts more expensive than calls?)
- 'unusual activity': contracts where today's volume exceeds open interest

All data comes from yfinance (free, ~15 min delayed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import N_EXPIRIES


def _safe_sum(series: pd.Series) -> float:
    """Sum a column, treating missing values as 0 (yfinance has many NaNs)."""
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def get_spot(symbol: str) -> float | None:
    """Latest traded price, or None if unavailable."""
    try:
        price = yf.Ticker(symbol).fast_info.get("lastPrice")
        return float(price) if price is not None else None
    except Exception:
        return None


def analyze_options(symbol: str, n_expiries: int = N_EXPIRIES) -> dict:
    """Return a dict of positioning metrics for the nearest `n_expiries` expiries."""
    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options or [])[:n_expiries]
    spot = get_spot(symbol)

    if not expiries:
        return {"available": False, "reason": "No options listed for this ticker."}

    calls_frames, puts_frames = [], []
    for expiry in expiries:
        try:
            chain = ticker.option_chain(expiry)
            calls_frames.append(chain.calls.assign(expiry=expiry))
            puts_frames.append(chain.puts.assign(expiry=expiry))
        except Exception:
            continue

    if not calls_frames or not puts_frames:
        return {"available": False, "reason": "Could not download option chains."}

    calls = pd.concat(calls_frames, ignore_index=True)
    puts = pd.concat(puts_frames, ignore_index=True)

    call_vol, put_vol = _safe_sum(calls["volume"]), _safe_sum(puts["volume"])
    call_oi, put_oi = _safe_sum(calls["openInterest"]), _safe_sum(puts["openInterest"])

    pc_volume = put_vol / call_vol if call_vol else np.nan
    pc_oi = put_oi / call_oi if call_oi else np.nan

    # IV skew proxy: average IV of out-of-the-money puts minus OTM calls.
    # A positive number means downside protection is bid up (fearful).
    iv_skew = np.nan
    if spot:
        otm_puts = puts.loc[puts["strike"] < spot * 0.95, "impliedVolatility"]
        otm_calls = calls.loc[calls["strike"] > spot * 1.05, "impliedVolatility"]
        if len(otm_puts) and len(otm_calls):
            iv_skew = float(otm_puts.mean() - otm_calls.mean())

    # Unusual activity: more contracts traded today than were already open.
    def _unusual(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        d = df.copy()
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
        d["openInterest"] = pd.to_numeric(d["openInterest"], errors="coerce").fillna(0)
        d = d[(d["volume"] > d["openInterest"]) & (d["volume"] > 100)]
        d["type"] = kind
        return d[["type", "expiry", "strike", "volume", "openInterest", "impliedVolatility"]]

    unusual = pd.concat(
        [_unusual(calls, "call"), _unusual(puts, "put")], ignore_index=True
    ).sort_values("volume", ascending=False).head(10)

    return {
        "available": True,
        "spot": spot,
        "expiries_used": expiries,
        "put_call_volume": pc_volume,
        "put_call_oi": pc_oi,
        "iv_skew": iv_skew,
        "total_put_volume": put_vol,
        "total_call_volume": call_vol,
        "unusual_activity": unusual,
    }
