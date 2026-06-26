"""Options Sentinel — Streamlit UI.

Run with:   streamlit run app.py

Contrasts HARD sentiment (put/call positioning) against SOFT sentiment (news +
StockTwits crowd emotion) and flags when they diverge.

SENTIMENT ANALYSIS ONLY — NOT FINANCIAL ADVICE. Data is delayed and may be wrong.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.options_data import analyze_options
from src.market_pc import get_market_put_call
from src.news import get_news
from src.social_data import get_social_posts, social_available, bull_bear_tally
from src.sentiment import vader_score, classify_emotion
from src.rag import news_narrative
from src.analysis import build_dossier

st.set_page_config(page_title="Options Sentinel", page_icon="🛰️", layout="wide")


# --- cached data layers: each source is fetched at most once per TTL window ---
@st.cache_data(ttl=600, show_spinner=False)
def _options(sym):
    return analyze_options(sym)


@st.cache_data(ttl=1800, show_spinner=False)
def _market_pc():
    return get_market_put_call()


@st.cache_data(ttl=600, show_spinner=False)
def _news(sym):
    return get_news(sym)


@st.cache_data(ttl=600, show_spinner=False)
def _social(sym):
    return get_social_posts(sym)


@st.cache_data(ttl=600, show_spinner=False)
def _narrative(sym, news):
    return news_narrative(sym, news)


@st.cache_data(ttl=600, show_spinner=False)
def _emotion(texts):
    return classify_emotion(texts)


# ----------------------------- sidebar -----------------------------
st.sidebar.title("🛰️ Options Sentinel")
symbol = st.sidebar.text_input("Ticker", "NVDA").strip().upper()
run = st.sidebar.button("Analyze", type="primary")
st.sidebar.caption("Sentiment analysis only — **not financial advice.**")

if not config.active_key():
    st.error(f"No API key found for provider '{config.PROVIDER}'. Copy `.env.example` "
             "to `.env` and add the right key (see README).")
    st.stop()

if not run:
    st.title("🛰️ Options Sentinel")
    st.write("Enter a ticker in the sidebar and click **Analyze**.")
    st.info("This tool contrasts **hard** sentiment (put/call positioning) with "
            "**soft** sentiment (news + StockTwits crowd emotion), and flags when they diverge.")
    st.stop()

# ----------------------------- pipeline ----------------------------
with st.spinner("Pulling options, news, and social sentiment…"):
    options = _options(symbol)
    market_pc = _market_pc()
    news = _news(symbol)
    social_posts = _social(symbol)
    texts = [f"{n['title']}. {n['summary']}" for n in news] + [p["text"] for p in social_posts]
    vader = vader_score(texts)
    emotion = _emotion(texts)
    narrative = _narrative(symbol, news)
    dossier = build_dossier(symbol, options, market_pc, narrative, emotion, vader,
                            len(social_posts))

# ----------------------------- header ------------------------------
st.title(f"🛰️ {symbol}")
spot = options.get("spot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"${spot:,.2f}" if spot else "—")
c2.metric("Put/Call (volume)",
          f"{options.get('put_call_volume'):.2f}" if options.get("available") else "—")
c3.metric("Put/Call (open int.)",
          f"{options.get('put_call_oi'):.2f}" if options.get("available") else "—")
iv = options.get("iv_skew")
c4.metric("IV skew (put−call)", f"{iv * 100:+.1f}%" if iv is not None and iv == iv else "—")

# ---------------------- headline: the divergence -------------------
d = dossier["divergence"]
banner = {"divergent": st.warning, "aligned": st.success, "incomplete": st.info}.get(
    d["state"], st.info
)
banner(f"**{d['headline']}**\n\n{d['detail']}")

# ----------------------------- tabs --------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Positioning", "🌐 Market context", "📰 News (RAG)", "🧠 Crowd emotion"]
)

with tab1:
    if options.get("available"):
        st.write("Aggregated over expiries: " + ", ".join(options["expiries_used"]))
        st.subheader("Unusual activity (today's volume > open interest)")
        ua = options.get("unusual_activity")
        if isinstance(ua, pd.DataFrame) and not ua.empty:
            st.dataframe(ua, use_container_width=True)
        else:
            st.caption("Nothing unusual today.")
    else:
        st.info(options.get("reason", "No options data."))

with tab2:
    if market_pc.get("available"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Market put/call", f"{market_pc['latest']:.2f}")
        m2.metric("vs last year", f"{market_pc['percentile']:.0f}th pct")
        z = market_pc["zscore"]
        m3.metric("Z-score", f"{z:+.2f}" if z == z else "—")
        st.caption(f"CBOE data as of {market_pc['as_of']}. "
                   "High readings = broad hedging / fear across the market.")
    else:
        st.info(market_pc.get("reason", "Market put/call unavailable."))

with tab3:
    st.write(narrative)
    if news:
        with st.expander(f"Sources ({len(news)})"):
            for n in news:
                if n["link"]:
                    st.markdown(f"- [{n['title']}]({n['link']}) — {n['publisher']}")
                else:
                    st.markdown(f"- {n['title']} — {n['publisher']}")

with tab4:
    if social_posts:
        st.caption(f"Analyzed {len(social_posts)} StockTwits posts + {len(news)} news items.")
        # Explicit crowd vote: users tag their own posts Bullish/Bearish.
        tally = bull_bear_tally(social_posts)
        if tally["bull_pct"] is not None:
            t1, t2, t3 = st.columns(3)
            t1.metric("StockTwits 👍 Bullish", tally["bull"])
            t2.metric("StockTwits 👎 Bearish", tally["bear"])
            t3.metric("Bullish share", f"{tally['bull_pct']:.0f}%")
    else:
        st.info("No StockTwits posts returned (the public endpoint may be rate-limited — "
                "try again shortly). Crowd emotion below uses news text in the meantime.")

    if vader is not None:
        st.metric("VADER tone (−1 to +1)", f"{vader:+.2f}")

    if emotion is not None:
        cols = ["fear", "greed", "euphoria", "panic", "fomo", "capitulation", "apathy"]
        vals = [getattr(emotion, c) for c in cols]
        fig = go.Figure(go.Bar(x=cols, y=vals))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Dominant emotion:** {emotion.dominant_emotion}  ·  "
                 f"**Read:** {emotion.bullish_bearish}")
        st.caption(emotion.rationale)
    else:
        st.caption("Emotion model returned nothing — check your API key and try again.")

st.markdown("---")
st.caption("Options Sentinel · sentiment analysis only · not financial advice · "
           "data delayed and may be wrong.")
