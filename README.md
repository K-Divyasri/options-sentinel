# 🛰️ Options Sentinel

An options-sentiment intelligence app that contrasts **hard** sentiment (put/call
positioning) against **soft** sentiment (news + Reddit crowd emotion) for any
US-listed ticker — and flags when the two **diverge**.

> **Sentiment analysis only. Not financial advice.** Data is delayed and can be wrong.

## What it does

- **Options positioning** — put/call ratio by volume and by open interest, an IV-skew
  proxy, and unusual-activity detection (volume > open interest) — via `yfinance`.
- **Market context** — the market-wide CBOE put/call ratio, placed in historical
  context with a **percentile** and **z-score** vs the trailing year.
- **News (RAG)** — recent headlines embedded into a vector store; the LLM writes a
  **source-cited** net-narrative summary grounded only in retrieved articles.
- **Crowd emotion** — a fast **VADER** baseline plus an **LLM emotion classifier**
  (fear / greed / euphoria / panic / FOMO / capitulation / apathy) with typed,
  schema-validated output.
- **The divergence read** — the headline insight: do the bet and the mood agree?

## Tech stack

Python · Streamlit · yfinance · pandas/numpy · LangChain (RAG + structured LLM output)
· VADER · PRAW (Reddit) · Plotly. LLM is provider-agnostic (Gemini / Groq / OpenAI),
selected by one line in `config.py`. All inference runs via API, so it stays light on
any laptop.

## Project structure

```
options-sentinel/
├── app.py              # Streamlit UI
├── config.py           # settings + provider switch + loads secrets from .env
├── requirements.txt
├── .env.example        # template for your key
└── src/
    ├── __init__.py
    ├── llm.py          # provider-agnostic chat + embeddings factory
    ├── options_data.py # put/call ratios, IV skew, unusual activity
    ├── market_pc.py    # CBOE market put/call + percentile/z-score
    ├── news.py         # news fetch
    ├── reddit_data.py  # Reddit fetch (optional)
    ├── sentiment.py    # VADER + LLM emotion classification
    ├── rag.py          # RAG over news
    └── analysis.py     # divergence logic + dossier assembly
```

## Setup & run

```bash
# 1. virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. secrets — copy the template, then paste your key into .env
cp .env.example .env              # Windows: copy .env.example .env

# 4. run
streamlit run app.py
```

Opens at http://localhost:8501.

### Keys

- **Gemini (default, free, no credit card):** https://aistudio.google.com/apikey —
  put it in `.env` as `GOOGLE_API_KEY=...`. Does both chat and embeddings.
- **Reddit (optional):** https://www.reddit.com/prefs/apps → create a "script" app for a
  client id + secret. Without it, crowd emotion runs on news text only.

To switch the LLM provider, change `PROVIDER` in `config.py` to `"groq"` or `"openai"`
(see comments in `requirements.txt` for the extra package each needs).

## What this project demonstrates (for a resume)

- **Modular, production-style Python** — `src/` package, type hints, docstrings, clean
  separation of data / analysis / UI, and a provider-agnostic model layer.
- **Secrets management** — keys in `.env`, never in source; `.gitignore` enforces it.
- **Quant rigor** — put/call ratios, IV skew, and statistical context (percentile,
  z-score) rather than raw numbers.
- **Modern AI engineering** — RAG with source grounding, and LLM **structured output**
  validated by a Pydantic schema.
- **Resilient data engineering** — four free, flaky sources wrapped with caching and
  graceful degradation.

## Honesty & limitations

- This is **not** a trading signal. The value is the analytical lens (divergence).
- Free data is delayed and occasionally wrong; sources + timestamps are shown.
- The emotion classifier can misread sarcasm; it reports a vibe, not ground truth.
- Reddit access is non-commercial / research use, within free-tier rate limits.

## Roadmap

Refactor the pipeline into a **LangGraph** multi-agent graph (supervisor → options /
news / emotion specialists → synthesizer → critic) with LangSmith tracing — see
`PROJECT_SPEC.md`.
