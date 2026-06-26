# Options Sentinel — an Agentic Options-Sentiment Intelligence System

> A LangGraph multi-agent system that fuses **options positioning** (put/call ratios,
> unusual activity, IV skew), **news**, and **Reddit crowd emotion** into a single
> cited, structured "sentiment dossier" for any US-listed ticker.
>
> **This is a research/analysis tool. It produces sentiment intelligence, not buy/sell
> advice or recommendations.**

---

## 1. Problem statement

Retail traders and analysts trying to read market sentiment around a stock face three
disconnected, low-signal sources:

1. **Options positioning** (put/call ratios, open interest, IV skew) is the single most
   honest sentiment gauge because it's what people bet real money on — but it's buried
   in raw option-chain tables that are unreadable without manual aggregation.
2. **News** is voluminous, repetitive, and slow to synthesize. A human can't read 40
   headlines and tell you the *net* narrative in 10 seconds.
3. **Reddit / social chatter** carries the crowd's *emotional* state (FOMO, panic,
   euphoria, capitulation) — the soft signal that often leads or contradicts the
   positioning data — but it's noisy, sarcastic, and impossible to read at scale.

Critically, **no free tool measures these against each other.** The most valuable signal
is not any single source but the **divergence** between hard positioning (put/call ratio)
and soft crowd emotion (Reddit). When the crowd is euphoric while the put/call ratio is
quietly rising (traders buying downside protection), that disagreement is the story.

**The gap:** there is no lightweight, free, explainable system that ingests all three,
reconciles them, and produces a *cited* sentiment read that a human can act on in seconds.

---

## 2. What the product does

**Input:** a ticker symbol (e.g. `NVDA`) and an optional question.

**Output:** a structured **Sentiment Dossier** containing:

- **Options positioning panel** — volume-based and open-interest-based put/call ratios for
  the ticker, vs. the market-wide CBOE put/call ratio for context; IV skew; and a flag for
  unusual options activity (volume >> open interest at specific strikes).
- **News synthesis** — the net narrative across recent headlines, grounded in a RAG index
  so every claim cites a real article.
- **Crowd-emotion index** — Reddit posts/comments classified into an emotion taxonomy
  (fear, greed, euphoria, panic, FOMO, capitulation, apathy), aggregated into a score.
- **The divergence read** — the headline insight: *do the hard signal (P/C) and soft signal
  (emotion) agree or disagree, and what does that historically tend to mean?* Framed as a
  contrarian-vs-momentum lens, **explicitly not a recommendation.**
- **"What would change this read"** — the conditions that would flip the analysis, so it's
  honest about uncertainty.

Everything is cited. Nothing is asserted without a source.

---

## 3. Why this shows technical range

This single project demonstrates, in one coherent system:

| Skill area | How it shows up here |
|---|---|
| **Agent orchestration** | LangGraph `StateGraph` with a supervisor routing to specialist nodes |
| **RAG** | Vector index over news + a glossary of options concepts; retrieval grounds every claim |
| **LLM structured output** | Emotion classification + dossier generation with typed/JSON schemas |
| **Tool use / function calling** | Agents call data tools (options, P/C, Reddit, news) as LangChain tools |
| **Data engineering** | Reconciling 4 messy free APIs with caching, retries, rate-limit handling |
| **Quant/financial literacy** | Put/call ratio math, IV skew, OI vs volume, contrarian theory |
| **Observability** | LangSmith tracing of every node, edge, token, and cost |
| **Product thinking** | A genuine insight (divergence) instead of a feature dump |

---

## 4. Architecture (LangGraph multi-agent)

A typed shared state flows through a graph. A **supervisor** node decides which specialists
to run; specialists write their findings back into state; a **synthesizer** reconciles them;
a **critic** checks for unsupported claims and can loop back for a revision.

```
            ┌──────────────┐
   ticker → │  Supervisor   │ ──── routes / plans
            └──────┬───────┘
       ┌───────────┼───────────┐
       ▼           ▼           ▼
┌────────────┐ ┌──────────┐ ┌──────────────┐
│  Options    │ │  News     │ │  Reddit       │
│  Analyst    │ │  Researcher│ │  Emotion      │
│ (P/C, IV,   │ │ (RAG over │ │  Analyst      │
│  unusual)   │ │  articles)│ │ (emotion tax.)│
└─────┬──────┘ └────┬─────┘ └──────┬───────┘
      └────────────┬┴───────────────┘
                   ▼
            ┌──────────────┐
            │  Synthesizer  │ ── builds dossier + divergence read
            └──────┬───────┘
                   ▼
            ┌──────────────┐      revise?
            │    Critic     │ ──── loops back if claims uncited
            └──────┬───────┘
                   ▼
              Sentiment Dossier
```

Each specialist is a node that reads shared state, calls its tools, and writes results.
Conditional edges (e.g. `critic → synthesizer` if a claim lacks a citation) create the
self-correcting loop that distinguishes this from a linear script.

---

## 5. Tech stack (and why each piece is laptop-light)

| Layer | Choice | Why it's light |
|---|---|---|
| Orchestration | **LangGraph** + **LangChain** | Pure Python control flow; no compute |
| LLM reasoning | **API model** (Anthropic / OpenAI / Groq) | All inference is remote — zero local load |
| Embeddings | API embeddings **or** `all-MiniLM-L6-v2` | MiniLM is ~80 MB, runs on CPU in ms |
| Vector store | **Chroma** (local) or FAISS in-memory | Tiny corpus (dozens of articles), trivial |
| Options data | **yfinance** (`option_chain`) | HTTP only |
| Market P/C | **CBOE free CSV** | Single CSV download |
| News | yfinance news / RSS / a free news API | HTTP only |
| Reddit | **PRAW** (free tier) | HTTP only, cache aggressively |
| Fast sentiment baseline | **VADER** (rule-based) | No model at all |
| UI | **Streamlit** | You already know it |
| Observability | **LangSmith** (free tier) | Hosted; nothing local |

**Rule of thumb:** if a step needs a GPU or a multi-gigabyte download, push it to an API
instead. The laptop's job is glue, not math.

---

## 6. Data sources (all free)

- **Per-ticker put/call ratio** — compute from `yfinance.Ticker(sym).option_chain(expiry)`:
  `put_call_volume_ratio = puts.volume.sum() / calls.volume.sum()` (and the same for
  `openInterest`). Do it per-expiry and aggregated.
- **Market-wide put/call ratio** — CBOE's free daily CSV archive (index + equity P/C),
  downloadable directly; use as the macro backdrop the ticker is read against.
- **News** — yfinance `.news`, or RSS feeds, or a free-tier news API for breadth.
- **Reddit** — PRAW over `r/wallstreetbets`, `r/options`, `r/stocks`, and ticker mentions.
  Free tier ≈ 100 requests/min; **cache everything** and respect the User-Agent rules.
  (Pushshift historical archive is gone, so this is current-window only.)

---

## 7. The analytical core: divergence

This is the part to nail in interviews.

- **Put/call ratio = hard sentiment.** High P/C ⇒ heavy put buying ⇒ bearish/hedging.
  Extremes are classically read *contrarian* (everyone hedged ⇒ little selling left).
- **Reddit emotion = soft sentiment.** Euphoria/FOMO vs panic/capitulation.
- **The signal is the gap.** Build a simple 2×2: (P/C high vs low) × (emotion bullish vs
  bearish). The two *agreeing* is a weak/expected state; the two *disagreeing* is the
  noteworthy one. Surface that disagreement as the dossier's headline, with the historical
  contrarian framing — and always with the explicit caveat that this is sentiment analysis,
  not a prediction or recommendation.

You do **not** need a profitable trading model. You need a *defensible analytical lens*.
That's what makes it a portfolio piece rather than a get-rich scheme.

---

## 8. Build phases (ship something at every step)

**Phase 0 — Data plumbing (no AI yet).**
Functions that return: ticker P/C ratios, CBOE market P/C, recent news list, recent Reddit
posts. Cache each. Prove all four sources work before adding intelligence.

**Phase 1 — Single-shot LLM dossier.**
One LLM call: feed it all four data blobs, ask for a structured dossier. No graph yet.
This is your honest MVP and already demo-able.

**Phase 2 — RAG for news.**
Embed articles into Chroma; retrieve top-k per query; require citations. Now claims are
grounded instead of hallucinated.

**Phase 3 — Emotion taxonomy.**
LLM (or VADER baseline + LLM) classifies Reddit text into the emotion set; aggregate into a
crowd-emotion index. Add the divergence 2×2.

**Phase 4 — LangGraph multi-agent.**
Refactor Phase 1–3 into supervisor + specialist + synthesizer + critic nodes with a typed
state and a critic→synthesizer revision loop. This is the headline technical achievement.

**Phase 5 — Polish.**
Streamlit dossier UI, LangSmith tracing, README with architecture diagram, a few worked
examples (a divergence case is the best demo).

---

## 9. Resume framing (example bullets)

- *Built an agentic options-sentiment system (LangGraph + RAG) that reconciles put/call
  positioning, news, and Reddit crowd-emotion into a cited analytical brief; self-correcting
  critic loop enforces source-grounded claims.*
- *Engineered a 4-source free data pipeline (yfinance options, CBOE put/call archive, news,
  Reddit/PRAW) with caching and rate-limit handling; full LLM inference offloaded to APIs to
  run on any laptop.*
- *Designed a contrarian "signal-divergence" model contrasting hard options positioning
  against soft social emotion, instrumented end-to-end with LangSmith for cost/quality tracing.*

---

## 10. Ethics & honesty (keep these in the product)

- Label every output: **sentiment analysis, not financial advice.**
- Data is delayed and free sources can be wrong — show source + timestamp on every claim.
- The emotion classifier is imperfect on sarcasm; report confidence, don't overclaim.
- Respect Reddit's terms and rate limits; this is non-commercial/research use only.
