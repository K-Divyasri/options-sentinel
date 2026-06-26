"""Two layers of sentiment.

1. VADER — a fast, rule-based score in [-1, +1]. Free, instant, no model.
2. LLM emotion classification — the LLM reads a sample of posts and returns a
   STRUCTURED emotion profile (fear/greed/euphoria/...). Using
   `with_structured_output` with a Pydantic schema guarantees typed, validated
   output instead of hoping the model returns clean JSON.

The LLM itself comes from src/llm.py, so this works with Gemini, Groq, or OpenAI
unchanged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.llm import get_chat_model

_vader = SentimentIntensityAnalyzer()


def vader_score(texts: list[str]) -> float | None:
    """Mean VADER compound score across texts, or None if there are none."""
    if not texts:
        return None
    scores = [_vader.polarity_scores(t)["compound"] for t in texts]
    return sum(scores) / len(scores)


class CrowdEmotion(BaseModel):
    """Structured emotion read the LLM must return (validated by Pydantic)."""

    fear: float = Field(ge=0, le=1)
    greed: float = Field(ge=0, le=1)
    euphoria: float = Field(ge=0, le=1)
    panic: float = Field(ge=0, le=1)
    fomo: float = Field(ge=0, le=1)
    capitulation: float = Field(ge=0, le=1)
    apathy: float = Field(ge=0, le=1)
    dominant_emotion: str = Field(description="the single strongest emotion above")
    bullish_bearish: str = Field(description="one of: bullish, bearish, neutral")
    rationale: str = Field(description="one sentence describing the overall vibe")


def classify_emotion(texts: list[str]) -> CrowdEmotion | None:
    """One cheap LLM call that scores the crowd's emotional state over a sample."""
    if not texts:
        return None

    sample = "\n---\n".join(texts[:40])  # cap the sample to keep the call cheap
    llm = get_chat_model(temperature=0).with_structured_output(CrowdEmotion)
    prompt = (
        "You are a market-sentiment analyst. Read these social/news snippets about "
        "a stock and score the CROWD'S emotional state. Each score is a 0-1 intensity. "
        "Be robust to sarcasm, hype, and memes. Snippets:\n\n" + sample
    )
    try:
        return llm.invoke(prompt)
    except Exception:
        return None
