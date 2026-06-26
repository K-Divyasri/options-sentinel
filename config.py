"""Central configuration.

Secrets (API keys) are read from a .env file so they never live in source code.
Pick your LLM PROVIDER below — that one line is the only thing you change to
switch between Gemini (free), Groq (free), or OpenAI (paid).
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# ============================================================
#  CHOOSE YOUR PROVIDER HERE  --  "gemini" | "groq" | "openai"
# ============================================================
PROVIDER = "gemini"   # free, no credit card, does chat + embeddings

# ---- Secrets (from .env — only the key for your chosen provider is needed) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "options-sentinel/0.1")

# ---- Chat model name per provider (free-tier-friendly defaults) ----
CHAT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}
CHAT_MODEL = CHAT_MODELS[PROVIDER]


def active_key() -> str:
    """The API key for whichever provider is selected."""
    return {"gemini": GOOGLE_API_KEY, "groq": GROQ_API_KEY, "openai": OPENAI_API_KEY}[PROVIDER]


# ---- Analysis settings ----
N_EXPIRIES = 3                  # how many nearest option expiries to aggregate
NEWS_LIMIT = 15
REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "options", "investing"]
REDDIT_LOOKBACK = "week"        # one of: hour, day, week, month, year, all
REDDIT_LIMIT_PER_SUB = 15

# CBOE's free daily put/call ratio archive (no key needed)
CBOE_PC_URL = (
    "https://cdn.cboe.com/resources/options/"
    "volume_and_call_put_ratios/indexpcarchive.csv"
)
