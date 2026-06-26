"""Provider-agnostic factory for the chat model and the embeddings model.

Everything else in the app calls get_chat_model() / get_embeddings() and never
cares which provider is behind them. To switch providers, change PROVIDER in
config.py — nothing here or elsewhere needs editing. (That single seam is a small
but real piece of good design worth pointing to.)
"""
from __future__ import annotations

from config import PROVIDER, CHAT_MODEL, GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY


def get_chat_model(temperature: float = 0):
    """Return a LangChain chat model for the selected provider."""
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL, temperature=temperature, google_api_key=GOOGLE_API_KEY
        )
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=CHAT_MODEL, temperature=temperature, api_key=GROQ_API_KEY)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)


def get_embeddings():
    """Return a LangChain embeddings model for the selected provider."""
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # If this model name ever errors, try "models/gemini-embedding-001".
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
        )
    if PROVIDER == "groq":
        # Groq has no embeddings API, so use a small local model (CPU, ~90 MB).
        # Requires: pip install langchain-huggingface sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
