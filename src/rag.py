"""Lightweight RAG (Retrieval-Augmented Generation) over the day's news.

We embed each headline into an in-memory vector store, retrieve the most relevant
items, and have the LLM write a CITED net-narrative summary using only those items.
Grounding the summary in retrieved sources is what stops the model from making
things up — the whole point of RAG.

The chat model and embeddings come from src/llm.py, so this works with Gemini,
Groq, or OpenAI unchanged. InMemoryVectorStore needs no database and is instant
for a tiny news corpus.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_chat_model, get_embeddings


def news_narrative(symbol: str, news: list[dict]) -> str:
    """Build a short, source-attributed summary of the net news narrative."""
    if not news:
        return "No recent news found to summarize."

    docs = [
        Document(
            page_content=f"{n['title']}. {n['summary']}",
            metadata={"source": n.get("publisher") or "news", "link": n.get("link", "")},
        )
        for n in news
    ]

    store = InMemoryVectorStore.from_documents(docs, embedding=get_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": min(6, len(docs))})

    def format_docs(retrieved) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('source', 'news')}] {d.page_content}" for d in retrieved
        )

    prompt = ChatPromptTemplate.from_template(
        "You are a financial news analyst. Using ONLY the context, summarize the NET "
        "narrative for {symbol} in 3-4 sentences. State whether coverage skews positive, "
        "negative, or mixed. Attribute claims to the [source] in brackets. If the context "
        "is thin, say so honestly.\n\nContext:\n{context}\n\nNet narrative:"
    )

    chain = (
        {"context": retriever | format_docs, "symbol": lambda _: symbol}
        | prompt
        | get_chat_model(temperature=0)
        | StrOutputParser()
    )

    try:
        return chain.invoke(symbol)
    except Exception as exc:
        return f"News summary unavailable ({exc})."
