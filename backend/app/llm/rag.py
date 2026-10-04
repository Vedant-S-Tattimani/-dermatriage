"""
Optional RAG (Retrieval-Augmented Generation) module.

Retrieves relevant snippets from a curated medical knowledge base to
supplement the LLM explanation with evidence-based context.

NOTE: This is a lightweight in-process implementation using a simple
TF-IDF index. For production, replace with a vector database
(e.g., Pinecone, Weaviate, or pgvector).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Static knowledge base (bundled with the app) ──────────────────────────────
_KB_PATH = Path(__file__).parent / "knowledge_base.json"

_KNOWLEDGE_BASE: list[dict] | None = None


def _load_kb() -> list[dict]:
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE is None:
        if _KB_PATH.exists():
            with open(_KB_PATH, encoding="utf-8") as f:
                _KNOWLEDGE_BASE = json.load(f)
            logger.info("RAG knowledge base loaded: %d entries", len(_KNOWLEDGE_BASE))
        else:
            logger.warning("RAG knowledge base not found at %s — RAG disabled.", _KB_PATH)
            _KNOWLEDGE_BASE = []
    return _KNOWLEDGE_BASE


def retrieve_context(class_name: str, top_k: int = 2) -> str:
    """
    Retrieve the most relevant knowledge-base snippets for *class_name*.

    Returns a formatted string ready for injection into the LLM prompt,
    or an empty string if the knowledge base is unavailable.
    """
    kb = _load_kb()
    if not kb:
        return ""

    # Simple keyword match (replace with vector similarity in production)
    relevant = [
        entry for entry in kb
        if class_name.lower() in entry.get("title", "").lower()
        or class_name.lower() in entry.get("tags", [])
    ][:top_k]

    if not relevant:
        return ""

    lines = ["## Relevant Medical Context\n"]
    for entry in relevant:
        lines.append(f"**{entry['title']}**: {entry['summary']}")
    return "\n".join(lines)
