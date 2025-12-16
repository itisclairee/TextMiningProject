# backend/rag_pipeline.py

from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_core.documents import Document

from .config import RAGConfig
from .rag_single_agent import single_agent_answer_question
from .rag_multiagent import multiagent_answer_question
from .hybrid_rag import hybrid_answer_question


def answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Public entrypoint used by the Chatbot page.

    Routing strategy:
    - If config.use_hybrid is True:
        → Hybrid RAG (metadata-based logical filtering + vector similarity).
        → Optional extension (not part of core tasks).
    - Else if config.use_multiagent is True:
        → Multi-agent supervisor with specialized agents (Task B).
    - Else:
        → Single-agent RAG over the full corpus (Task A).
    """

    # --- Optional hybrid extension (explicitly non-agentic) ---
    if getattr(config, "use_hybrid", False):
        answer, docs, reasoning, _meta = hybrid_answer_question(
            question, config, show_reasoning
        )
        return answer, docs, reasoning

    # --- Core Task B: multi-agent supervisor ---
    if getattr(config, "use_multiagent", False):
        return multiagent_answer_question(question, config, show_reasoning)

    # --- Core Task A: single-agent ---
    return single_agent_answer_question(question, config, show_reasoning)