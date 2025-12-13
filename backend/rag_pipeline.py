# backend/rag_pipeline.py
from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_core.documents import Document

from .config import RAGConfig
from .rag_single_agent import single_agent_answer_question
from .rag_multiagent import multiagent_answer_question


def answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Public entrypoint used by the Chatbot page.

    - If config.use_multiagent is False → single-agent RAG (previous behavior).
    - If config.use_multiagent is True  → multi-agent supervisor pipeline.
    """
    if getattr(config, "use_multiagent", False):
        return multiagent_answer_question(question, config, show_reasoning)

    return single_agent_answer_question(question, config, show_reasoning)
