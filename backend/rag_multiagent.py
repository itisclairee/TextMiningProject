# backend/rag_multiagent.py
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple, Optional, Dict

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .rag_utils import (
    _get_vector_db_dirs,
    _describe_databases,
    _decide_which_dbs,
    _build_agent_config_log,
)
from .rag_single_agent import single_agent_answer_question


def _multiagent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Multi-agent pipeline (tool-calling style):

    - Supervisor LLM chooses which specialized RAG agents (DBs) to call.
    - Each specialized agent runs the single-agent RAG on its own DB.
    - Supervisor synthesizes a final answer from sub-agent answers.
    """
    supervisor_backend = LLMBackend(config)
    db_map = _get_vector_db_dirs(config)  # {db_name -> path}

    # Build descriptions for each DB (used to define "subagent" specializations)
    embedding_model = get_embedding_model(config)
    db_descriptions = _describe_databases(db_map, embedding_model)

    # Decide which DBs / sub-agents to use
    chosen_db_names, routing_log = _decide_which_dbs(
        question=question,
        db_map=db_map,
        db_descriptions=db_descriptions,
        llm_backend=supervisor_backend,
    )

    per_agent_answers: List[Tuple[str, str]] = []
    all_docs: List[Document] = []
    sub_traces: Dict[str, str] = {}

    # Call each selected sub-agent (single-agent RAG restricted to that DB)
    for db_name in chosen_db_names:
        db_path = db_map[db_name]

        local_cfg = replace(config)
        local_cfg.vector_store_dirs = [db_path]
        local_cfg.vector_store_dir = db_path
        # avoid recursion: local agent is single-agent only
        if hasattr(local_cfg, "use_multiagent"):
            local_cfg.use_multiagent = False

        sub_answer, sub_docs, sub_trace = single_agent_answer_question(
            question, local_cfg, show_reasoning=True
        )
        per_agent_answers.append((db_name, sub_answer))
        all_docs.extend(sub_docs)
        if sub_trace:
            sub_traces[db_name] = sub_trace

    # If no sub-agents were chosen or produced answers, fallback to single-agent
    if not per_agent_answers:
        fallback_answer, fallback_docs, fallback_trace = single_agent_answer_question(
            question, config, show_reasoning=show_reasoning
        )
        reasoning_trace = None
        if show_reasoning and fallback_trace:
            reasoning_trace = (
                "**Multi-agent Supervisor**: No specialized agents were selected; "
                "falling back to single-agent RAG over all databases.\n\n"
                + fallback_trace
            )
        return fallback_answer, fallback_docs, reasoning_trace

    # Supervisor synthesizes final answer from sub-agent outputs
    agents_block_lines = []
    for db_name, ans in per_agent_answers:
        header = f"[Agent: {db_name}]"
        agents_block_lines.append(f"{header}\n{ans}\n")
    agents_block = "\n\n".join(agents_block_lines)

    system_prompt = (
        "You are a supervisor agent coordinating several specialized RAG agents.\n"
        "You are given their partial answers to the user's question. "
        "Your job is to synthesize a single, clear, non-redundant answer for the user.\n"
        "If agents disagree, explain the discrepancy briefly, then give your best judgment.\n"
        "Do not mention internal tools or agents; just answer as a single assistant."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Specialized agent answers:\n{agents_block}\n\n"
        "Now provide a single final answer to the user, in your own words."
    )

    final_answer = supervisor_backend.chat(system_prompt, user_prompt)

    # Optional high-level reasoning trace (including agent settings + sub-agent logs)
    reasoning_trace: Optional[str] = None
    if show_reasoning:
        routing_info = (
            "Supervisor selected the following specialized agents: "
            + ", ".join(f"`{n}`" for n, _ in per_agent_answers)
            + "."
        )
        per_agent_summary_lines = []
        for db_name, ans in per_agent_answers:
            short_ans = ans[:400]
            if len(ans) > 400:
                short_ans += "..."
            per_agent_summary_lines.append(
                f"- **Agent `{db_name}`** produced an answer starting with:\n"
                f"  {short_ans}"
            )
        per_agent_summary = "\n".join(per_agent_summary_lines)

        # Sub-agent detailed traces (include their own ReAct + retrieval logs)
        subagent_log_block = ""
        for db_name, trace in sub_traces.items():
            subagent_log_block += (
                f"\n\n[Sub-agent `{db_name}` detailed trace]\n{trace}"
            )

        # Global agent/DB configuration log for the supervisor view
        agent_config_log = _build_agent_config_log(
            config=config,
            db_map=db_map,
            db_descriptions=db_descriptions,
        )

        reasoning_trace = (
            f"**Multi-agent Supervisor Thought**: The supervisor routed the question "
            f"to specialized agents based on the available vector databases.\n\n"
            f"**Routing / Action**: {routing_info}\n\n"
            f"**Sub-agent outputs (summarized)**:\n{per_agent_summary}\n\n"
            f"**Supervisor Routing Log**:\n```text\n{routing_log}\n```\n\n"
            f"**Agent / DB Configuration (Supervisor view)**:\n"
            f"```text\n{agent_config_log}\n```"
        )

        if subagent_log_block:
            reasoning_trace += (
                "\n\n**Sub-agent Retrieval / Post-Retrieval Logs**:\n"
                f"{subagent_log_block}"
            )

    return final_answer, all_docs, reasoning_trace


def multiagent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return _multiagent_answer_question_core(question, config, show_reasoning)
