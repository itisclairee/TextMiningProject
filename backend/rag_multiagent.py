# backend/rag_multiagent.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .rag_utils import _build_agent_config_log
from .rag_single_agent import (
    decide_need_retrieval,
    decide_relevant_slices,
    country_gate,
    retrieve_from_db,
    get_supported_countries_from_config,
)

# =====================================================================
# Helpers 
# =====================================================================

def get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    """Ritorna un dizionario {db_name -> path} dalle cartelle dei vector store."""
    return {Path(p).name.lower(): p for p in config.vector_store_dirs}

def describe_databases(db_map: Dict[str, str], embedding_model=None) -> Dict[str, str]:
    """Restituisce descrizioni dei DB (qui solo il nome)."""
    return {db_name: f"Vector DB for {db_name}" for db_name in db_map.keys()}

def decide_which_dbs(
    question: str,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
    llm_backend: Optional[LLMBackend] = None
) -> Tuple[List[str], str]:
    """
    Ritorna la lista di DB scelti. Qui default: tutti i DB.
    """
    chosen = list(db_map.keys())
    routing_log = f"Default selection: all available DBs {chosen}"
    return chosen, routing_log

# =====================================================================
# Sub-agent wrapper con country gating e slices
# =====================================================================

def subagent_answer_question_with_slices(
    question: str,
    db_path: str,
    config: RAGConfig,
    llm: Optional[LLMBackend] = None,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:

    if llm is None:
        llm = LLMBackend(config)

    # ---- Step 1: decide if retrieval is needed ----
    need_retrieval, thought_log = decide_need_retrieval(question, llm)

    retrieved_docs: List[Document] = []
    reasoning_trace: Optional[str] = None
    semantic_log: str = "No retrieval performed."

    if need_retrieval:
        # ---- Step 2: decide relevant slices (countries + content types) ----
        countries, content_types, semantic_log = decide_relevant_slices(question, llm)

        # ---- Step 3: country gate ----
        allowed, gate_log = country_gate(countries, config)
        semantic_log += "\n" + gate_log

        if allowed:
            embedding_model = get_embedding_model(config)
            db_info = {"db_name": Path(db_path).name.lower(), "path": db_path}
            retrieved_docs = retrieve_from_db(question, db_info, config, embedding_model)

    # ---- Step 4: generate answer ----
    context = "\n\n".join(d.page_content for d in retrieved_docs)
    system_prompt = "You are a legal assistant. Use provided legal documents as authoritative sources if present."
    user_prompt = f"Question:\n{question}"
    if context:
        user_prompt += f"\n\nContext:\n{context}"

    answer = llm.chat(system_prompt, user_prompt)

    # ---- Step 5: optional reasoning trace ----
    if show_reasoning:
        supported_countries = sorted(get_supported_countries_from_config(config))
        reasoning_trace = (
            f"**Thought**: {thought_log}\n\n"
            f"**Action**:\n{semantic_log}\n\n"
            f"**Supported countries (from json_folders)**:\n{supported_countries}\n"
        )

    return answer, retrieved_docs, reasoning_trace

# =====================================================================
# Multi-agent pipeline
# =====================================================================

def _multiagent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:

    supervisor_backend = LLMBackend(config)
    db_map = get_vector_db_dirs(config)
    embedding_model = get_embedding_model(config)
    db_descriptions = describe_databases(db_map, embedding_model)

    chosen_db_names, routing_log = decide_which_dbs(
        question=question,
        db_map=db_map,
        db_descriptions=db_descriptions,
        llm_backend=supervisor_backend,
    )

    per_agent_answers: List[Tuple[str, str]] = []
    all_docs: List[Document] = []
    sub_traces: Dict[str, str] = {}

    # ---- Call each selected sub-agent ----
    for db_name in chosen_db_names:
        db_path = db_map[db_name]

        local_cfg = replace(config)
        local_cfg.vector_store_dirs = [db_path]
        local_cfg.vector_store_dir = db_path
        if hasattr(local_cfg, "use_multiagent"):
            local_cfg.use_multiagent = False

        sub_answer, sub_docs, sub_trace = subagent_answer_question_with_slices(
            question,
            db_path,
            local_cfg,
            show_reasoning=True
        )

        per_agent_answers.append((db_name, sub_answer))
        all_docs.extend(sub_docs)
        if sub_trace:
            sub_traces[db_name] = sub_trace

    # ---- Fallback single-agent if nessessario ----
    if not per_agent_answers:
        fallback_answer, fallback_docs, fallback_trace = subagent_answer_question_with_slices(
            question, config.vector_store_dirs[0], config, show_reasoning=show_reasoning
        )
        reasoning_trace = None
        if show_reasoning and fallback_trace:
            reasoning_trace = (
                "**Multi-agent Supervisor**: No specialized agents selected; "
                "falling back to single-agent RAG over all databases.\n\n"
                + fallback_trace
            )
        return fallback_answer, fallback_docs, reasoning_trace

    # ---- Supervisor synthesizes final answer ----
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

    reasoning_trace: Optional[str] = None
    if show_reasoning:
        routing_info = (
            "Supervisor selected the following specialized agents: "
            + ", ".join(f"`{n}`" for n, _ in per_agent_answers)
            + "."
        )
        per_agent_summary_lines = []
        for db_name, ans in per_agent_answers:
            short_ans = ans[:400] + ("..." if len(ans) > 400 else "")
            per_agent_summary_lines.append(
                f"- **Agent `{db_name}`** produced an answer starting with:\n"
                f"  {short_ans}"
            )
        per_agent_summary = "\n".join(per_agent_summary_lines)

        subagent_log_block = ""
        for db_name, trace in sub_traces.items():
            subagent_log_block += f"\n\n[Sub-agent `{db_name}` detailed trace]\n{trace}"

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
            f"**Agent / DB Configuration (Supervisor view)**:\n```text\n{agent_config_log}\n```"
        )

        if subagent_log_block:
            reasoning_trace += "\n\n**Sub-agent Retrieval / Post-Retrieval Logs**:\n" + subagent_log_block

    return final_answer, all_docs, reasoning_trace

# =====================================================================
# Public API
# =====================================================================

def multiagent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return _multiagent_answer_question_core(question, config, show_reasoning)