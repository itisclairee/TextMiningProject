from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from .config import RAGConfig
from .vector_store import load_vector_store
from .llm_provider import LLMBackend


# =====================================================================
# 1. Describe vector databases
# =====================================================================
def _describe_databases(
    db_map: Dict[str, str],
    embedding_model,
) -> Dict[str, str]:
    """
    Build a SHORT semantic description for each vector DB.

    Each DB corresponds to a specialized dataset slice, typically defined by:
    - country
    - legal area (Inheritance / Divorce)
    - document type (civil code, case law, etc.)

    These descriptions are used by:
    - the single-agent for DB selection
    - the multi-agent supervisor for routing decisions
    """
    descriptions: Dict[str, str] = {}

    for db_name, path in db_map.items():
        try:
            vs = load_vector_store(path, embedding_model)
        except Exception:
            descriptions[db_name] = "Database could not be loaded."
            continue

        docs: List[Document] = []
        try:
            if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
                docs = list(vs.docstore._dict.values())[:20]
        except Exception:
            docs = []

        countries = set()
        laws = set()
        types_ = set()

        for d in docs:
            meta = d.metadata or {}
            if "country" in meta:
                countries.add(str(meta["country"]))
            if "law" in meta:
                laws.add(str(meta["law"]))
            if "type" in meta:
                types_.add(str(meta["type"]))

        parts = []
        if countries:
            parts.append("country: " + ", ".join(sorted(countries)))
        if laws:
            parts.append("law: " + ", ".join(sorted(laws)))
        if types_:
            parts.append("content: " + ", ".join(sorted(types_)))

        if parts:
            descriptions[db_name] = "; ".join(parts)
        else:
            descriptions[db_name] = "general legal corpus slice."

    return descriptions


# =====================================================================
# 2. Decide which DBs to query
# =====================================================================
def _decide_which_dbs(
    question: str,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
    llm_backend: LLMBackend,
) -> Tuple[List[str], str]:
    """
    Decide which vector databases are relevant for a given question.

    - If only one DB exists → use it by default.
    - Otherwise, ask the LLM to select relevant DB(s) based on their descriptions.
    - Returns: (chosen_db_names, log_string)
    """
    db_names = list(db_map.keys())
    if len(db_names) == 1:
        return db_names, "Only one DB available → using it by default."

    # Prepare descriptions block for LLM
    lines = []
    for name in db_names:
        desc = db_descriptions.get(name, "no description")
        lines.append(f"- {name}: {desc}")
    db_descr_block = "\n".join(lines)

    system_prompt = (
        "You are a system that selects the most relevant knowledge databases "
        "for a user's legal question. You are given a list of database names "
        "and short descriptions.\n"
        "Return a comma-separated list of database names that should be used "
        "to answer the question. If only one database is relevant, return just that name. "
        "If none are relevant, return 'NONE'."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Available databases:\n{db_descr_block}\n\n"
        "Which database names should be used? Reply with names separated by commas, "
        "or 'NONE'."
    )

    resp = llm_backend.chat(system_prompt, user_prompt).strip()
    resp_lower = resp.lower()

    if "none" in resp_lower:
        return [], f"DB selection: model answered '{resp}' → NONE (no DB)."

    # Parse and keep only valid DB names
    chosen = [name.strip() for name in resp.split(",") if name.strip()]
    chosen_valid = [c for c in chosen if c in db_map]

    if not chosen_valid:
        log = (
            f"DB selection: model answered '{resp}' but no valid DB name was parsed → "
            "falling back to ALL DBs."
        )
        return db_names, log

    log = f"DB selection: model answered '{resp}' → using DBs: " + ", ".join(chosen_valid)
    return chosen_valid, log


# =====================================================================
# 3. Build agent config log
# =====================================================================
def _build_agent_config_log(
    config: RAGConfig,
    db_map: Dict[str, str],
    db_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a compact log describing the agent / DB settings:
      - LLM provider/model
      - Embedding provider/model
      - top_k, agentic_mode, use_multiagent
      - DB names, paths, and optional short descriptions
    """
    lines: List[str] = []

    lines.append(f"LLM provider: {config.llm_provider}")
    lines.append(f"LLM model: {config.llm_model_name}")
    lines.append(f"Embedding provider: {config.embedding_provider}")
    lines.append(f"Embedding model: {config.embedding_model_name}")
    lines.append(f"top_k: {config.top_k}")
    lines.append(f"agentic_mode: {getattr(config, 'agentic_mode', False)}")
    use_multiagent = getattr(config, "use_multiagent", False)
    lines.append(f"use_multiagent: {use_multiagent}")

    lines.append("Vector DBs:")
    for name, path in db_map.items():
        if db_descriptions and name in db_descriptions:
            desc = db_descriptions[name]
            lines.append(f"  - {name}: path={path} | {desc}")
        else:
            lines.append(f"  - {name}: path={path}")

    return "\n".join(lines)