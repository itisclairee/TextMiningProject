# backend/rag_utils.py
from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document

from .config import RAGConfig
from .llm_provider import LLMBackend
from .vector_store import load_vector_store


def _get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    """
    Returns a mapping: {db_name -> folder_path}

    - If config.vector_store_dirs exists and is non-empty, use that.
    - Else fall back to single db: {<basename(vector_store_dir)>: vector_store_dir}.
    """
    dirs: List[str] = []

    if getattr(config, "vector_store_dirs", None):
        v = config.vector_store_dirs or []
        if isinstance(v, list) and len(v) > 0:
            dirs.extend(v)

    if not dirs:
        dirs.append(config.vector_store_dir)

    db_map: Dict[str, str] = {}
    for path in dirs:
        name = os.path.basename(os.path.normpath(path)) or path
        db_map[name] = path
    return db_map


def _describe_databases(
    db_map: Dict[str, str],
    embedding_model,
) -> Dict[str, str]:
    """
    Build a SHORT description for each DB based on a few documents' metadata.
    Used so the LLM (single agent or supervisor) can choose the right DB(s).
    """
    descriptions: Dict[str, str] = {}

    for db_name, path in db_map.items():
        try:
            vs = load_vector_store(path, embedding_model)
        except Exception:
            descriptions[db_name] = "Database could not be loaded."
            continue

# If the vector store vs has a docstore with an internal _dict,
# take all the Document objects from that internal dictionary
# and put them into a list called all_docs. Then take the first 20 documents
# from all_docs and put them into a list called docs. If any error occurs   

        docs: List[Document] = []
        try:
            if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
                all_docs = list(vs.docstore._dict.values())
                docs = all_docs[:20]
        except Exception:
            docs = []

        corpus_names = set()
        laws = set()
        types_ = set()
        other_tags = set()

        for d in docs:
            meta = d.metadata or {}
            if "law" in meta:
                laws.add(str(meta["law"]))
            if "type" in meta:
                types_.add(str(meta["type"]))
            for k in ("presence_of_children", "disputed_issues", "subject_of_succession", "financial_support", "subject_of_succession"):
                if k in meta:
                    other_tags.add(f"{k}={meta[k]}")

        parts = []
        if corpus_names:
            parts.append("corpus: " + ", ".join(sorted(corpus_names)))
        if laws:
            parts.append("law: " + ", ".join(sorted(laws)))
        if types_:
            parts.append("type: " + ", ".join(sorted(types_)))
        if other_tags:
            parts.append("tags: " + ", ".join(sorted(other_tags)))

        if parts:
            descriptions[db_name] = "; ".join(parts)
        else:
            descriptions[db_name] = "general corpus with mixed content."

    return descriptions


def _decide_which_dbs(
    question: str,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
    llm_backend: LLMBackend,
) -> Tuple[List[str], str]:
    db_names = list(db_map.keys())
    if len(db_names) == 1:
        return db_names, "Only one DB available → using it by default."

    lines = []
    for name in db_names:
        desc = db_descriptions.get(name, "no description")
        lines.append(f"- {name}: {desc}")
    db_descr_block = "\n".join(lines)

    system_prompt = (
        "You are selecting which knowledge databases are relevant for a user question.\n"
        "You are given a list of database names with short descriptions.\n"
        "Return a comma-separated list of the database names that should be used "
        "to answer the question. If only one database is relevant, return just that name. "
        "If multiple are relevant, include all of them. If none are relevant, return 'NONE'."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Available databases:\n{db_descr_block}\n\n"
        "Which database names should be used? Reply with names separated by commas, "
        "or 'NONE'."
    )

    resp = llm_backend.chat(system_prompt, user_prompt).strip()
    resp_lower = resp.lower()

    if "none" in resp_lower:
        return [], f"DB selection: model answered '{resp}' → NONE (no DB)."

    chosen = [name.strip() for name in resp.split(",") if name.strip()]
    chosen_valid = [c for c in chosen if c in db_map]

    if not chosen_valid:
        log = (
            f"DB selection: model answered '{resp}' but no valid DB name was parsed → "
            "falling back to ALL DBs."
        )
        return db_names, log

    log = (
        f"DB selection: model answered '{resp}' → using DBs: "
        + ", ".join(chosen_valid)
    )
    return chosen_valid, log


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
    lines.append(f"agentic_mode: {config.agentic_mode}")
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
