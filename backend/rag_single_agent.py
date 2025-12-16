# backend/rag_single_agent.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Set
import json
from pathlib import Path
import re
import numpy as np
from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
from .rag_utils import _build_agent_config_log

# =====================================================================
# Helpers
# =====================================================================

def get_supported_countries_from_config(config: RAGConfig) -> Set[str]:
    countries: Set[str] = set()
    for folder in config.json_folders:
        country = Path(folder).name
        if country:
            countries.add(country)
    return countries

# =====================================================================
# Dataset Registry
# =====================================================================

def build_db_registry(config: RAGConfig) -> List[Dict]:
    registry: List[Dict] = []
    for path in config.vector_store_dirs:
        name = Path(path).name.lower()
        registry.append({
            "db_name": name,
            "path": path,
        })
    return registry

# =====================================================================
# Thought: do we need retrieval?
# =====================================================================

def decide_need_retrieval(
    question: str,
    llm: LLMBackend,
) -> Tuple[bool, str]:

    # Parole chiave per forzare retrieval
    legal_keywords = ["art.", "articolo", "codice civile", "divorzio", "legge", "comma"]

    if any(k.lower() in question.lower() for k in legal_keywords):
        return True, "Thought: question contains legal keywords → forcing retrieval."

    resp = llm.chat(
        "Decide if legal documents are required. Answer YES or NO.",
        f"Question:\n{question}",
    ).strip().lower()

    if resp == "yes":
        return True, "Thought: retrieval is necessary."
    if resp == "no":
        return False, "Thought: retrieval is not necessary."

    return True, "Thought: ambiguous → defaulting to retrieval."

# =====================================================================
# Semantic selection
# =====================================================================

def decide_relevant_slices(
    question: str,
    llm: LLMBackend,
) -> Tuple[List[str], List[str], str]:
    system_prompt = (
        "You are a legal domain classifier.\n"
        "Return a JSON object with keys:\n"
        "- countries\n"
        "- content_types\n"
    )

    raw = llm.chat(system_prompt, f"Question:\n{question}")

    try:
        parsed = json.loads(raw)
        countries = parsed.get("countries", [])
        content_types = parsed.get("content_types", [])
    except Exception:
        countries = []
        content_types = []

    log = (
        "Action (semantic selection):\n"
        f"- Countries selected: {countries}\n"
        f"- Content types selected: {content_types}"
    )

    return countries, content_types, log

# =====================================================================
# COUNTRY GATE
# =====================================================================

def country_gate(
    countries: List[str],
    config: RAGConfig,
) -> Tuple[bool, str]:
    supported_countries = get_supported_countries_from_config(config)

    if not countries:
        return False, "❌ Retrieval blocked: no country selected."
    if not supported_countries:
        return False, "❌ Retrieval blocked: no supported countries configured."
    if not supported_countries.intersection(countries):
        return False, f"❌ Retrieval blocked: unsupported country. Supported: {sorted(supported_countries)}."

    return True, "✅ Retrieval allowed: supported country detected."

# =====================================================================
# Similarity filtering con priorità articoli
# =====================================================================

def _similarity_rank_and_filter(
    question: str,
    docs: List[Document],
    embedding_model,
    top_k: int,
    min_sim: float = 0.1,
) -> List[Document]:
    if not docs:
        return []

    # --- Identifico articoli specifici nella query ---
    article_matches = re.findall(r"art\.?\s*(\d+)", question.lower())
    forced_docs = []
    forced_indices = []

    if article_matches:
        for art_num in article_matches:
            for i, d in enumerate(docs):
                codes = d.metadata.get("civil_codes_used", "")
                if isinstance(codes, list):
                    codes_str = " ".join(codes).lower()
                else:
                    codes_str = str(codes).lower()
                # match esatto del numero articolo con "art." o "art"
                if f"art. {art_num}" in codes_str or f"art {art_num}" in codes_str:
                    if i not in forced_indices:
                        forced_docs.append(d)
                        forced_indices.append(i)

    # --- Rimuovo documenti già forzati ---
    remaining_docs = [d for i, d in enumerate(docs) if i not in forced_indices]

    # --- Similarità normale per gli altri documenti ---
    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    if remaining_docs:
        doc_texts = [d.page_content for d in remaining_docs]
        doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype="float32")
        sims = (doc_vecs @ q_vec) / np.maximum(np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q_vec), 1e-8)
    else:
        sims = np.array([])

    final_docs = []

    # --- Prima i documenti forzati ---
    for d in forced_docs:
        d.metadata = d.metadata or {}
        d.metadata["similarity_score"] = 1.0  # priorità massima
        final_docs.append(d)

    # --- Poi i documenti ordinati per similarity ---
    if remaining_docs:
        indices_sorted = sorted(range(len(remaining_docs)), key=lambda i: sims[i], reverse=True)
        for i in indices_sorted:
            if sims[i] >= min_sim:
                remaining_docs[i].metadata = remaining_docs[i].metadata or {}
                remaining_docs[i].metadata["similarity_score"] = float(sims[i])
                final_docs.append(remaining_docs[i])

    return final_docs[:top_k]

# =====================================================================
# Retrieval con similarity filtering
# =====================================================================

def retrieve_from_db(
    question: str,
    db_info: Dict,
    config: RAGConfig,
    embedding_model,
) -> List[Document]:
    store = load_vector_store(db_info["path"], embedding_model)
    retriever = store.as_retriever(search_kwargs={"k": config.top_k * 3})
    raw_docs = retriever.invoke(question)
    return _similarity_rank_and_filter(question, raw_docs, embedding_model, config.top_k)

# =====================================================================
# Context builder
# =====================================================================

def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
    chunks = []
    total = 0
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "")
        db_prefix = f"[DB: {db_name}] " if db_name else ""
        header = f"[DOC {i+1} | {db_prefix}source: {src}]\n"
        piece = header + d.page_content + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)

# =====================================================================
# Main pipeline
# =====================================================================

def single_agent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:

    llm = LLMBackend(config)
    registry = build_db_registry(config)

    need_retrieval, thought_log = decide_need_retrieval(question, llm)
    retrieved_docs: List[Document] = []
    reasoning_trace: Optional[str] = None
    semantic_log = "No retrieval performed."

    if need_retrieval:
        countries, content_types, semantic_log = decide_relevant_slices(question, llm)
        allowed, gate_log = country_gate(countries, config)
        semantic_log += "\n" + gate_log

        if allowed:
            embedding_model = get_embedding_model(config)
            for db in registry:
                retrieved_docs.extend(retrieve_from_db(question, db, config, embedding_model))

    context = _build_context(retrieved_docs)

    system_prompt = (
        "You are a legal assistant. "
        "Use provided legal documents as authoritative sources if present."
    )
    user_prompt = f"Question:\n{question}"
    if context:
        user_prompt += f"\n\nContext:\n{context}"

    answer = llm.chat(system_prompt, user_prompt)

    if show_reasoning:
        supported_countries = sorted(get_supported_countries_from_config(config))
        reasoning_trace = (
            f"**Thought**: {thought_log}\n\n"
            f"**Action**:\n{semantic_log}\n\n"
            f"**Supported countries (from json_folders)**:\n"
            f"{supported_countries}\n\n"
            f"**Agent configuration**:\n"
            f"text\n{_build_agent_config_log(config, {d['db_name']: d['path'] for d in registry})}\n"
        )

    return answer, retrieved_docs, reasoning_trace