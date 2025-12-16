# backend/rag_single_agent.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Set
import json
from pathlib import Path

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

def decide_need_retrieval(question: str, llm: LLMBackend) -> Tuple[bool, str]:
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

def decide_relevant_slices(question: str, llm: LLMBackend) -> Tuple[List[str], List[str], str]:
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
# Country gate
# =====================================================================

def country_gate(countries: List[str], config: RAGConfig) -> Tuple[bool, str]:
    supported_countries = get_supported_countries_from_config(config)
    if not countries:
        return False, "❌ Retrieval blocked: no country selected."
    if not supported_countries:
        return False, "❌ Retrieval blocked: no supported countries configured."
    if not supported_countries.intersection(countries):
        return False, f"❌ Retrieval blocked: unsupported country. Supported countries are {sorted(supported_countries)}."
    return True, "✅ Retrieval allowed: supported country detected."

# =====================================================================
# Retrieval with metadata filtering
# =====================================================================

def retrieve_from_db(
    question: str,
    db_info: Dict,
    config: RAGConfig,
    embedding_model,
    country_filter: str | None = None,
    law_filter: str | None = None,
) -> List[Document]:
    """
    Retrieve documents from FAISS with optional country/law filtering.
    """
    store = load_vector_store(db_info["path"], embedding_model)
    retriever = store.as_retriever(search_kwargs={"k": config.top_k})
    retrieved_docs: List[Document] = retriever.invoke(question)

    filtered_docs = []
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        if country_filter and meta.get("state", "").lower() != country_filter.lower():
            continue
        if law_filter and law_filter.lower() not in str(meta.get("law", "")).lower():
            continue
        filtered_docs.append(doc)

    return filtered_docs

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

    # ---- Thought ----
    need_retrieval, thought_log = decide_need_retrieval(question, llm)

    retrieved_docs: List[Document] = []
    reasoning_trace: Optional[str] = None
    semantic_log = "No retrieval performed."

    # ---- Action ----
    if need_retrieval:
        countries, content_types, semantic_log = decide_relevant_slices(question, llm)
        allowed, gate_log = country_gate(countries, config)
        semantic_log += "\n" + gate_log

        if allowed:
            embedding_model = get_embedding_model(config)

            # Decidi law_filter dai content_types se presente
            law_filter = None
            for ct in content_types:
                if "divorce" in ct.lower():
                    law_filter = "Divorce"
                elif "civil" in ct.lower():
                    law_filter = "Civil Law"

            # Recupera documenti filtrando per country/law
            for db in registry:
                for country in countries:
                    retrieved_docs.extend(
                        retrieve_from_db(
                            question,
                            db,
                            config,
                            embedding_model,
                            country_filter=country,
                            law_filter=law_filter,
                        )
                    )

    # ---- Answer ----
    context = "\n\n".join(d.page_content for d in retrieved_docs)
    system_prompt = (
        "You are a legal assistant. "
        "Use provided legal documents as authoritative sources if present."
    )
    user_prompt = f"Question:\n{question}"
    if context:
        user_prompt += f"\n\nContext:\n{context}"
    answer = llm.chat(system_prompt, user_prompt)

    # ---- Optional reasoning trace ----
    if show_reasoning:
        supported_countries = sorted(get_supported_countries_from_config(config))
        reasoning_trace = (
            f"**Thought**: {thought_log}\n\n"
            f"**Action**:\n{semantic_log}\n\n"
            f"**Supported countries (from json_folders)**:\n{supported_countries}\n\n"
            f"**Agent configuration**:\n"
            f"text\n{_build_agent_config_log(config, {d['db_name']: d['path'] for d in registry})}\n"
        )

    return answer, retrieved_docs, reasoning_trace