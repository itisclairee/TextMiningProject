from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
from .rag_utils import _build_agent_config_log


# =====================================================================
# 1. Legal metadata schema
# =====================================================================

LEGAL_METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cost": {"type": "string"},
        "duration": {"type": "string"},
        "civil_codes_used": {"type": "array", "items": {"type": "string"}},
        "law": {"type": "string", "enum": ["Inheritance", "Divorce"]},
        "succession_type": {"type": "string", "enum": ["testamentary", "legal"]},
        "subject_of_succession": {"type": "string"},
        "testamentary_clauses": {"type": "array", "items": {"type": "string"}},
        "disputed_issues": {"type": "array", "items": {"type": "string"}},
        "relationship_between_parties": {"type": "string"},
        "number_of_persons_involved": {"type": "integer"},
        "nature_of_separation": {"type": "string", "enum": ["Voluntary", "Judicial", "consensual", "contentious"]},
        "presence_of_children": {"type": "boolean"},
        "marital_regime": {"type": "string"},
        "financial_support": {"type": "string"},
        "duration_of_marriage": {"type": "string"},
    },
    "required": ["law"],
    "additionalProperties": False,
}


# =====================================================================
# 2. DB mapping
# =====================================================================

def _get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    dirs: List[str] = getattr(config, "vector_store_dirs", []) or [config.vector_store_dir]
    db_map: Dict[str, str] = {}
    for path in dirs:
        name = os.path.basename(os.path.normpath(path)) or path
        db_map[name] = path
    return db_map


def _describe_databases(db_map: Dict[str, str], embedding_model) -> Dict[str, str]:
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
                all_docs = list(vs.docstore._dict.values())
                docs = all_docs[:20]
        except Exception:
            docs = []

        laws, types, subjects = set(), set(), set()
        for d in docs:
            meta = d.metadata or {}
            if "law" in meta: laws.add(str(meta["law"]))
            if "type" in meta: types.add(str(meta["type"]))
            if "subject_of_succession" in meta: subjects.add(str(meta["subject_of_succession"]))

        parts = []
        if laws: parts.append("law: " + ", ".join(sorted(laws)))
        if types: parts.append("type: " + ", ".join(sorted(types)))
        if subjects: parts.append("subject: " + ", ".join(sorted(subjects)))

        descriptions[db_name] = "; ".join(parts) if parts else "general legal corpus."
    return descriptions


# =====================================================================
# 3. Metadata extraction
# =====================================================================

def _classify_law(question: str, llm_backend: LLMBackend) -> Tuple[str, str]:
    q = question.lower()
    succession_kw = ["succession", "successione", "eredit", "inheritance"]
    divorce_kw = ["divorce", "divorz", "separazione", "separation", "matrimonio"]

    has_succession = any(k in q for k in succession_kw)
    has_divorce = any(k in q for k in divorce_kw)
    heuristic_log = []

    if has_succession:
        heuristic_log.append("Heuristic: succession keywords detected.")
    if has_divorce:
        heuristic_log.append("Heuristic: divorce keywords detected.")

    if has_succession and not has_divorce:
        return "Inheritance", "\n".join(heuristic_log)
    if has_divorce and not has_succession:
        return "Divorce", "\n".join(heuristic_log)

    # fallback LLM
    system_prompt = "Classify query as 'Inheritance' or 'Divorce'. Return only one."
    user_prompt = f"Question:\n{question}"
    resp = llm_backend.chat(system_prompt, user_prompt).strip().upper()
    law = "Inheritance" if "INHERIT" in resp else "Divorce"
    log = "\n".join(heuristic_log + [f"LLM law decision: '{resp}' → {law}"])
    return law, log


def _extract_legal_metadata_from_query(question: str, llm_backend: LLMBackend) -> Tuple[Dict[str, Any], str]:
    law_hint, law_class_log = _classify_law(question, llm_backend)
    default_meta: Dict[str, Any] = {k: ([] if v.get("type")=="array" else None) for k,v in LEGAL_METADATA_SCHEMA["properties"].items()}
    default_meta["law"] = law_hint

    schema_json = json.dumps(LEGAL_METADATA_SCHEMA, ensure_ascii=False)
    system_prompt = f"Extract legal metadata from the query according to schema:\n{schema_json}\nSet 'law'={law_hint}."
    user_prompt = f"Question:\n{question}\nReturn only JSON."
    raw = llm_backend.chat(system_prompt, user_prompt)
    try:
        meta = json.loads(raw)
    except Exception:
        meta = default_meta

    for k in default_meta:
        meta.setdefault(k, default_meta[k])
    meta["law"] = law_hint
    log = f"Metadata extracted:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\nLLM law log:\n{law_class_log}"
    return meta, log


def _build_metadata_filter(meta: Dict[str, Any]) -> Dict[str, Any]:
    filt: Dict[str, Any] = {}
    if meta.get("law"):
        filt["law"] = meta["law"]
    civil_codes = meta.get("civil_codes_used") or []
    if civil_codes:
        filt["civil_codes_used"] = civil_codes[0]
    return filt


# =====================================================================
# 4. Similarity ranking
# =====================================================================

def _similarity_rank_and_filter(question: str, docs: List[Document], embedding_model, top_k: int, min_sim: float=0.1) -> Tuple[List[Document], str]:
    if not docs:
        return [], "No documents returned."
    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    doc_vecs = np.array(embedding_model.embed_documents([d.page_content for d in docs]), dtype="float32")
    sims = (doc_vecs @ q_vec) / (np.linalg.norm(doc_vecs, axis=1)*np.linalg.norm(q_vec) + 1e-8)
    indices = [i for i,s in enumerate(sims) if s>=min_sim]
    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    filtered_docs = [docs[i] for i in indices_sorted]
    return filtered_docs, f"Similarity rank: kept {len(filtered_docs)} of {len(docs)} docs."


# =====================================================================
# 5. Hybrid retrieval per DB
# =====================================================================

def _retrieve_from_db_hybrid(question: str, db_name: str, db_path: str, embedding_model, top_k: int, use_rerank: bool, metadata_filter: Optional[Dict[str, Any]]=None) -> Tuple[List[Document], str]:
    log_lines: List[str] = [f"[DB {db_name}] path={db_path}"]
    vector_store = load_vector_store(db_path, embedding_model)
    k_base = max(top_k*3, top_k)
    search_kwargs: Dict[str, Any] = {"k": k_base}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    raw_docs = retriever.invoke(question)
    log_lines.append(f"Raw docs: {len(raw_docs)}")

    if use_rerank:
        docs, sim_log = _similarity_rank_and_filter(question, raw_docs, embedding_model, top_k)
        log_lines.append(sim_log)
    else:
        docs = raw_docs[:top_k]

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["db_name"] = db_name
    log_lines.append(f"Docs kept: {len(docs)}")
    return docs, "\n".join(log_lines)


def _build_context(docs: List[Document], max_chars: int=4000) -> str:
    chunks = []
    total = 0
    for i,d in enumerate(docs):
        text = d.page_content
        header = f"[DOC {i+1} | source={d.metadata.get('source','unknown')}] "
        piece = header + text + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _metadata_to_text(meta: Dict[str, Any]) -> str:
    def fmt(v):
        if v is None:
            return "null"
        if isinstance(v,list):
            return "[" + ", ".join(v) + "]"
        return str(v)
    return "; ".join(f"{k}: {fmt(v)}" for k,v in meta.items() if v not in (None, []))


# =====================================================================
# 6. Public entrypoint
# =====================================================================

def hybrid_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str], Dict[str, Any]]:

    llm_backend = LLMBackend(config)
    embedding_model = get_embedding_model(config)
    db_map = _get_vector_db_dirs(config)
    db_descriptions = _describe_databases(db_map, embedding_model)

    # Step 1: metadata from query
    query_meta, metadata_log = _extract_legal_metadata_from_query(question, llm_backend)
    metadata_filter = _build_metadata_filter(query_meta)
    metadata_text = _metadata_to_text(query_meta)

    # Step 2: retrieve from DBs
    all_docs: List[Document] = []
    per_db_logs: Dict[str,str] = {}
    for db_name, db_path in db_map.items():
        docs, log_db = _retrieve_from_db_hybrid(
            question, db_name, db_path, embedding_model, config.top_k, config.use_rerank, metadata_filter
        )
        per_db_logs[db_name] = log_db
        all_docs.extend(docs)

    # Step 3: merge document metadata
    merged_meta = query_meta.copy()
    for d in all_docs:
        doc_meta = d.metadata or {}
        for k,v in doc_meta.items():
            if k in merged_meta and (merged_meta[k] is None or (isinstance(merged_meta[k],list) and not merged_meta[k])):
                merged_meta[k] = v

    merged_text = _metadata_to_text(merged_meta)
    context = _build_context(all_docs)

    # Step 4: LLM answer
    system_prompt = "You are a legal assistant for Italian civil law. Use metadata and documents as source."
    user_prompt = f"Question:\n{question}\nMetadata:\n{merged_text}\nContext:\n{context}"
    answer = llm_backend.chat(system_prompt, user_prompt)

    reasoning_trace = None
    if show_reasoning:
        per_db_log_block = "\n\n".join(f"[DB {k}]\n{v}" for k,v in per_db_logs.items())
        agent_config_log = _build_agent_config_log(config, db_map, db_descriptions)
        reasoning_trace = f"Metadata log:\n{metadata_log}\n\nPer-DB logs:\n{per_db_log_block}\n\nAgent config:\n{agent_config_log}"

    return answer, all_docs, reasoning_trace, merged_meta