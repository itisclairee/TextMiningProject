# backend/hybrid_rag.py

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


# =====================================================================
# 1. Legal metadata schema (law mandatory, others optional)
# =====================================================================

LEGAL_METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cost": {
            "type": "string",
            "description": (
                "Cost associated with the legal procedure. "
                "Return ONLY the amount followed by '€', e.g. '375760 €'. "
                "If unknown, return null."
            ),
        },
        "duration": {
            "type": "string",
            "description": (
                "Duration of the legal procedure. "
                "Return a short phrase using years or months only, "
                "e.g. '2 years', '6 months'. If unknown, return null."
            ),
        },
        "civil_codes_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of civil codes referenced. "
                "Return each as 'Art. <number>', e.g. 'Art. 536'."
            ),
        },
        # 👇 This is the ONLY mandatory field and matches your corpus key
        "law": {
            "type": "string",
            "enum": ["Inheritance", "Divorce"],
            "description": (
                "Main legal area of the case/query. "
                "Use exactly 'Inheritance' for succession/eredità cases, "
                "or 'Divorce' for divorce/separazione cases."
            ),
        },
        # Succession / inheritance specific
        "succession_type": {
            "type": "string",
            "enum": ["testamentary", "legal"],
            "description": (
                "Succession type. 'testamentary' = with a will, "
                "'legal' = without a will. If not inferable, return null."
            ),
        },
        "subject_of_succession": {
            "type": "string",
            "description": (
                "Type of inheritance assets involved: 'real estate', "
                "'bank accounts', 'company shares', etc. Very concise. "
                "If unknown, return null."
            ),
        },
        "testamentary_clauses": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific testamentary clauses mentioned, e.g. legacies, trusts, "
                "disinheritance clauses. Use short phrases. If none, return []."
            ),
        },
        "disputed_issues": {
            "type": "array",
            "items": {"type": "string"},
            "enum": ["validity of will", "division of assets", "legitimacy"],
            "description": (
                "Specific disputes: 'validity of will', 'division of assets', "
                "'legitimacy'. Only use these values. If none, return []."
            ),
        },
        "relationship_between_parties": {
            "type": "string",
            "description": (
                "Relationship between heir and de cuius, e.g. 'spouse', 'child', "
                "'sister'. If unknown, return null."
            ),
        },
        "number_of_persons_involved": {
            "type": "integer",
            "description": (
                "Number of heirs and/or legatees. If unknown, return null."
            ),
        },
        # Divorce / separation specific
        "nature_of_separation": {
            "type": "string",
            "enum": ["Voluntary", "Judicial", "consensual", "contentious"],
            "description": (
                "Nature of separation, e.g. 'Voluntary', 'Judicial', "
                "'consensual', or 'contentious'. If not clear, return null."
            ),
        },
        "presence_of_children": {
            "type": "boolean",
            "description": (
                "True if children are involved (maintenance, custody, visitation), "
                "false if explicitly none, null if not mentioned."
            ),
        },
        "marital_regime": {
            "type": "string",
            "enum": [
                "community of property",
                "separation of property",
                "issues relatively division of common properties",
            ],
            "description": (
                "Marital regime affecting separation/divorce. If not inferable, null."
            ),
        },
        "financial_support": {
            "type": "string",
            "description": (
                "Details about alimony/spousal maintenance. "
                "Return ONLY the amount (if mentioned) followed by '€', e.g. "
                "'1200 €'. If no amount, return null."
            ),
        },
        "duration_of_marriage": {
            "type": "string",
            "description": (
                "Duration of marriage being dissolved. "
                "Short phrase using years or months, e.g. '10 years', '18 months'. "
                "If unknown, return null."
            ),
        },
    },
    # 👉 Only 'law' is mandatory, everything else optional
    "required": ["law"],
    "additionalProperties": False,
}


# =====================================================================
# 2. DB mapping & descriptions
# =====================================================================

def _get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    """
    Returns a mapping: {db_name -> folder_path} based on config.

    - If config.vector_store_dirs exists and is non-empty, use that.
    - Else fall back to a single DB: {basename(vector_store_dir): vector_store_dir}.
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
    Used only for logging and simple heuristic routing.
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
                all_docs = list(vs.docstore._dict.values())
                docs = all_docs[:20]
        except Exception:
            docs = []

        laws = set()
        types_ = set()
        subjects = set()
        other_tags = set()

        for d in docs:
            meta = d.metadata or {}
            if "law" in meta:
                laws.add(str(meta["law"]))
            if "type" in meta:
                types_.add(str(meta["type"]))
            if "subject_of_succession" in meta:
                subjects.add(str(meta["subject_of_succession"]))
            for k in ("domain", "category", "state"):
                if k in meta:
                    other_tags.add(f"{k}={meta[k]}")

        parts = []
        if laws:
            parts.append("law: " + ", ".join(sorted(laws)))
        if types_:
            parts.append("type: " + ", ".join(sorted(types_)))
        if subjects:
            parts.append("subject: " + ", ".join(sorted(subjects)))
        if other_tags:
            parts.append("tags: " + ", ".join(sorted(other_tags)))

        if parts:
            descriptions[db_name] = "; ".join(parts)
        else:
            descriptions[db_name] = "general legal corpus."

    return descriptions


# =====================================================================
# 3. Similarity + context helpers
# =====================================================================

def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
    chunks = []
    total = 0
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "")
        db_prefix = f"[DB: {db_name}] " if db_name else ""
        header = f"[DOC {i+1} | {db_prefix}source: {src}]\n"
        text = d.page_content
        piece = header + text + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _similarity_rank_and_filter(
    question: str,
    docs: List[Document],
    embedding_model,
    top_k: int,
    min_sim: float = 0.1,
) -> Tuple[List[Document], str]:
    """
    Rank docs by cosine similarity and filter below min_sim.
    Returns (filtered_docs, log_string).
    """
    log_lines: List[str] = []

    if not docs:
        log_lines.append("No documents returned from base retriever.")
        return [], "\n".join(log_lines)

    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    doc_texts = [d.page_content for d in docs]
    doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype="float32")

    q_norm = np.linalg.norm(q_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    denom = np.maximum(q_norm * doc_norms, 1e-8)
    sims = (doc_vecs @ q_vec) / denom

    num_raw = len(docs)
    sims_min = float(np.min(sims))
    sims_max = float(np.max(sims))
    sims_mean = float(np.mean(sims))

    indices = [i for i, s in enumerate(sims) if s >= min_sim]
    num_after_threshold = len(indices)

    if not indices:
        log_lines.append(
            f"Similarity filtering: {num_raw} raw docs → 0 kept "
            f"(threshold={min_sim:.3f}, "
            f"sim range=[{sims_min:.3f}, {sims_max:.3f}], mean={sims_mean:.3f})."
        )
        return [], "\n".join(log_lines)

    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    final_docs = [docs[i] for i in indices_sorted]

    sims_kept = sims[indices_sorted]
    sims_kept_min = float(np.min(sims_kept))
    sims_kept_max = float(np.max(sims_kept))
    sims_kept_mean = float(np.mean(sims_kept))

    log_lines.append(
        "Similarity filtering + reranking:\n"
        f"- Raw docs from retriever: {num_raw}\n"
        f"- Docs above threshold {min_sim:.3f}: {num_after_threshold}\n"
        f"- Final top_k={top_k} docs kept: {len(final_docs)}\n"
        f"- Similarity stats (all raw): min={sims_min:.3f}, max={sims_max:.3f}, "
        f"mean={sims_mean:.3f}\n"
        f"- Similarity stats (kept):   min={sims_kept_min:.3f}, max={sims_kept_max:.3f}, "
        f"mean={sims_kept_mean:.3f}"
    )

    return final_docs, "\n".join(log_lines)


# =====================================================================
# 4. LLM-based law classification & metadata extraction 
# =====================================================================

def _classify_law(
    question: str,
    llm_backend: LLMBackend,
) -> Tuple[str, str]:
    """
    Classify the query as 'Inheritance' or 'Divorce' (matching document metadata 'law').

    - First: heuristic keyword classification (fast).
    - Then: if ambiguous, LLM classification.
    - Always returns one of 'Inheritance' or 'Divorce' (mandatory).
    """
    q = question.lower()

    succession_kw = ["succession", "successione", "eredit", "inheritance"]
    divorce_kw = ["divorce", "divorz", "separazione", "separation", "matrimonio"]

    has_succession = any(k in q for k in succession_kw)
    has_divorce = any(k in q for k in divorce_kw)

    heuristic_log = []
    if has_succession:
        heuristic_log.append("Heuristic: succession/eredità keywords detected.")
    if has_divorce:
        heuristic_log.append("Heuristic: divorce/separazione keywords detected.")

    if has_succession and not has_divorce:
        law = "Inheritance"
        return law, "\n".join(heuristic_log) or "Heuristic → Inheritance."
    if has_divorce and not has_succession:
        law = "Divorce"
        return law, "\n".join(heuristic_log) or "Heuristic → Divorce."

    # Ambiguous or no strong heuristic → ask LLM
    system_prompt = (
        "You are a classifier for Italian civil law queries.\n"
        "Given a user question, you MUST decide if it is about succession/inheritance "
        "or about divorce/separation.\n"
        "Return ONLY one of these strings:\n"
        "- 'Inheritance'\n"
        "- 'Divorce'\n"
        "If you are unsure, pick the most plausible."
    )
    user_prompt = f"Question:\n{question}\n\nAnswer with 'Inheritance' or 'Divorce' only."

    resp = llm_backend.chat(system_prompt, user_prompt).strip()
    resp_up = resp.upper()

    if "INHERIT" in resp_up:
        law = "Inheritance"
    elif "DIVOR" in resp_up:
        law = "Divorce"
    else:
        # Fallback: if ambiguous, choose Inheritance by default
        law = "Inheritance"

    log = "\n".join(heuristic_log + [f"LLM law decision: '{resp}' → {law}."])
    return law, log


def _extract_legal_metadata_from_query(
    question: str,
    llm_backend: LLMBackend,
) -> Tuple[Dict[str, Any], str]:
    """
    Use the LLM as a metadata extraction agent, based on LEGAL_METADATA_SCHEMA.

    - law is mandatory and MUST be exactly 'Inheritance' or 'Divorce'.
    - All other fields are optional and can be null/[].
    """
    law_hint, law_class_log = _classify_law(question, llm_backend)

    schema_json = json.dumps(LEGAL_METADATA_SCHEMA, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a legal metadata extraction assistant for Italian civil law cases.\n"
        "Given a natural language user query or case description, you must extract "
        "a concise JSON object that conforms EXACTLY to the following JSON schema:\n\n"
        f"{schema_json}\n\n"
        "Important rules:\n"
        f"- 'law' is MANDATORY and MUST be exactly '{law_hint}'.\n"
        "- Set 'law' in the JSON to this value, unless the text clearly contradicts it.\n"
        "- If a field is not clearly inferable, set it to null (or [] for arrays).\n"
        "- 'cost' and 'financial_support' must be a value followed by '€' if an amount "
        "is mentioned, otherwise null.\n"
        "- 'duration' and 'duration_of_marriage' must use only years or months, "
        "short strings like '2 years', '6 months'.\n"
        "- 'civil_codes_used' must be an array of strings like 'Art. 536'.\n"
        "- OUTPUT: ONLY the JSON object, with no explanation, no markdown."
    )

    user_prompt = f"Text:\n{question}\n\nReturn ONLY the JSON object."

    raw = llm_backend.chat(system_prompt, user_prompt)

    # Default empty structure
    default_meta: Dict[str, Any] = {}
    for k, v in LEGAL_METADATA_SCHEMA["properties"].items():
        t = v.get("type")
        if t == "array":
            default_meta[k] = []
        else:
            default_meta[k] = None
    # law mandatory: set default to law_hint
    default_meta["law"] = law_hint

    try:
        meta = json.loads(raw)
        if not isinstance(meta, dict):
            meta = default_meta
    except Exception:
        meta = default_meta

    # Ensure all keys exist
    for k, v in default_meta.items():
        if k not in meta:
            meta[k] = v

    # Enforce 'law' = law_hint (mandatory)
    meta["law"] = law_hint

    log = (
        "Hybrid legal metadata extracted from query:\n"
        + json.dumps(meta, ensure_ascii=False, indent=2)
        + "\n\nLaw classification log:\n"
        + law_class_log
    )
    return meta, log


def _build_metadata_filter(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a metadata filter for the vector store based on extracted metadata.

    - 'law' → used directly as metadata filter.
    - First civil code (if present) → 'civil_codes_used' filter.
    """
    filt: Dict[str, Any] = {}

    law = meta.get("law")
    if isinstance(law, str) and law:
        filt["law"] = law

    civil_codes = meta.get("civil_codes_used") or []
    if isinstance(civil_codes, list) and civil_codes:
        filt["civil_codes_used"] = civil_codes[0]

    return filt


def _heuristic_db_candidates(
    meta: Dict[str, Any],
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
) -> Tuple[List[str], str]:
    """
    Heuristic DB selection based on 'law' and DB names/descriptions.

    If law = 'Inheritance' → prefer DBs with 'inherit', 'succession', 'successione',
    'eredit' in name/description.

    If law = 'Divorce'     → prefer DBs with 'divorce', 'divorz', 'separat',
    'separazione' in name/description.

    If no match → use ALL DBs.
    """
    law = meta.get("law")
    all_db_names = list(db_map.keys())

    if not isinstance(law, str):
        return all_db_names, "Heuristic DB selection: no 'law' → using all DBs."

    law_lower = law.lower()
    if "inherit" in law_lower:
        keywords = ["inherit", "succession", "successione", "eredit"]
    elif "divorce" in law_lower:
        keywords = ["divorce", "divorz", "separat", "separazione"]
    else:
        return all_db_names, f"Heuristic DB selection: law='{law}' not recognized → using all DBs."

    candidates = []
    log_lines = [
        f"Heuristic DB selection: law='{law}' → searching for keywords {keywords} in DB names/descriptions."
    ]

    for name, path in db_map.items():
        desc = db_descriptions.get(name, "")
        text = (name + " " + desc).lower()
        if any(k in text for k in keywords):
            candidates.append(name)

    if candidates:
        log_lines.append(
            "Heuristic DB selection: matched DBs → " + ", ".join(candidates)
        )
        return candidates, "\n".join(log_lines)

    log_lines.append(
        "Heuristic DB selection: no DB matched law-specific keywords → fallback to all DBs."
    )
    return all_db_names, "\n".join(log_lines)


# =====================================================================
# 5. Retrieval & logs (static filters + similarity, with fallback)
# =====================================================================

def _retrieve_from_db_hybrid(
    question: str,
    db_name: str,
    db_path: str,
    embedding_model,
    top_k: int,
    use_rerank: bool,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Document], str]:
    """
    Retrieve docs from a single FAISS DB combining:
      - metadata_filter (all fields = mandatory + marginal)
      - if that is too strict (len(docs) < top_k), fall back to ONLY mandatory filter:
           -> 'law' (Inheritance / Divorce)
      - optional embedding-based similarity reranking (use_rerank flag)
    """
    log_lines: List[str] = [f"[DB {db_name}] path={db_path}"]

    vector_store = load_vector_store(db_path, embedding_model)
    k_base = max(top_k * 3, top_k)

    # Full filter (mandatory + marginal) from metadata
    full_filter: Dict[str, Any] = metadata_filter or {}

    # Mandatory filter: only "law" if present
    mandatory_filter: Dict[str, Any] = {}
    if "law" in full_filter:
        mandatory_filter["law"] = full_filter["law"]

    def _run_once(
        which: str,
        f: Optional[Dict[str, Any]],
    ) -> Tuple[List[Document], str]:
        """
        Helper to run a single retrieval pass with filter f.
        Returns (docs, log_string).
        """
        local_logs: List[str] = [f"[DB {db_name}] Retrieval phase = {which}"]

        search_kwargs: Dict[str, Any] = {"k": k_base}
        if f:
            search_kwargs["filter"] = f
            local_logs.append(
                f"[DB {db_name}] Using metadata filter: "
                f"{json.dumps(f, ensure_ascii=False)}"
            )
        else:
            local_logs.append(f"[DB {db_name}] No metadata filter used.")

        base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        local_logs.append(
            f"[DB {db_name}] Base retriever k={k_base} (top_k={top_k})."
        )

        raw_docs = base_retriever.invoke(question)
        local_logs.append(
            f"[DB {db_name}] Raw docs from retriever: {len(raw_docs)}"
        )

        if use_rerank:
            local_logs.append(
                f"[DB {db_name}] Similarity reranking ENABLED (use_rerank=True)."
            )
            docs, sim_log = _similarity_rank_and_filter(
                question=question,
                docs=raw_docs,
                embedding_model=embedding_model,
                top_k=top_k,
                min_sim=0.1,
            )
            local_logs.append(sim_log)
        else:
            local_logs.append(
                f"[DB {db_name}] Similarity reranking DISABLED (use_rerank=False); "
                f"using top_k={top_k} raw docs in original order."
            )
            docs = raw_docs[:top_k]

        if not docs:
            local_logs.append(
                f"[DB {db_name}] Result: no docs kept after retrieval/rerank."
            )
        else:
            local_logs.append(
                f"[DB {db_name}] Result: {len(docs)} doc(s) kept for context."
            )

        # Tag docs with db_name
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["db_name"] = db_name

        return docs, "\n".join(local_logs)

    # --- Phase 1: full filter (mandatory + marginal) ---
    docs, log_primary = _run_once("primary (full filter)", full_filter)
    log_lines.append(log_primary)

    # If we didn't reach top_k AND there is a stricter filter than just 'law',
    # run a fallback retrieval using only the mandatory filter.
    used_fallback = False
    if len(docs) < top_k and mandatory_filter and mandatory_filter != full_filter:
        used_fallback = True
        log_lines.append(
            f"[DB {db_name}] Fallback triggered: only {len(docs)} doc(s) "
            f"from full filter (< top_k={top_k}) → retry with mandatory 'law' only."
        )
        docs_fallback, log_fallback = _run_once(
            "fallback (mandatory 'law' only)", mandatory_filter
        )
        log_lines.append(log_fallback)

        # If fallback found something, prefer those docs
        if docs_fallback:
            docs = docs_fallback
        else:
            log_lines.append(
                f"[DB {db_name}] Fallback with mandatory 'law' only "
                "did not find additional docs; keeping primary result."
            )

    elif mandatory_filter and mandatory_filter == full_filter:
        log_lines.append(
            f"[DB {db_name}] Full filter == mandatory ('law' only); "
            "no fallback needed."
        )
    else:
        log_lines.append(
            f"[DB {db_name}] Full filter produced >= top_k docs "
            f"({len(docs)} >= {top_k}); no fallback to mandatory filter."
        )

    # Final summary line
    log_lines.append(
        f"[DB {db_name}] FINAL docs kept for context: {len(docs)} "
        f"(fallback used: {used_fallback})"
    )

    return docs, "\n".join(log_lines)


def _build_observation_text(
    used_db_names: List[str],
    docs: List[Document],
) -> str:
    """
    Purely static summary of what was retrieved (no LLM, no ReAct-style reasoning).
    """
    lines: List[str] = []

    if used_db_names:
        lines.append(
            "Databases used: " + ", ".join(sorted(set(used_db_names)))
        )
    else:
        lines.append("Databases used: none (no DB selected).")

    lines.append(f"Total documents used as context: {len(docs)}")

    max_show = min(len(docs), 5)
    for i in range(max_show):
        d = docs[i]
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "unknown_db")
        snippet = d.page_content[:200].replace("\n", " ").strip()
        lines.append(f"- DOC {i+1} (db={db_name}, source={src}): {snippet}")

    return "\n".join(lines)


def _build_agent_config_log(
    config: RAGConfig,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
) -> str:
    lines: List[str] = []

    lines.append(f"LLM provider: {config.llm_provider}")
    lines.append(f"LLM model: {config.llm_model_name}")
    lines.append(f"Embedding provider: {config.embedding_provider}")
    lines.append(f"Embedding model: {config.embedding_model_name}")
    lines.append(f"top_k: {config.top_k}")
    lines.append(f"use_rerank: {config.use_rerank}")
    lines.append("Hybrid RAG mode: LLM metadata + static filters + vector similarity")
    use_multiagent = getattr(config, "use_multiagent", False)
    lines.append(f"use_multiagent (global config): {use_multiagent}")

    lines.append("Vector DBs:")
    for name, path in db_map.items():
        desc = db_descriptions.get(name, "no description")
        lines.append(f"  - {name}: path={path} | {desc}")

    return "\n".join(lines)


# =====================================================================
# 6. Metadata → compact string for generation
# =====================================================================

def _metadata_to_text(meta: Dict[str, Any]) -> str:
    """
    Turn the extracted metadata into a concise, single string
    that we can inject into the prompt before generation.
    """

    def fmt_value(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, list):
            if not v:
                return "[]"
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    keys_order = [
        "law",                      # mandatory
        "succession_type",
        "subject_of_succession",
        "testamentary_clauses",
        "disputed_issues",
        "relationship_between_parties",
        "number_of_persons_involved",
        "nature_of_separation",
        "presence_of_children",
        "marital_regime",
        "financial_support",
        "duration",
        "duration_of_marriage",
        "cost",
        "civil_codes_used",
    ]

    parts: List[str] = []
    for k in keys_order:
        if k not in meta:
            continue
        v = meta[k]
        # 'law' always included; others only if non-null / non-empty
        if k == "law":
            parts.append(f"{k}: {fmt_value(v)}")
        else:
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            parts.append(f"{k}: {fmt_value(v)}")

    if not parts:
        return "law: null"

    return "; ".join(parts)


# =====================================================================
# 7. Public entrypoint: hybrid legal RAG (LLM metadata, static retrieval)
# =====================================================================

def hybrid_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str], Dict[str, Any]]:
    """
    Hybrid legal RAG (NO ReAct):

    1. LLM-based metadata extraction from query:
        - 'law' (Inheritance / Divorce) → mandatory (matches document metadata).
        - other fields (civil_codes_used, cost, duration, etc.) → optional.

    2. Static filters:
        - 'law'  → used directly as metadata filter.
        - first civil code (if present) → civil_codes_used filter.

    3. Heuristic DB selection (NO LLM here):
        - Choose DBs whose name/description matches 'law' keywords.
        - Fall back to ALL DBs if unclear.

    4. Retrieval:
        - For each chosen DB:
            - FAISS retriever with metadata filter.
            - Optional similarity reranking controlled by config.use_rerank.
            - If full filter (law + marginal) is too strict (len(docs) < top_k),
              fallback to using only the mandatory filter {'law': ...}.
        - Concatenate all docs, build context.

    5. Answer:
        - Single LLM call using:
            - user question
            - stringified metadata injected into the prompt
            - retrieved document context (if any)

    Returns:
      - answer_text
      - retrieved_docs
      - reasoning_trace (logs for UI, not chain-of-thought)
      - metadata_dict (LLM-extracted legal metadata)
    """
    llm_backend = LLMBackend(config)
    embedding_model = get_embedding_model(config)

    db_map = _get_vector_db_dirs(config)
    db_descriptions = _describe_databases(db_map, embedding_model)

    # ---- Step 1: LLM-based metadata from query ('law' mandatory) ----
    meta, metadata_log = _extract_legal_metadata_from_query(
        question, llm_backend
    )
    metadata_filter = _build_metadata_filter(meta)
    metadata_text = _metadata_to_text(meta)  # compact string to inject in prompt

    # ---- Step 2: heuristic DB selection (NO extra LLM) ----
    chosen_db_names, routing_log = _heuristic_db_candidates(
        meta=meta,
        db_map=db_map,
        db_descriptions=db_descriptions,
    )

    # ---- Step 3: hybrid retrieval ----
    all_docs: List[Document] = []
    per_db_logs: Dict[str, str] = {}

    if chosen_db_names:
        for db_name in chosen_db_names:
            db_path = db_map[db_name]
            docs_db, log_db = _retrieve_from_db_hybrid(
                question=question,
                db_name=db_name,
                db_path=db_path,
                embedding_model=embedding_model,
                top_k=config.top_k,
                use_rerank=config.use_rerank,
                metadata_filter=metadata_filter,
            )
            per_db_logs[db_name] = log_db
            all_docs.extend(docs_db)

    context = _build_context(all_docs) if all_docs else ""

    # ---- Step 4: final answer LLM (metadata string + context) ----
    system_prompt = (
        "You are a legal assistant for Italian civil law, focusing on inheritance "
        "(succession) and divorce cases.\n"
        "You receive:\n"
        "1) The user's natural language question.\n"
        "2) A structured metadata string extracted from the question "
        "   (field 'law' is mandatory; other fields may be partial or null).\n"
        "3) Context from retrieved legal documents (if available).\n\n"
        "Use the metadata as a concise summary of the legal scenario, "
        "and treat the retrieved documents as primary factual support. "
        "If there is a conflict, prefer the documents but do not contradict "
        "obvious metadata like 'law'.\n"
        "Do NOT reveal your internal chain-of-thought; just provide a clear final answer."
    )

    user_parts = [
        f"User question:\n{question}",
        "Structured metadata extracted from the user query "
        "(do NOT ignore these; treat them as constraints for classification):\n"
        f"{metadata_text}",
    ]

    if context:
        user_parts.append(
            "Context from retrieved documents (primary legal reference):\n"
            f"{context}"
        )

    user_parts.append(
        "Provide a clear, concise answer, esplicitando se alcune parti "
        "sono solo valutazioni generali e non basate su documenti specifici."
    )

    user_prompt = "\n\n".join(user_parts)
    answer = llm_backend.chat(system_prompt, user_prompt)

    # ---- Reasoning / logs (but NOT ReAct-style) ----
    reasoning_trace: Optional[str] = None
    if show_reasoning:
        observation_str = _build_observation_text(
            used_db_names=chosen_db_names,
            docs=all_docs,
        )

        per_db_log_block = ""
        for db_name, log in per_db_logs.items():
            per_db_log_block += f"\n\n[DB {db_name}]\n{log}"

        retrieval_log_block = (
            f"LLM-based metadata extraction log:\n{metadata_log}\n\n"
            f"DB routing log:\n{routing_log}\n"
            f"{per_db_log_block}"
        ).strip()

        agent_config_log = _build_agent_config_log(
            config=config,
            db_map=db_map,
            db_descriptions=db_descriptions,
        )

        reasoning_trace = (
            "Hybrid legal RAG log (LLM metadata, static retrieval; no ReAct):\n\n"
            f"{observation_str}\n\n"
            f"---\n\n"
            f"Retrieval & filtering details:\n"
            f"```text\n{retrieval_log_block}\n```\n\n"
            f"Configuration:\n"
            f"```text\n{agent_config_log}\n```"
        )

    return answer, all_docs, reasoning_trace, meta
