# backend/rag_single_agent.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
from .rag_utils import (
    _get_vector_db_dirs,
    _describe_databases,
    _decide_which_dbs,
    _build_agent_config_log,
)


# =====================================================================
# Context builder + similarity filtering (with logging)
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


def _retrieve_documents_from_db(
    question: str,
    config: RAGConfig,
    embedding_model,
    db_name: str,
    db_path: str,
) -> Tuple[List[Document], str]:
    """
    Retrieve docs from a single FAISS DB at db_path, single-query only.
    Returns (docs_kept, log_string).
    """
    log_lines: List[str] = [f"[DB {db_name}] path={db_path}"]

    vector_store = load_vector_store(db_path, embedding_model)

    k_base = max(config.top_k * 3, config.top_k)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k_base})
    log_lines.append(f"[DB {db_name}] Base retriever k={k_base} (top_k={config.top_k}).")

    log_lines.append(f"[DB {db_name}] Multi-query retrieval DISABLED.")
    raw_docs = base_retriever.invoke(question)

    log_lines.append(f"[DB {db_name}] Raw docs from retriever: {len(raw_docs)}")

    docs, sim_log = _similarity_rank_and_filter(
        question=question,
        docs=raw_docs,
        embedding_model=embedding_model,
        top_k=config.top_k,
        min_sim=0.1,
    )
    log_lines.append(sim_log)

    if not docs:
        log_lines.append(f"[DB {db_name}] Result: no docs kept after filtering.")
    else:
        log_lines.append(f"[DB {db_name}] Result: {len(docs)} doc(s) kept for context.")

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["db_name"] = db_name

    return docs, "\n".join(log_lines)


# =====================================================================
# Agentic decision: do we need retrieval?
# =====================================================================
def _decide_need_retrieval(
    question: str,
    config: RAGConfig,
    llm_backend: LLMBackend,
) -> Tuple[bool, str]:
    system_prompt = (
        "You are a classifier that decides if a question needs external documents "
        "to answer accurately.\n"
        "Reply with a single word:\n"
        "- 'YES' if external documents or context WOULD help or are needed.\n"
        "- 'NO' if the question can be answered reliably from general knowledge."
    )
    user_prompt = f"Question:\n{question}\n\nAnswer YES or NO only."

    resp = llm_backend.chat(system_prompt, user_prompt).strip().lower()

    if "yes" in resp and "no" not in resp:
        return True, f"Retrieval decision: model answered '{resp}' → USE retrieval."
    if "no" in resp and "yes" not in resp:
        return False, f"Retrieval decision: model answered '{resp}' → NO retrieval."

    return True, f"Retrieval decision: ambiguous answer '{resp}' → default to USE retrieval."


# =====================================================================
# Helper: summarized Observation text (using content + LLM)
# =====================================================================
def _build_observation_text(
    question: str,
    need_retrieval: bool,
    used_db_names: List[str],
    docs: List[Document],
    llm_backend: LLMBackend,
) -> str:
    if not need_retrieval:
        return "No external vector databases were used; the answer relies on internal knowledge."

    if need_retrieval and not docs:
        if used_db_names:
            db_list = ", ".join(sorted(set(used_db_names)))
            return (
                f"Retrieval was attempted on databases: {db_list}, "
                "but no sufficiently relevant documents were found."
            )
        return "Retrieval was attempted, but no sufficiently relevant documents were found."

    doc_blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "unknown_db")
        snippet = d.page_content[:400].replace("\n", " ").strip()
        doc_blocks.append(
            f"[DOC {i}] db={db_name} | source={src}\n"
            f"Snippet: {snippet}\n"
        )

    docs_text = "\n\n".join(doc_blocks)
    db_list = ", ".join(sorted(set(used_db_names))) if used_db_names else "unknown"

    system_prompt = (
        "You are summarizing how retrieved documents from one or more vector databases "
        "help answer a user's question.\n"
        "You MUST be concise and high-level. Do NOT reveal detailed chain-of-thought.\n"
        "Your job:\n"
        "- Mention briefly WHICH databases were used.\n"
        "- In 2–4 bullet points, explain at a high level how the retrieved content "
        "is useful or relevant for answering the question.\n"
        "- Keep the explanation short."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Databases used: {db_list}\n\n"
        f"Retrieved documents (summarized):\n{docs_text}\n\n"
        "Now produce a SHORT observation in this format:\n\n"
        "Databases used: <comma-separated list>\n"
        "- bullet point 1\n"
        "- bullet point 2\n"
        "- (optional) bullet point 3\n"
    )

    explanation = llm_backend.chat(system_prompt, user_prompt)
    return explanation


# =====================================================================
# SINGLE-AGENT CORE (ReAct-style)
# =====================================================================
def _single_agent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """
    Original ReAct-style single-agent RAG pipeline (no multi-agent supervisor).
    """
    llm_backend = LLMBackend(config)
    db_map = _get_vector_db_dirs(config)  # {db_name -> path}

    # ---- Thought: need retrieval? ----
    need_retrieval, decision_log = _decide_need_retrieval(
        question, config, llm_backend
    )

    retrieved_docs: List[Document] = []
    used_db_names: List[str] = []
    context = ""
    db_selection_log = ""
    per_db_logs: Dict[str, str] = {}

    # ---- Action: if needed, pick DBs & retrieve ----
    db_descriptions: Dict[str, str] = {}
    if need_retrieval:
        embedding_model = get_embedding_model(config)
        db_descriptions = _describe_databases(db_map, embedding_model)

        used_db_names, db_selection_log = _decide_which_dbs(
            question=question,
            db_map=db_map,
            db_descriptions=db_descriptions,
            llm_backend=llm_backend,
        )

        if used_db_names:
            all_docs: List[Document] = []
            for db_name in used_db_names:
                db_path = db_map[db_name]
                docs_db, log_db = _retrieve_documents_from_db(
                    question=question,
                    config=config,
                    embedding_model=embedding_model,
                    db_name=db_name,
                    db_path=db_path,
                )
                per_db_logs[db_name] = log_db
                all_docs.extend(docs_db)

            retrieved_docs = all_docs
            context = _build_context(retrieved_docs)
        else:
            need_retrieval = False  # model explicitly selected NONE

    # ---- Answer: main LLM call ----
    if config.agentic_mode == "react":
        system_prompt = (
            "You are an agentic reasoning assistant. "
            "If context from retrieved documents is provided, use it as your "
            "primary source of truth. If no context is provided, rely on your "
            "own knowledge. In all cases, do not reveal your internal chain-of-"
            "thought; provide only a clear final answer. If you are uncertain, "
            "say so explicitly."
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append(
            "Provide a clear, concise final answer without exposing your internal steps."
        )
        user_prompt = "\n\n".join(user_parts)
    else:
        system_prompt = (
            "You are a helpful assistant answering questions. "
            "If context from retrieved documents is provided, treat it as the most "
            "authoritative source. If no context is provided, rely on your own "
            "knowledge to answer. If the question cannot be answered reliably, "
            "explain that you are unsure."
        )
        user_parts = [f"Question:\n{question}"]
        if context:
            user_parts.append(f"Context from retrieved documents:\n{context}")
        user_parts.append("Provide a concise, accurate answer.")
        user_prompt = "\n\n".join(user_parts)

    answer = llm_backend.chat(system_prompt, user_prompt)

    # ---- Optional ReAct-style trace + retrieval + agent config logs ----
    reasoning_trace: Optional[str] = None
    if config.agentic_mode == "react" and show_reasoning:
        if need_retrieval:
            thought_str = (
                "The agent analyzed the question to understand its topic and "
                "determined that consulting one or more vector databases would "
                "improve the answer."
            )
        else:
            thought_str = (
                "The agent analyzed the question and decided it could be answered "
                "reliably without consulting any external databases."
            )

        if need_retrieval and used_db_names:
            action_str = (
                "The agent chose to retrieve from the following databases: "
                + ", ".join(f"`{n}`" for n in used_db_names)
                + "."
            )
        elif need_retrieval and not used_db_names:
            action_str = (
                "The agent considered retrieval, but did not select any specific "
                "database for this question."
            )
        else:
            action_str = (
                "The agent skipped retrieval and relied solely on its own knowledge."
            )

        observation_str = _build_observation_text(
            question=question,
            need_retrieval=need_retrieval,
            used_db_names=used_db_names,
            docs=retrieved_docs,
            llm_backend=llm_backend,
        )

        per_db_log_block = ""
        if per_db_logs:
            for db_name, log in per_db_logs.items():
                per_db_log_block += f"\n\n[DB {db_name}]\n{log}"

        retrieval_log_block = (
            f"{decision_log}\n\n"
            f"{db_selection_log}\n"
            f"{per_db_log_block.strip()}"
        ).strip()

        agent_config_log = _build_agent_config_log(
            config=config,
            db_map=db_map,
            db_descriptions=db_descriptions if db_descriptions else None,
        )

        reasoning_trace = (
            f"**Thought**: {thought_str}\n\n"
            f"**Action**: {action_str}\n\n"
            f"**Observation**:\n{observation_str}\n\n"
            f"**Retrieval / Post-Retrieval Optimization Log**:\n"
            f"```text\n{retrieval_log_block}\n```\n\n"
            f"**Agent / DB Configuration**:\n"
            f"```text\n{agent_config_log}\n```"
        )

    return answer, retrieved_docs, reasoning_trace


# Public alias
def single_agent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    return _single_agent_answer_question_core(question, config, show_reasoning)
