# pages/3_Chatbot_QA.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

from backend.config import RAGConfig
from backend.rag_pipeline import answer_question as rag_answer_question
from backend.hybrid_rag import hybrid_answer_question


CHAT_DB_PATH = Path("chat_sessions.json")


# ---------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------
def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


# ---------------------------------------------------------------------
# Chat DB helpers
# ---------------------------------------------------------------------
def load_chat_db() -> List[Dict[str, Any]]:
    if CHAT_DB_PATH.exists():
        try:
            with open(CHAT_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat_db(db: List[Dict[str, Any]]) -> None:
    try:
        with open(CHAT_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[chat_db] Error saving chat DB: {e}")


def append_chat_to_db(history: List[Dict[str, Any]]) -> None:
    """
    Append the current chat history as a new session in chat_sessions.json.

    Each message in `history` is expected to be like:
      {
        "role": "user" | "assistant",
        "content": "...",
        # optional:
        "contexts": [...],
        "source_ids": [...],
        "extracted_metadata": {...}
      }
    """
    if not history:
        return

    db = load_chat_db()
    next_id = max([c.get("id", 0) for c in db], default=0) + 1

    # Use the first user message as a title, if available
    title = None
    for msg in history:
        if msg.get("role") == "user":
            title = msg.get("content", "").strip()
            break
    if not title:
        title = f"Chat {next_id}"

    db.append(
        {
            "id": next_id,
            "title": title[:80],
            "history": history,
        }
    )
    save_chat_db(db)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("💬 Agentic / Hybrid RAG Chatbot")

config = get_config()

# Initialize chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []

# Top controls: new chat + clear chat + info on saved chats
col_top1, col_top2, col_top3 = st.columns([1, 1, 2])
with col_top1:
    if st.button("🆕 New chat (save current)"):
        append_chat_to_db(st.session_state.chat_history)
        st.session_state.chat_history = []
        st.success("Current chat saved. Started a new chat.")

with col_top2:
    if st.button("🗑 Clear current chat"):
        st.session_state.chat_history = []
        st.info("Chat cleared (not saved).")

with col_top3:
    db = load_chat_db()
    st.caption(f"📁 Saved chats in DB: **{len(db)}** (stored in `{CHAT_DB_PATH.name}`)")


# ---------------------------------------------------------------------
# Options: reasoning, logs, sources, agents
# ---------------------------------------------------------------------
agentic_mode = getattr(config, "agentic_mode", "standard_rag")
use_multiagent = getattr(config, "use_multiagent", False)

col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

# ReAct trace only meaningful when agentic_mode == "react" and NOT hybrid
with col_opt1:
    show_react_trace = False
    if agentic_mode == "react" and not use_multiagent:
        show_react_trace = st.checkbox(
            "Show ReAct trace",
            value=False,
            help=(
                "Mostra la traccia ad alto livello (Thought / Action / Observation) "
                "per il singolo agente ReAct."
            ),
        )

with col_opt2:
    show_sources = st.checkbox(
        "Show sources",
        value=True,
        help="Mostra i documenti recuperati (chunk di contesto) usati nella risposta.",
    )

with col_opt3:
    show_retrieval_logs = st.checkbox(
        "Show retrieval logs",
        value=False,
        help=(
            "Mostra i log di retrieval / post-retrieval (filtri, similarity, top-k) "
            "o, in modalità ibrida, i log del RAG ibrido."
        ),
    )

with col_opt4:
    show_agent_logs = False
    if use_multiagent:
        show_agent_logs = st.checkbox(
            "Show agent logs",
            value=False,
            help=(
                "Mostra i log del supervisore multi-agente "
                "(routing verso gli agenti specializzati)."
            ),
        )


# ---------------------------------------------------------------------
# Utility: split reasoning_trace into sections for nicer display
# ---------------------------------------------------------------------
def split_reasoning_trace(
    agentic_mode: str,
    use_multiagent: bool,
    reasoning_trace: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (react_part, retrieval_logs_part, agent_logs_part)

    - For ReAct single-agent: reasoning_trace contains Thought/Action/Observation
      + retrieval logs → we split on "Retrieval / Post-Retrieval Optimization Log".
    - For multiagent: entire trace is considered "agent logs".
    - For hybrid_legal: entire trace is retrieval/logs (no ReAct or agents).
    """
    if not reasoning_trace:
        return None, None, None

    # Multi-agent trace: treat as agent logs only
    if use_multiagent:
        return None, None, reasoning_trace

    # Hybrid legal: everything is "retrieval/logs"
    if agentic_mode == "hybrid_legal":
        return None, reasoning_trace, None

    # ReAct single-agent
    if agentic_mode == "react":
        marker = "**Retrieval / Post-Retrieval Optimization Log**:"
        if marker in reasoning_trace:
            head, logs = reasoning_trace.split(marker, 1)
            react_part = head.strip()
            logs_part = (marker + "\n" + logs.strip()).strip()
            return react_part or None, logs_part or None, None
        else:
            # no explicit marker; treat everything as ReAct part
            return reasoning_trace, None, None

    # Fallback: treat everything as retrieval logs
    return None, reasoning_trace, None


# ---------------------------------------------------------------------
# Render existing history
# ---------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------------------------
# Chat input + answer
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask me something about your legal corpus (succession/divorce)...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Decide which pipeline: hybrid legal or standard RAG
            use_hybrid = agentic_mode == "hybrid_legal"

            # We request reasoning logs if any of the UI toggles need them
            need_reasoning = show_react_trace or show_retrieval_logs or show_agent_logs

            if use_hybrid:
                answer, docs, reasoning_trace, extracted_meta = hybrid_answer_question(
                    user_input,
                    config,
                    show_reasoning=need_reasoning,
                )
            else:
                answer, docs, reasoning_trace = rag_answer_question(
                    user_input,
                    config,
                    show_reasoning=need_reasoning,
                )
                extracted_meta = None

        answer_text = answer
        st.markdown(answer_text)

        # ---------- Optional reasoning / logs display ----------
        if reasoning_trace:
            react_part, retrieval_logs_part, agent_logs_part = split_reasoning_trace(
                agentic_mode=agentic_mode,
                use_multiagent=use_multiagent,
                reasoning_trace=reasoning_trace,
            )

            # ReAct trace only (single agent, when enabled)
            if show_react_trace and react_part:
                with st.expander("🔍 ReAct trace (Thought / Action / Observation)"):
                    st.markdown(react_part)

            # Retrieval / post-retrieval logs
            if show_retrieval_logs and retrieval_logs_part:
                with st.expander("📈 Retrieval / post-retrieval logs"):
                    st.markdown(retrieval_logs_part)

            # Agent logs (multi-agent supervisor)
            if show_agent_logs and agent_logs_part:
                with st.expander("🤖 Multi-agent routing logs"):
                    st.markdown(agent_logs_part)

        # ---------- Show sources (retrieved documents) ----------
        if show_sources and docs:
            with st.expander("📎 Sources used"):
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", "unknown")
                    db_name = d.metadata.get("db_name", "")
                    prefix = f"[DB: {db_name}] " if db_name else ""
                    st.markdown(f"**Source {i+1}:** {prefix}`{src}`")

                    # Content preview
                    preview = d.page_content[:500]
                    if len(d.page_content) > 500:
                        preview += "..."
                    st.write(preview)

                    # 👉 Show full metadata as JSON
                    st.markdown("**Metadata:**")
                    st.json(d.metadata or {})

                    st.markdown("---")


        # ---------- Hybrid legal metadata (if available) ----------
        if extracted_meta is not None:
            with st.expander("📑 Extracted legal metadata (hybrid RAG)"):
                st.json(extracted_meta)

    # Store assistant message in history, including retrieved sources for RAG evaluation
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": answer_text,
    }

    # Attach retrieved contexts and source ids so we can use them later with RAGAS
    if docs:
        assistant_msg["contexts"] = [d.page_content for d in docs]
        assistant_msg["source_ids"] = [d.metadata.get("source", "unknown") for d in docs]

    # Attach hybrid metadata if present
    if extracted_meta is not None:
        assistant_msg["extracted_metadata"] = extracted_meta

    st.session_state.chat_history.append(assistant_msg)
