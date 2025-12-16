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


# -------------------------------
# Config helper
# -------------------------------
def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


# -------------------------------
# Chat DB helpers
# -------------------------------
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
    if not history:
        return

    db = load_chat_db()
    next_id = max([c.get("id", 0) for c in db], default=0) + 1

    title = next((msg.get("content", "").strip() for msg in history if msg.get("role") == "user"), None)
    if not title:
        title = f"Chat {next_id}"

    db.append(
        {"id": next_id, "title": title[:80], "history": history}
    )
    save_chat_db(db)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("RAG Chatbot (Agentic - Hybrid)")

config = get_config()

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []

col_top1, col_top2, col_top3 = st.columns([1, 1, 2])
with col_top1:
    if st.button("New Chat"):
        append_chat_to_db(st.session_state.chat_history)
        st.session_state.chat_history = []
        st.success("Current chat saved. Started a new chat.")

with col_top2:
    if st.button("Clear"):
        st.session_state.chat_history = []
        st.info("Chat cleared (not saved).")

with col_top3:
    db = load_chat_db()
    st.caption(f"📁 Saved chats in DB: **{len(db)}** (stored in `{CHAT_DB_PATH.name}`)")


# -------------------------------
# Options
# -------------------------------
agentic_mode = getattr(config, "agentic_mode", "standard_rag")
use_multiagent = getattr(config, "use_multiagent", False)

# Definiamo 4 colonne
col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

# Se agent logs è presente, lo mettiamo nella prima colonna
show_agent_logs = False
if use_multiagent:
    show_agent_logs = col_opt1.checkbox("Show agent logs", value=False)
    col_left1 = col_opt2
    col_left2 = col_opt3
    col_left3 = col_opt4
else:
    col_left1 = col_opt1
    col_left2 = col_opt2
    col_left3 = col_opt3

# Le altre checkbox seguono
show_sources = col_left1.checkbox("Show sources", value=True)
show_retrieval_logs = col_left2.checkbox("Show retrieval logs", value=False)

show_react_trace = False
if col_left3 and agentic_mode == "react" and not use_multiagent:
    show_react_trace = col_left3.checkbox("Show ReAct trace", value=False)


# -------------------------------
# Render existing history
# -------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------
# Chat input + answer
# -------------------------------
user_input = st.chat_input("Ask me something about your legal corpus...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            use_hybrid = agentic_mode == "hybrid_legal"
            need_reasoning = show_react_trace or show_retrieval_logs or show_agent_logs

            if use_hybrid:
                answer, docs, reasoning_trace, extracted_meta = hybrid_answer_question(
                    user_input, config, show_reasoning=need_reasoning
                )
            else:
                answer, docs, reasoning_trace = rag_answer_question(
                    user_input, config, show_reasoning=need_reasoning
                )
                extracted_meta = None

        st.markdown(answer)

        # Optional reasoning/logs
        if reasoning_trace:
            # You can use your split_reasoning_trace function here if needed
            st.text_area("Reasoning trace", reasoning_trace, height=300)

        # Show sources
        if show_sources and docs:
            with st.expander("📎 Sources used"):
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", "unknown")
                    db_name = d.metadata.get("db_name", "")
                    prefix = f"[DB: {db_name}] " if db_name else ""
                    st.markdown(f"**Source {i+1}:** {prefix}`{src}`")
                    st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
                    st.json(d.metadata or {})

        # Hybrid metadata
        if extracted_meta is not None:
            with st.expander("📑 Extracted legal metadata"):
                st.json(extracted_meta)

    # Store assistant message
    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": answer}
    if docs:
        assistant_msg["contexts"] = [d.page_content for d in docs]
        assistant_msg["source_ids"] = [d.metadata.get("source", "unknown") for d in docs]
    if extracted_meta is not None:
        assistant_msg["extracted_metadata"] = extracted_meta

    st.session_state.chat_history.append(assistant_msg)



# -------------------------------
# Custom CSS per badge messaggi
# -------------------------------
st.markdown("""
<style>
/* Badge dell'utente */

            #agentic-hybrid-rag-chatbot {
position: fixed!important;
background: white;
z-index: 999999;
top: 20px;
width: 100%;
}
</style>
            
            
""", unsafe_allow_html=True)



# CSS personalizzato
st.markdown(
    """
    <style>
    .st-emotion-cache-pk3c77 h1, 
    .st-emotion-cache-pk3c77 h2, 
    .st-emotion-cache-pk3c77 h3, 
    .st-emotion-cache-pk3c77 h4, 
    .st-emotion-cache-pk3c77 h5, 
    .st-emotion-cache-pk3c77 h6 {
        font-family: "Source Sans", sans-serif;
        line-height: 1.2;
        margin: 0px;
        color: inherit;
        font-family: Helvetica;
    }
    p {
        font-family: Helvetica;
    }
    li {
        font-family: Helvetica !important;
    }
    .st-emotion-cache-23r7bk {
        background-color: lightblue !important;
        color: white !important;
        border-radius: 0.5rem !important;
    }

    /* Badge dell'assistente */
    div[role="listitem"] > div > div:last-child {
        background-color: green !important;
        color: white !important;
        border-radius: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)