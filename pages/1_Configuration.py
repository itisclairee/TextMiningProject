# pages/1_Configuration.py
# Global configuration for the Agentic RAG app (OpenAI + HuggingFace)

import streamlit as st

from backend.config import RAGConfig


def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


st.title("Settings")

config = get_config()

# ---------------- LLM SETTINGS ----------------
st.subheader("LLM Settings")

col1, col2 = st.columns(2)

with col1:
    config.llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "huggingface"],
        index=["openai", "huggingface"].index(config.llm_provider)
        if config.llm_provider in ["openai", "huggingface"]
        else 0,
        help=(
            "- **openai**: uses ChatOpenAI (needs `OPENAI_API_KEY` in `.env`).\n"
            "- **huggingface**: any HF generative model (hub id or local path)."
        ),
    )

with col2:
    config.llm_model_name = st.text_input(
        "LLM Model Name or Path",
        value=config.llm_model_name,
        help=(
            "For OpenAI: e.g. `gpt-4o-mini`.\n"
            "For Hugging Face: model id (e.g. `meta-llama/Llama-3.1-8B-Instruct`) "
            "or a local folder path."
        ),
    )

# ---------------- EMBEDDING SETTINGS ----------------
st.subheader("Embedding Settings")

col3, col4 = st.columns(2)

with col3:
    config.embedding_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "openai"],
        index=["huggingface", "openai"].index(config.embedding_provider)
        if config.embedding_provider in ["huggingface", "openai"]
        else 0,
        help=(
            "huggingface → `HuggingFaceEmbeddings` (any HF model or local path).\n"
            "openai → `OpenAIEmbeddings` (e.g. `text-embedding-3-small`)."
        ),
    )

with col4:
    config.embedding_model_name = st.text_input(
        "Embedding Model Name or Path",
        value=config.embedding_model_name,
        help=(
            "For Hugging Face: e.g. `all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`, "
            "or a local path.\n"
            "For OpenAI: e.g. `text-embedding-3-small`."
        ),
    )

# ---------------- DATA FOLDERS ----------------
st.subheader("Data Folders (JSON content + metadata)")

# Input for new folder path
new_folder = st.text_input(
    "Add JSON folder path",
    value="",
    placeholder="data/folder1",
)

cols_add = st.columns([1, 1, 2])

with cols_add[0]:
    # Add folder only when the button is clicked
    if st.button("➕ Add folder") and new_folder:
        if new_folder not in config.json_folders:
            config.json_folders.append(new_folder)
            st.session_state.config = config
            st.success(f"Added folder: {new_folder}")
            # Force a “soft rerun” by updating a dummy state
            st.session_state._rerun = st.session_state.get("_rerun", 0) + 1

with cols_add[1]:
    # Remove all folders at once
    if st.button("🧹 Clear folders"):
        config.json_folders = []
        st.session_state.config = config
        st.session_state._rerun = st.session_state.get("_rerun", 0) + 1

# Show list of added folders
if config.json_folders:
    st.write(f"Current JSON folders ({len(config.json_folders)}):")

    for folder in config.json_folders.copy():
        colA, colB = st.columns([5, 1])

        with colA:
            st.code(folder)

        with colB:
            if st.button("❌", key=f"remove_{folder}"):
                config.json_folders.remove(folder)
                st.session_state.config = config
                st.session_state._rerun = st.session_state.get("_rerun", 0) + 1
else:
    st.info("No JSON folders added yet.")

# ---------------- VECTOR STORE SETTINGS ----------------
st.subheader("Vector Store Settings (FAISS)")

config.vector_store_base_dir = st.text_input(
    "Base directory for all vector stores",
    value=config.vector_store_base_dir,
    help=(
        "Root folder under which all FAISS vector DBs will be created.\n"
        "Example: `vector_store` → `vector_store/legal`, `vector_store/tax`, etc."
    ),
)

config.vector_store_dir = st.text_input(
    "Default / Legacy Vector Store Directory",
    value=config.vector_store_dir,
    help=(
        "Used as the default single vector DB path and for backward compatibility.\n"
        "For multi-DB setups, the Vector DB Builder will create subfolders under "
        "the base directory and update `vector_store_dirs`."
    ),
)

if config.vector_store_dirs:
    st.write("Registered vector DB directories (multi-DB mode):")
    for p in config.vector_store_dirs:
        st.code(p)
else:
    st.caption(
        "No explicit vector DB directories registered yet. "
    )

# ---------------- RETRIEVAL SETTINGS ----------------
st.subheader("Retrieval & Post-processing")

col_r1, col_r2 = st.columns(2)

with col_r1:
    config.top_k = st.slider(
        "Top-K documents after filtering",
        min_value=1,
        max_value=20,
        value=config.top_k,
        help="Number of documents kept after similarity-based filtering.",
    )

with col_r2:
    config.use_rerank = st.checkbox(
        "Use extra post-retrieval reranker (placeholder)",
        value=config.use_rerank,
        help=(
            "Reserved flag for plugging in an advanced reranker later.\n"
            "Current pipeline already does cosine-similarity filtering and reranking."
        ),
    )

# ---------------- AGENTIC MODE (within each RAG agent) ----------------
st.subheader("Agentic RAG Reasoning Mode (per agent)")

config.agentic_mode = st.radio(
    "Agentic mode",
    options=["standard_rag", "hybrid_legal"],
    index=["standard_rag", "hybrid_legal"].index(config.agentic_mode)
    if config.agentic_mode in ["standard_rag", "hybrid_legal"]
    else 0,
    help=(
        "- standard_rag: classic RAG.\n"
        "- hybrid_legal: legal metadata extraction (Succession/Divorce + cost, "
        "duration, civil codes, etc.) + metadata-aware vector search."
    ),
)


# ---------------- MULTI-AGENT SUPERVISOR ----------------
st.subheader("Multi-agent Supervisor (tool-calling)")

config.use_multiagent = st.checkbox(
    "Enable multi-agent supervisor over specialized RAG agents",
    value=config.use_multiagent,
    help=(
        "If enabled, the chatbot will use a supervisor LLM that routes questions "
        "to specialized RAG agents (one per vector database) and synthesizes their "
        "answers. If disabled, a single agent handles everything."
    ),
)

# ---------------- SAVE ----------------
if st.button("💾 Save configuration"):
    st.session_state.config = config
    st.success("Configuration saved in session.")

st.caption("Next: go to **Vector DB Builder** to create the FAISS vector store(s).")
