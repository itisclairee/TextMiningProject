# pages/2_Vector_DB_Builder.py

import os

import streamlit as st

from backend.config import RAGConfig
from backend.document_loader import load_documents_from_folders
from backend.embeddings import get_embedding_model
from backend.vector_store import build_vector_store
from backend.vector_store import clear_vector_store_cache

BASE_VECTOR_DIR = "vector_store"  # 🔹 all vector DBs live under this folder


def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


# ---------- CACHING LAYERS ----------

@st.cache_data(show_spinner=True)
def cached_load_documents_from_folders(
    folders: tuple[str, ...],
    corpus_name: str,
):
    """
    Cache documents loaded from the given folders + corpus tagging.

    Streamlit cache needs hashable args, so we pass folders as a tuple.
    """
    docs = load_documents_from_folders(list(folders))
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["corpus"] = corpus_name
    return docs


@st.cache_resource(show_spinner=False)
def cached_get_embedding_model(config: RAGConfig):
    """
    Cache the embedding model so it's created only once per session/config.
    """
    return get_embedding_model(config)


# ---------- UI ----------

st.title("📚 Vector DB Builder (LangChain + FAISS)")

config = get_config()

# Ensure base "vector_store" directory exists
os.makedirs(BASE_VECTOR_DIR, exist_ok=True)

if not config.json_folders:
    st.warning("No JSON folders configured. Go to the **Configuration** page first.")
    st.stop()

# ---------------- SHOW CONFIGURED JSON FOLDERS ----------------
st.subheader("Configured JSON folders")
for f in config.json_folders:
    st.code(f)

# ---------------- EXISTING VECTOR DBs ----------------
st.subheader("Existing Vector Databases (inside 'vector_store/')")

if getattr(config, "vector_store_dirs", None):
    st.write("Registered vector store directories:")
    for d in config.vector_store_dirs:
        st.code(d)
else:
    st.info("No vector stores registered yet. You can create one below.")

# ---------------- SELECT FOLDERS FOR THIS DB ----------------
st.subheader("Source folders for this vector store")

selected_folders = st.multiselect(
    "Choose which JSON folders to include in this vector database",
    options=config.json_folders,
    default=config.json_folders,
)

if not selected_folders:
    st.warning("Select at least one folder to build a vector store.")

# ---------------- VECTOR STORE NAME + CORPUS NAME ----------------
st.subheader("Target Vector Store settings")

current_path = config.vector_store_dir or ""
current_name = os.path.basename(os.path.normpath(current_path)) if current_path else "default"

vector_store_name = st.text_input(
    "Vector Store Name (subfolder inside 'vector_store/')",
    value=current_name,
    help=(
        "This will create/use a subfolder inside the 'vector_store' directory.\n"
        "Example: name = 'legal' -> path = 'vector_store/legal'."
    ),
)

target_vector_dir = os.path.join(BASE_VECTOR_DIR, vector_store_name) if vector_store_name else ""

default_corpus_name = vector_store_name or "corpus"
corpus_name = st.text_input(
    "Logical corpus name (stored in metadata)",
    value=default_corpus_name,
    help=(
        "This name will be added to the metadata of all documents in this "
        "vector store (metadata['corpus']). It can be used later for analysis "
        "or more advanced orchestration logic."
    ),
)

st.caption(f"Resulting vector store path will be: `{target_vector_dir}`")

# ---------------- BUILD BUTTON ----------------
if st.button("🔍 Scan folders & Build Vector DB"):
    if not selected_folders:
        st.error("Please select at least one folder to build a vector store.")
    elif not vector_store_name:
        st.error("Please specify a vector store name (e.g. 'legal').")
    else:
        progress = st.progress(0)

        # 1) Load docs (CACHED)
        with st.spinner("Loading JSON documents from selected folders..."):
            docs = cached_load_documents_from_folders(
                tuple(selected_folders),
                corpus_name,
            )
        progress.progress(30)

        if not docs:
            st.error("No documents found in the selected folders.")
            st.stop()

        st.success(f"Loaded {len(docs)} documents. Initializing embedding model...")
        progress.progress(50)

        # 2) Create embeddings model (CACHED)
        embedding_model = cached_get_embedding_model(config)
        progress.progress(65)

        # Ensure target directory exists (under vector_store/)
        os.makedirs(target_vector_dir, exist_ok=True)

        # 3) Build FAISS vector store at the chosen directory
        with st.spinner(
            f"Computing embeddings & building FAISS index at `{target_vector_dir}`..."
        ):
            build_vector_store(docs, embedding_model, target_vector_dir)
        progress.progress(100)

        # Update config
        config.vector_store_dir = target_vector_dir
        if target_vector_dir not in config.vector_store_dirs:
            config.vector_store_dirs.append(target_vector_dir)

        st.session_state.config = config

        st.success(
            f"Vector store created at: {target_vector_dir} "
            f"(docs: {len(docs)})"
        )

        st.info(
            "You can build another vector store by changing the 'Vector Store Name'. "
            "All vector DBs will live inside the 'vector_store/' folder. "
            "The ReAct agent can then choose which DB(s) to use at query time."
        )

st.caption(
    "When at least one vector DB is built, go to **Chatbot Q&A** "
    "to start querying your corpora."
)



# ---------- CACHE MANAGEMENT UI ----------
with st.expander("Advanced: cache management"):
    if st.button("♻️ Clear caches & Data"):
        # Clear Streamlit caches
        cached_load_documents_from_folders.clear()
        cached_get_embedding_model.clear()

        # Clear in-memory FAISS cache
        clear_vector_store_cache(current_name)

        st.success("Caches cleared. The next build will reload everything from disk.")