# backend/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Role of this module:
# All other backend modules read from one shared configuration object.
# The Streamlit config page writes into RAGConfig, and the backend behaves accordingly


@dataclass
class RAGConfig:
    """
    Global configuration object stored in Streamlit session state.

    It controls:
      - LLM provider & model
      - Embedding provider & model
      - Data folders with JSON corpus
      - Vector store locations (single & multi-DB)
      - Retrieval behavior
      - Agentic behavior (standard RAG, ReAct, hybrid legal RAG)
      - Optional multi-agent supervisor
    """

    # ---------------- LLM ----------------
    # "openai"       -> ChatOpenAI (needs OPENAI_API_KEY)
    # "huggingface"  -> HuggingFaceEndpoint / ChatHuggingFace (needs HF token for private models)
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"

    # ---------------- Embeddings ----------------
    # "huggingface" -> HuggingFaceEmbeddings (any HF model or local path)
    # "openai"      -> OpenAIEmbeddings
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------- Data (JSON corpus) ----------------
    # List of folders where JSON corpus lives
    json_folders: List[str] = field(default_factory=list)

    # ---------------- Vector stores (paths) ----------------
    # Root/base folder under which all FAISS vector DBs will be created.
    # Example: "vector_store" -> "vector_store/divorce_db", "vector_store/succession_db", etc.
    #
    # NOTE: `vector_store_base_dir` is kept for compatibility with the
    # Vector DB Builder / Config pages, which reference this attribute.
    vector_store_base_dir: str = "vector_store"

    # Alias used by some newer code; kept equal to base_dir by default.
    vector_store_root: str = "vector_store"

    # Default / active vector store directory (single-DB mode)
    # Example: "vector_store/default"
    vector_store_dir: str = "vector_store"

    # Optional: multiple DBs (multi-DB mode)
    # Example: ["vector_store/divorce_db", "vector_store/succession_db"]
    vector_store_dirs: List[str] = field(default_factory=list)

    # ---------------- Retrieval ----------------
    top_k: int = 5
    # Reserved for future reranking strategies; currently not used in the pipeline.
    use_rerank: bool = False

    # ---------------- Agentic behavior ----------------
    # agentic_mode:
    #   - "standard_rag"  -> classic RAG (vector retrieval + answer)
    #   - "react"         -> ReAct-style agentic RAG (Thought / Action / Observation)
    #   - "hybrid_legal"  -> hybrid legal RAG (metadata extraction + metadata-aware vector search)
    agentic_mode: str = "standard_rag"

    # Multi-agent supervisor switch (used only in rag_pipeline for multi-DB agent routing)
    use_multiagent: bool = False
