# backend/config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Role: centralized configuration object for backend modules.
# Streamlit config page writes to this, and other modules read from it.

@dataclass
class RAGConfig:
    """
    Global configuration object stored in Streamlit session state.

    Controls:
      - LLM provider & model
      - Embedding provider & model
      - Data folders with JSON corpus
      - Vector store locations (single & multi-DB)
      - Retrieval behavior
      - Agentic behavior (standard RAG, ReAct, hybrid legal RAG)
      - Optional multi-agent supervisor
    """

    # ---------------- LLM ----------------
    llm_provider: str = "openai"           # "openai" or "huggingface"
    llm_model_name: str = "gpt-4o-mini"

    # ---------------- Embeddings ----------------
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------- Data (JSON corpus) ----------------
    json_folders: List[str] = field(default_factory=list)

    # ---------------- Vector stores ----------------
    # Base folder under which all FAISS vector DBs live
    vector_store_base_dir: str = "vector_store"     # main base dir
    vector_store_root: str = "vector_store"         # alias for new code
    vector_store_dir: str = "vector_store"          # default single-DB dir
    vector_store_dirs: List[str] = field(default_factory=list)  # multi-DB optional

    # Legacy compatibility for code referencing `vector_db_base_dir`
    @property
    def vector_db_base_dir(self) -> str:
        """Alias for older code referencing `vector_db_base_dir`."""
        return self.vector_store_base_dir

    # ---------------- Retrieval ----------------
    top_k: int = 5
    use_rerank: bool = False  # placeholder for future reranking strategies

    # ---------------- Agentic behavior ----------------
    # Modes:
    #   - "standard_rag"  -> classic RAG (vector retrieval + answer)
    #   - "react"         -> ReAct-style agentic RAG
    #   - "hybrid_legal"  -> hybrid legal RAG (metadata-aware)
    agentic_mode: str = "standard_rag"

    # Multi-agent supervisor switch (used in rag_pipeline)
    use_multiagent: bool = False