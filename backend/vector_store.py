# backend/vector_store.py

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


# In-memory cache: {path -> FAISS vector store}
_VECTOR_STORE_CACHE: Dict[str, FAISS] = {}


# ------------------------------------------------------------------
# BUILD
# ------------------------------------------------------------------
def build_vector_store(
    docs: List[Document],
    embedding_model: Embeddings,
    target_dir: str,
) -> None:
    """
    Build a FAISS vector store from documents and save it to disk.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(docs, embedding_model)
    vs.save_local(str(target_path))

    _VECTOR_STORE_CACHE[str(target_path)] = vs


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
def load_vector_store(
    path: str,
    embedding_model: Embeddings,
) -> FAISS:
    """
    Load a FAISS vector store from disk.
    Raises a clear error if the index does not exist.
    """
    path = Path(path)

    # Cache hit
    cached = _VECTOR_STORE_CACHE.get(str(path))
    if cached is not None:
        return cached

    # Validate existence
    index_file = path / "index.faiss"
    store_file = path / "index.pkl"

    if not index_file.exists() or not store_file.exists():
        raise RuntimeError(
            f"FAISS vector store not found at:\n"
            f"  {path}\n\n"
            f"Expected files:\n"
            f"  - index.faiss\n"
            f"  - index.pkl\n\n"
            f"You must build the vector store first "
            f"(offline ingestion step missing)."
        )

    vs = FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    _VECTOR_STORE_CACHE[str(path)] = vs
    return vs


# ------------------------------------------------------------------
# CLEAR
# ------------------------------------------------------------------
def clear_vector_store_cache(path: str) -> None:
    """
    Delete vector store from disk and from in-memory cache.
    """
    path = Path(path)

    _VECTOR_STORE_CACHE.pop(str(path), None)

    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    
# ------------------------------------------------------------------
# DEBUG: visualizza documenti nel vector store
# ------------------------------------------------------------------
def debug_vector_store_contents(path: str, embedding_model: Embeddings, max_preview: int = 200) -> None:
    """
    Stampa i documenti presenti in un FAISS vector store.

    Args:
        path: percorso del vector store
        embedding_model: modello di embedding usato per il caricamento
        max_preview: numero massimo di caratteri del contenuto da stampare
    """
    print(f"\n=== DEBUG: Vector store at '{path}' ===\n")
    
    vs = load_vector_store(path, embedding_model)
    
    try:
        # Recupera i documenti dalla docstore (FAISS li memorizza qui)
        docs: list[Document] = []
        if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
            docs = list(vs.docstore._dict.values())
        
        if not docs:
            print("[DEBUG] Nessun documento trovato nel vector store.")
            return

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "UNKNOWN")
            preview = doc.page_content[:max_preview].replace("\n", " ")
            print(f"Doc {i}: source={source}\nPreview: {preview}\n---")
    except Exception as e:
        print(f"[DEBUG] Errore durante la lettura dei documenti: {e}")