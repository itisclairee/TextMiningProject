from __future__ import annotations
from pathlib import Path
from typing import List
import os
from langchain_core.documents import Document  
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import shutil

# This is the vector database layer: which builds and loads FAISS vector stores using LangChain-Documents.

# During offline step, called by the Vector DB Builder page to create FAISS DBs.
# During online step, called by RAG pipelines to load the correct DB and create retrievers.


# backend/vector_store.py
# Simple in-memory cache: {path -> FAISS vector store}
_VECTOR_STORE_CACHE: dict[str, FAISS] = {}

def build_vector_store(
    docs: List[Document],
    embedding_model,
    target_dir: str,
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    vs = FAISS.from_documents(docs, embedding_model)
    vs.save_local(target_dir)

    _VECTOR_STORE_CACHE[target_dir] = vs


def load_vector_store(
    path: str,
    embedding_model,
) -> FAISS:
    cached = _VECTOR_STORE_CACHE.get(path)
    if cached is not None:
        return cached

    vs = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    _VECTOR_STORE_CACHE[path] = vs
    return vs


def clear_vector_store_cache(path: str) -> None:
    """Delete vector store from disk and from in-memory cache."""
    if path in _VECTOR_STORE_CACHE:
        del _VECTOR_STORE_CACHE[path]
    if os.path.isdir(path):
        shutil.rmtree(path)