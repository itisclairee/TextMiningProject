import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

# Role of this module:
#This is the ingestion layer: it converts your raw JSON legal cases into 
# LangChain Documents that can then be embedded and stored.
# Used mainly by vector_store.py and the Vector DB Builder page.

def _extract_docs_from_json_object(obj: Any, source: str) -> List[Document]:
    """
    Accepts:
      - A dict with 'content'/'text'/'corpus' and optional 'metadata'
      - A list of such dicts
    Returns a list of LangChain Document objects.
    """

    docs: List[Document] = []

    def normalize_single(item: dict) -> Document | None:
        content_key = None
        for k in ["content", "text", "corpus"]:
            if k in item:
                content_key = k
                break

        if content_key is None:
            return None

        content = item[content_key]
        if not isinstance(content, str):
            content = str(content)

        meta = item.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {"metadata_raw": str(meta)}

        # Attach source path
        meta = {"source": source, **meta}
        return Document(page_content=content, metadata=meta)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                d = normalize_single(item)
                if d is not None:
                    docs.append(d)
    elif isinstance(obj, dict):
        d = normalize_single(obj)
        if d is not None:
            docs.append(d)

    return docs


def load_documents_from_folders(folders: List[str]) -> List[Document]:
    all_docs: List[Document] = []

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[document_loader] Folder does not exist: {folder_path}")
            continue

        for json_file in folder_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = _extract_docs_from_json_object(data, source=str(json_file))
                all_docs.extend(docs)
            except Exception as e:
                print(f"[document_loader] Error loading {json_file}: {e}")

    return all_docs
