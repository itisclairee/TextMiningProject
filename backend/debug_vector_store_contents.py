# backend/debug_vector_store_contents.py

from backend.vector_store import load_vector_store
from backend.embeddings import get_embedding_model
from backend.config import RAGConfig

def debug_vector_store(path: str):
    config = RAGConfig()  # Assicurati che il tuo config punti ai parametri giusti
    embedding_model = get_embedding_model(config)

    # Carica il vector store
    vs = load_vector_store(path, embedding_model)

    # Accedi a tutti i documenti
    docs = list(vs.docstore._dict.values())
    print(f"Numero di documenti nel vector store '{path}': {len(docs)}\n")

    # Mostra i primi 20 documenti (testo e metadata)
    for i, doc in enumerate(docs[:20]):
        print(f"--- Document {i+1} ---")
        print("Source:", doc.metadata.get("source", "N/A"))
        print("Law:", doc.metadata.get("law", "N/A"))
        print("Type:", doc.metadata.get("type", "N/A"))
        print("Contenuto (prime 200 caratteri):", doc.page_content[:200], "...\n")

if __name__ == "__main__":
    vector_store_path = "vector_store/vector_storeItaly"  # Modifica se necessario
    debug_vector_store(vector_store_path)