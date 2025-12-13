import os

import streamlit as st
from dotenv import load_dotenv

# Load env vars from .env (OPENAI_API_KEY, etc.)
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Playground (LangChain)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agentic RAG Playground (LangChain)")

if os.getenv("OPENAI_API_KEY"):
    st.caption("🔑 OPENAI_API_KEY loaded from .env")
else:
    st.warning(
        "OPENAI_API_KEY not found in environment. "
        "Set it in your .env file if you want to use OpenAI."
    )

st.markdown(
    """
Welcome!  

Use the sidebar to navigate:

1. **Configuration** – define LLM, embedding model, JSON data folders, and agentic RAG options.  
2. **Vector DB Builder** – load your JSON data and create the FAISS vector database (via LangChain).  
3. **Chatbot Q&A** – talk with your agentic RAG chatbot.
"""
)

st.info("➡️ Select a page from the sidebar to get started.")
