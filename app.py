import os

import streamlit as st
from dotenv import load_dotenv

# Load env vars from .env (OPENAI_API_KEY, etc.)
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Playground (LangChain)",
    layout="wide",
)

st.title("Agentic RAG Playground (LangChain)")


st.markdown(
    """
    Welcome!

    Use the sidebar to explore the app:

    1. **Settings** – Set up your LLM provider and model, choose an embedding model, specify JSON corpus folders, and configure the agentic RAG behavior.  
    2. **Vector DB Builder** – Import your JSON data and build the FAISS vector databases, ready for retrieval with LangChain.  
    3. **Chatbot Q&A** – Interact with your intelligent RAG chatbot, powered by either single-agent, multi-agent, or hybrid legal pipelines.
    """
)
st.info("Select a page from the sidebar to get started.")


