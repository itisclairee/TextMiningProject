import os
import streamlit as st
from dotenv import load_dotenv

# Load env vars from .env (OPENAI_API_KEY, etc.)
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Playground (LangChain)",
    layout="wide",
)

# CSS personalizzato
st.markdown(
    """
    <style>
    .st-emotion-cache-pk3c77 h1, 
    .st-emotion-cache-pk3c77 h2, 
    .st-emotion-cache-pk3c77 h3, 
    .st-emotion-cache-pk3c77 h4, 
    .st-emotion-cache-pk3c77 h5, 
    .st-emotion-cache-pk3c77 h6 {
        font-family: "Source Sans", sans-serif;
        line-height: 1.2;
        margin: 0px;
        color: inherit;
        font-family: Helvetica;
    }
    p {
        font-family: Helvetica;
    }
    li {
        font-family: Helvetica !important;
    }
  
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Smart Agentic RAG  HUB")

st.markdown(
    """
    Welcome!

    Use the sidebar to explore the app:

    1. **Settings** – Set up your LLM provider and model, choose an embedding model, specify JSON corpus folders, and configure the agentic RAG behavior.  
    2. **Vector DB Builder** – Import your JSON data and build the FAISS vector databases, ready for retrieval with LangChain.  
    3. **Chatbot Q&A** – Interact with your intelligent RAG chatbot, powered by either single-agent, multi-agent, or hybrid legal pipelines.
    """
)
