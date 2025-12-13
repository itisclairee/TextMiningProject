# backend/llm_provider.py

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from .config import RAGConfig


# Role of this module:
# Abstracts away LLM details so all other modules call the same simple interface, regardless of provider or model.


class LLMBackend:
    """
    Unified interface for LLM providers:

      - openai       → ChatOpenAI
      - huggingface  → HuggingFaceEndpoint + ChatHuggingFace

    Hugging Face notes:
      - `llm_model_name` must be a valid repo id on HF
        (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) or a local path
        if you later switch to local inference.
      - For private/gated models you must set
        HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in your .env.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.max_new_tokens = 512
        self.temperature = 0.2

    # ------------------------------------------------------------------
    # OPENAI
    # ------------------------------------------------------------------
    def _build_openai_chat(self) -> BaseChatModel:
        # Requires OPENAI_API_KEY in env
        return ChatOpenAI(
            model=self.config.llm_model_name,
            temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # HUGGING FACE (Inference API via HuggingFaceEndpoint)
    # ------------------------------------------------------------------
    def _build_hf_chat(self) -> Optional[BaseChatModel]:
        repo_id = (self.config.llm_model_name or "").strip()
        if not repo_id:
            print("[LLMBackend] Empty Hugging Face model name in config.")
            return None

        # Token for HF Inference API (needed for many models, especially private/gated)
        hf_token = (
            os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
        )
        if hf_token is None:
            print(
                "[LLMBackend] HUGGINGFACEHUB_API_TOKEN/HF_TOKEN not set. "
                "Public open models may still work, but private/gated ones will fail."
            )

        try:
            # Base HF LLM using Inference API
            base_llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                # provider="auto",  # optional, HF chooses the backend
            )
            # Chat wrapper with messages API
            chat_llm = ChatHuggingFace(llm=base_llm)
            return chat_llm
        except Exception as e:
            print(f"[LLMBackend] Error creating Hugging Face model: {e}")
            return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    def get_langchain_llm(self) -> Optional[BaseChatModel]:
        provider = self.config.llm_provider

        if provider == "openai":
            return self._build_openai_chat()

        if provider == "huggingface":
            return self._build_hf_chat()

        return None

    # ------------------------------------------------------------------
    # High-level chat method used by rag_pipeline
    # ------------------------------------------------------------------
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        llm = self.get_langchain_llm()
        if llm is None:
            return (
                "LLM provider is not correctly configured or the model could not be "
                "loaded.\n\n"
                "Please check your Configuration page:\n"
                "- If provider = **huggingface**, set `llm_model_name` to a valid "
                "Hugging Face repo id (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) and "
                "set `HUGGINGFACEHUB_API_TOKEN` in `.env` for private/gated models.\n"
                "- If provider = **openai**, make sure `OPENAI_API_KEY` is set."
            )

        try:
            # Preferred: role-based messages
            messages = [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
            resp = llm.invoke(messages)
        except TypeError:
            # Fallback if the model doesn't support (role, content) tuples
            combined_prompt = system_prompt + "\n\n" + user_prompt
            try:
                resp = llm.invoke(combined_prompt)
            except Exception as e:
                return f"[LLM error] {e}"
        except Exception as e:
            return f"[LLM error] {e}"

        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
