# backend/llm_provider.py

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from .config import RAGConfig


class LLMBackend:
    """
    Unified interface for LLM providers:

      - openai       → OpenAI SDK (supports OpenRouter)
      - huggingface  → HuggingFaceEndpoint + ChatHuggingFace
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.max_new_tokens = 512
        self.temperature = 0.2

    # ------------------------------------------------------------------
    # OPENAI / OPENROUTER
    # ------------------------------------------------------------------
    def _build_openai_chat(self) -> BaseChatModel:
        """
        Uses OpenAI-compatible API.
        Works with:
          - OpenAI
          - OpenRouter
        """
        return ChatOpenAI(
            model=self.config.llm_model_name,
            temperature=self.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            default_headers={
                # Required by OpenRouter for leaderboard & analytics
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_SITE_URL", "http://localhost"
                ),
                "X-Title": os.getenv(
                    "OPENROUTER_APP_NAME", "RAG-App"
                ),
            },
        )

    # ------------------------------------------------------------------
    # HUGGING FACE (Inference API)
    # ------------------------------------------------------------------
    def _build_hf_chat(self) -> Optional[BaseChatModel]:
        repo_id = (self.config.llm_model_name or "").strip()
        if not repo_id:
            print("[LLMBackend] Empty Hugging Face model name.")
            return None

        hf_token = (
            os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
        )

        if hf_token is None:
            print(
                "[LLMBackend] HF token not set. "
                "Public models may work, private/gated will fail."
            )

        try:
            base_llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return ChatHuggingFace(llm=base_llm)

        except Exception as e:
            print(f"[LLMBackend] HF model error: {e}")
            return None

    # ------------------------------------------------------------------
    # FACTORY
    # ------------------------------------------------------------------
    def get_langchain_llm(self) -> Optional[BaseChatModel]:
        provider = self.config.llm_provider

        if provider == "openai":
            return self._build_openai_chat()

        if provider == "huggingface":
            return self._build_hf_chat()

        print(f"[LLMBackend] Unknown provider: {provider}")
        return None

    # ------------------------------------------------------------------
    # HIGH-LEVEL CHAT API
    # ------------------------------------------------------------------
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        llm = self.get_langchain_llm()
        if llm is None:
            return (
                "LLM provider not configured correctly.\n\n"
                "- For OpenRouter/OpenAI: set OPENAI_API_KEY\n"
                "- For HuggingFace: set HF_TOKEN\n"
            )

        try:
            messages = [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
            response = llm.invoke(messages)
        except TypeError:
            # Fallback for non-chat models
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.invoke(prompt)
        except Exception as e:
            return f"[LLM ERROR] {e}"

        if hasattr(response, "content"):
            return response.content

        return str(response)