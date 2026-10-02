"""Local LLMs served by Ollama. Override with FRA_AGENT_MODEL / FRA_NUM_CTX / OLLAMA_HOST."""

import os

from langchain_ollama import ChatOllama

AGENT_MODEL = os.getenv("FRA_AGENT_MODEL", "qwen3:8b")
NUM_CTX = int(os.getenv("FRA_NUM_CTX", "16384"))


def chat_model(*, reasoning: bool = False, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or AGENT_MODEL,
        temperature=0,
        num_ctx=NUM_CTX,
        reasoning=reasoning,
        base_url=os.getenv("OLLAMA_HOST"),
    )
