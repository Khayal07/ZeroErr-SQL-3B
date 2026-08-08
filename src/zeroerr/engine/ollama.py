"""Ollama-backed LLM engine (GGUF models served locally)."""

from __future__ import annotations

import httpx

from zeroerr.engine.base import LLMEngine


class OllamaEngine(LLMEngine):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 512) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def list_models(self) -> list[str]:
        resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]