"""Request-scoped dependency factories."""

from __future__ import annotations

from functools import lru_cache

from zeroerr.config import Settings, settings as _settings
from zeroerr.engine.base import LLMEngine
from zeroerr.engine.ollama import OllamaEngine
from zeroerr.guardrail.sandbox import SandboxRegistry


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _settings


@lru_cache(maxsize=1)
def get_registry() -> SandboxRegistry:
    return SandboxRegistry(_settings.sandbox_data_dir)


@lru_cache(maxsize=1)
def get_engine() -> LLMEngine:
    if _settings.engine_backend == "ollama":
        return OllamaEngine(model=_settings.ollama_model, base_url=_settings.ollama_base_url)
    raise ValueError(f"unsupported engine_backend: {_settings.engine_backend}")