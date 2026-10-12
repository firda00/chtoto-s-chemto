"""Точка входа по CONVENTIONS.md: все вызовы LLM идут через `complete()`.

Реализация живёт в `local_llm.client`; модуль сохранён для единообразия с остальными проектами.
"""

from local_llm.client import LLMClient, LLMSettings, MockLLM, complete, get_llm

__all__ = ["LLMClient", "LLMSettings", "MockLLM", "complete", "get_llm"]
