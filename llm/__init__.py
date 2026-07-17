"""
llm/__init__.py — LLM Provider Factory for ABD V2
====================================================
Provides a single ``get_llm_client()`` factory that returns either an
``OllamaClient`` (local, default) or a ``GeminiClient`` (cloud fallback)
based on ``config.LLM_PROVIDER``.

Both clients expose the same interface used by ``brain/agent.py``:
  - send_message(message)         → response
  - send_tool_result(name, result) → response
  - get_text(response)            → str | None
  - get_function_calls(response)  → list[dict]
  - reset_session()

The tool schema parameter is provider-specific:
  - Ollama: expects a list of plain JSON Schema tool dicts (TOOL_SCHEMAS)
  - Gemini: expects a list of FunctionDeclaration objects (TOOL_DECLARATIONS)

brain/router.py exports both; this factory picks the right one.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import config

logger = logging.getLogger("abd.llm")


def get_llm_client(
    tool_declarations: list[Any],   # FunctionDeclaration list for Gemini
    tool_schemas: list[dict],        # Plain JSON Schema list for Ollama
    system_prompt: str,
) -> Union["GeminiClient", "OllamaClient"]:  # noqa: F821  (resolved at runtime)
    """Return the configured LLM client.

    Parameters
    ----------
    tool_declarations:
        ``types.FunctionDeclaration`` objects for the Gemini client.
    tool_schemas:
        Provider-neutral JSON Schema dicts for the Ollama client.
    system_prompt:
        System instruction injected at the start of every conversation.

    Returns
    -------
    GeminiClient or OllamaClient
    """
    provider = config.LLM_PROVIDER

    if provider == "ollama":
        from llm.ollama_client import OllamaClient
        logger.info(
            "LLM provider: Ollama | model=%s | stream=%s",
            config.OLLAMA_MODEL, config.OLLAMA_STREAM,
        )
        return OllamaClient(
            tool_schemas=tool_schemas,
            system_prompt=system_prompt,
        )

    if provider == "gemini":
        from llm.gemini_client import GeminiClient
        logger.info(
            "LLM provider: Gemini | model=%s", config.GEMINI_MODEL
        )
        return GeminiClient(
            tool_declarations=tool_declarations,
            system_prompt=system_prompt,
        )

    # Should never reach here — config.py validates LLM_PROVIDER at startup
    raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'")
