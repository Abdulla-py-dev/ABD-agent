"""
brain/agent.py — ABD V2 Core Agent Loop
=========================================
Orchestrates multi-turn conversations between the user and the configured
LLM provider (Ollama or Gemini), handling tool calls in a safe, bounded loop.

V2 changes:
  - Uses get_llm_client() factory instead of directly importing GeminiClient.
  - Imports TOOL_SCHEMAS (for Ollama) and TOOL_DECLARATIONS (for Gemini) from router.
  - All other orchestration logic is unchanged from V1.

Flow per user message:
  1. Send user message to the LLM.
  2. If the LLM requests tool calls → execute them via the router.
  3. Send tool results back to the LLM.
  4. Repeat until the LLM returns a final text response (or max iterations).
  5. Return the final response text to the caller (main.py).
"""

from __future__ import annotations

import logging
from typing import Any

from llm import get_llm_client
from brain.router import TOOL_DECLARATIONS, TOOL_SCHEMAS, execute_tool
from brain.prompts import SYSTEM_PROMPT
from safety.permissions import log_action
import config

logger = logging.getLogger("abd.agent")


class ABDAgent:
    """The central ABD agent that manages conversation and tool orchestration.

    Works with any LLM provider supported by the factory in llm/__init__.py.

    Usage:
        agent = ABDAgent()
        response = agent.chat("Open Calculator")
        print(response)
    """

    def __init__(self) -> None:
        self._client = get_llm_client(
            tool_declarations=TOOL_DECLARATIONS,  # for Gemini
            tool_schemas=TOOL_SCHEMAS,            # for Ollama
            system_prompt=SYSTEM_PROMPT,
        )
        self._turn_count = 0
        logger.info(
            "ABD Agent initialised | provider=%s", config.LLM_PROVIDER
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process a user message and return ABD's final text response.

        Parameters
        ----------
        user_message:
            The raw text input from the user.

        Returns
        -------
        str
            ABD's final response text (after any tool calls are resolved).
        """
        self._turn_count += 1
        log_action("USER_INPUT", user_message[:200])
        logger.info("Turn %d — User: %s", self._turn_count, user_message[:100])

        try:
            response = self._client.send_message(user_message)
        except Exception as exc:
            logger.error("LLM send_message failed: %s", exc)
            # Surface the specific error to the user so they know what to fix
            return f"⚠️ {exc}"

        # Run the tool-call resolution loop
        final_text = self._resolve_tool_calls(response)
        log_action("ABD_RESPONSE", final_text[:200])
        return final_text

    def reset(self) -> None:
        """Start a fresh conversation session."""
        self._client.reset_session()
        self._turn_count = 0
        logger.info("Agent session reset.")

    def set_stream_callback(self, callback) -> None:
        """Attach or detach a streaming token callback on the LLM client.

        Only has an effect when the active provider is OllamaClient with
        streaming enabled.  Safe to call on GeminiClient (no-op).

        Parameters
        ----------
        callback:
            A callable ``(token: str) -> None``, or ``None`` to disable.
        """
        if hasattr(self._client, "set_stream_callback"):
            self._client.set_stream_callback(callback)

    # ------------------------------------------------------------------
    # Internal: tool-call resolution loop
    # ------------------------------------------------------------------

    def _resolve_tool_calls(self, response: Any) -> str:
        """Handle zero or more rounds of tool calls from the LLM.

        The LLM may return multiple function calls in one response, or chain
        tool calls across multiple turns.  We loop until either:
          (a) The LLM returns a plain text response (no more tool calls), or
          (b) We hit the MAX_TOOL_ITERATIONS safety limit.

        Returns the final plain-text response from the LLM.
        """
        iterations = 0

        while iterations < config.MAX_TOOL_ITERATIONS:
            function_calls = self._client.get_function_calls(response)

            if not function_calls:
                # LLM gave us a plain-text final answer
                text = self._client.get_text(response)
                if text:
                    return text
                # Edge case: no text and no function calls (e.g. safety block)
                return (
                    "I wasn't able to generate a response. "
                    "The content may have been blocked by safety filters."
                )

            # Execute all tool calls returned in this response
            iterations += 1
            logger.info(
                "Tool calls in iteration %d: %s",
                iterations,
                [fc["name"] for fc in function_calls],
            )

            # Execute each function call and send results back to the LLM
            for fc in function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                logger.info("Calling tool: %s | args: %s", tool_name, tool_args)
                tool_result = execute_tool(tool_name, tool_args)

                # Send the result back to the LLM and get its next response
                try:
                    response = self._client.send_tool_result(tool_name, tool_result)
                except Exception as exc:
                    logger.error("Failed to send tool result to LLM: %s", exc)
                    return (
                        f"⚠️ I executed '{tool_name}' but failed to send the "
                        f"result back to the AI: {exc}"
                    )

        # Safety limit reached
        logger.warning("Max tool iterations (%d) reached.", config.MAX_TOOL_ITERATIONS)
        text = self._client.get_text(response)
        if text:
            return text
        return (
            f"⚠️ I reached the maximum number of tool-call steps "
            f"({config.MAX_TOOL_ITERATIONS}). The task may be incomplete."
        )
