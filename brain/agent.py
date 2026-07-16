"""
brain/agent.py — ABD V1 Core Agent Loop
=========================================
Orchestrates multi-turn conversations between the user and Gemini,
handling tool calls in a safe, bounded loop.

Flow per user message:
  1. Send user message to Gemini.
  2. If Gemini requests tool calls → execute them via the router.
  3. Send tool results back to Gemini.
  4. Repeat until Gemini returns a final text response (or max iterations).
  5. Return the final response text to the caller (main.py).
"""

from __future__ import annotations

import logging
from typing import Any

from llm.gemini_client import GeminiClient
from brain.router import TOOL_DECLARATIONS, execute_tool
from brain.prompts import SYSTEM_PROMPT
from safety.permissions import log_action
import config

logger = logging.getLogger("abd.agent")


class ABDAgent:
    """The central ABD agent that manages conversation and tool orchestration.

    Usage:
        agent = ABDAgent()
        response = agent.chat("Open Calculator")
        print(response)
    """

    def __init__(self) -> None:
        self._client = GeminiClient(
            tool_declarations=TOOL_DECLARATIONS,
            system_prompt=SYSTEM_PROMPT,
        )
        self._turn_count = 0
        logger.info("ABD Agent initialised.")

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
        except RuntimeError as exc:
            logger.error("Gemini send_message failed: %s", exc)
            return f"⚠️ I couldn't reach the Gemini API: {exc}"

        # Run the tool-call resolution loop
        final_text = self._resolve_tool_calls(response)
        log_action("ABD_RESPONSE", final_text[:200])
        return final_text

    def reset(self) -> None:
        """Start a fresh conversation session."""
        self._client.reset_session()
        self._turn_count = 0
        logger.info("Agent session reset.")

    # ------------------------------------------------------------------
    # Internal: tool-call resolution loop
    # ------------------------------------------------------------------

    def _resolve_tool_calls(self, response: Any) -> str:
        """Handle zero or more rounds of tool calls from Gemini.

        Gemini may return multiple function calls in one response, or chain
        tool calls across multiple turns.  We loop until either:
          (a) Gemini returns a plain text response (no more tool calls), or
          (b) We hit the MAX_TOOL_ITERATIONS safety limit.

        Returns the final plain-text response from Gemini.
        """
        iterations = 0

        while iterations < config.MAX_TOOL_ITERATIONS:
            function_calls = self._client.get_function_calls(response)

            if not function_calls:
                # Gemini gave us a plain-text final answer
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

            # Execute each function call and collect results
            # Gemini expects one tool_response per function_call in the same order
            for fc in function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                logger.info("Calling tool: %s | args: %s", tool_name, tool_args)
                tool_result = execute_tool(tool_name, tool_args)

                # Send the result back to Gemini and get its next response
                try:
                    response = self._client.send_tool_result(tool_name, tool_result)
                except RuntimeError as exc:
                    logger.error("Failed to send tool result to Gemini: %s", exc)
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
