"""
brain/agent.py — ABD V2 Core Agent Loop
=========================================
Orchestrates multi-turn conversations between the user and the configured
LLM provider (Ollama or Gemini), handling tool calls in a safe, bounded loop.

V2 changes:
  - Uses get_llm_client() factory instead of directly importing GeminiClient.
  - Imports TOOL_SCHEMAS (for Ollama) and TOOL_DECLARATIONS (for Gemini) from router.
  - All other orchestration logic is unchanged from V1.

V2 performance optimisation (dynamic tool selection):
  - Before each send_message() call, classify_intent() scans the user message
    with a fast keyword heuristic (~0 ms) and select_tool_schemas() returns
    only the relevant tool subset.
  - Conversational messages get no tool schemas → ~20x faster Ollama prefill.
  - Tool execution, safety gate, and conversation history are unchanged.
  - Only OllamaClient.update_tool_schemas() is called; GeminiClient is
    unaffected (it always receives the full TOOL_DECLARATIONS on init).

Flow per user message:
  1. classify_intent(message) → select the minimal tool subset.
  2. Send user message to the LLM (with the selected schemas).
  3. If the LLM requests tool calls → execute them via the router.
  4. Send tool results back to the LLM.
  5. Repeat until the LLM returns a final text response (or max iterations).
  6. Return the final response text to the caller (main.py).
"""

from __future__ import annotations

import logging
from typing import Any

from llm import get_llm_client
from brain.router import TOOL_DECLARATIONS, TOOL_SCHEMAS, execute_tool
from brain.prompts import SYSTEM_PROMPT
from brain.tool_selector import classify_intent, select_tool_schemas, Intent
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
            tool_schemas=TOOL_SCHEMAS,            # for Ollama (full set on init)
            system_prompt=SYSTEM_PROMPT,
        )
        self._turn_count = 0
        # Keep a reference to the full schema list so select_tool_schemas()
        # always has the complete catalogue to filter from.
        self._all_tool_schemas: list[dict] = list(TOOL_SCHEMAS)

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

        # Classify intent for this turn (used for dynamic schema selection
        # and to ensure action intents require an actual tool call).
        intent = classify_intent(user_message)
        # Keep last intent for the resolution loop to enforce action semantics
        self._last_intent = intent

        # --- Dynamic tool selection (Ollama only) ---
        # Gemini has its own schema type; we don't touch GeminiClient here.
        if config.LLM_PROVIDER == "ollama" and hasattr(self._client, "update_tool_schemas"):
            selected = select_tool_schemas(intent, self._all_tool_schemas)
            self._client.update_tool_schemas(selected)
            logger.info(
                "Turn %d intent=%s → %d/%d schemas",
                self._turn_count, intent.name,
                len(selected), len(self._all_tool_schemas),
            )

        try:
            response = self._client.send_message(user_message)
        except Exception as exc:
            logger.error("LLM send_message failed: %s", exc)
            # Surface the specific error to the user so they know what to fix
            return f"⚠️ {exc}"

        # Run the tool-call resolution loop (enforces that action intents
        # require valid tool calls; plain-text success claims are not enough)
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

        retry_attempted = False
        tool_executed_successfully = False
        last_tool_result: Any | None = None
        last_tool_name: str | None = None
        while iterations < config.MAX_TOOL_ITERATIONS:
            # Log raw response for debugging (safely at DEBUG level)
            raw = getattr(response, "raw", None)
            logger.debug("Raw LLM response (truncated): %s", raw if raw is None else str(raw)[:1000])

            function_calls = self._client.get_function_calls(response)
            logger.debug("Native function_calls from LLM: %s", function_calls)

            # Filter out any function calls that are not in the enabled tool set
            enabled: frozenset = getattr(response, "_enabled_tool_names", frozenset())
            allowed_calls = [fc for fc in function_calls if fc.get("name") in enabled]
            if function_calls and not allowed_calls:
                logger.warning(
                    "LLM requested tools that are not enabled for this turn: %s (enabled: %s)",
                    [fc.get("name") for fc in function_calls], sorted(enabled)
                )

            if not allowed_calls:
                # No valid structured tool calls were produced by the model.
                # If the last intent was an action intent, a plain-text
                # success claim must NOT be treated as execution proof.
                text = self._client.get_text(response)
                if text:
                    # Action intents require explicit tool calls
                    if hasattr(self, "_last_intent") and self._last_intent in (
                        Intent.APP,
                        Intent.FILE,
                        Intent.CODE,
                        Intent.WEB,
                        Intent.YOUTUBE,
                        Intent.PDF,
                        Intent.AMBIGUOUS_ACTION,
                    ):
                        logger.debug(
                            "Action intent %s produced plain text without tool calls; rejecting as proof of execution.",
                            self._last_intent.name,
                        )
                        # If a tool was already executed successfully, accept
                        # the final plain-text response from the model.
                        if tool_executed_successfully:
                            logger.debug(
                                "Tool %s executed successfully; accepting final text response.",
                                last_tool_name,
                            )
                            return text

                        # If a tool was executed but failed, report the failure
                        if last_tool_result is not None and not tool_executed_successfully:
                            msg = None
                            if isinstance(last_tool_result, dict):
                                msg = last_tool_result.get("message")
                            if not msg:
                                msg = str(last_tool_result)
                            return (
                                f"I attempted to execute '{last_tool_name}' but it failed: {msg}"
                            )

                        # Try a bounded retry asking the model to emit a structured
                        # tool call. This is a single, controlled retry to recover
                        # from models that respond in plain text by mistake.
                        if not retry_attempted:
                            logger.info("No tool call produced for action intent; requesting structured tool call (1 retry).")
                            retry_attempted = True
                            try:
                                response = self._client.send_message(
                                    "To execute the user's requested action I need a structured tool call. "
                                    "Please reply with a single function/tool call using only the available tools and include no extra confirmation text."
                                )
                                # Loop back to process the new response
                                continue
                            except Exception as exc:
                                logger.error("Retry request for structured tool call failed: %s", exc)
                                return (
                                    "I couldn't execute that action because no valid tool call was generated."
                                )

                        # Retry already attempted — be honest about failure
                        return (
                            "I couldn't execute that action because no valid tool call was generated."
                        )

                    # Non-action intents may return plain text normally
                    return text

                # Edge case: no text and no function calls (e.g. safety block)
                return (
                    "I wasn't able to generate a response. "
                    "The content may have been blocked by safety filters."
                )

            # Execute all validated tool calls returned in this response
            iterations += 1
            logger.info(
                "Tool calls in iteration %d: %s",
                iterations,
                [fc["name"] for fc in allowed_calls],
            )

            # Execute each validated function call and send results back to the LLM
            for fc in allowed_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                logger.info("Calling tool: %s | args: %s", tool_name, tool_args)
                logger.debug("Invoking execute_tool for %s", tool_name)
                tool_result = execute_tool(tool_name, tool_args)
                logger.debug("Tool execution result for %s: %s", tool_name, tool_result)
                # Track last tool execution and whether it succeeded
                last_tool_name = tool_name
                last_tool_result = tool_result
                try:
                    if isinstance(tool_result, dict) and bool(tool_result.get("success", False)):
                        tool_executed_successfully = True
                except Exception:
                    # Keep conservative default (False) if unexpected structure
                    tool_executed_successfully = False

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
