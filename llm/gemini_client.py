"""
llm/gemini_client.py — Google Gemini API Client for ABD V1
============================================================
Wraps the google-genai SDK (v2+) to provide a clean interface for the agent:
  - Initialises the client and chat session with tool declarations
  - Sends multi-turn conversations
  - Returns structured responses including function_call parts
  - Handles API errors gracefully

Migration note:
  Old SDK: google-generativeai  → import google.generativeai as genai
  New SDK: google-genai         → from google import genai
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

import config

logger = logging.getLogger("abd.gemini")

# Type alias for clarity
GenerateContentResponse = types.GenerateContentResponse


class GeminiClient:
    """Thin wrapper around the google-genai Chat client.

    Parameters
    ----------
    tool_declarations:
        A list of ``types.FunctionDeclaration`` objects describing the tools
        ABD can call.  Pass an empty list to disable tool calling.
    system_prompt:
        The system-level instruction string that defines ABD's personality.
    """

    def __init__(
        self,
        tool_declarations: list[types.FunctionDeclaration],
        system_prompt: str,
    ) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._system_prompt = system_prompt

        # Build GenerateContentConfig once — reused on every send_message call
        tools: list[types.Tool] | None = None
        if tool_declarations:
            tools = [types.Tool(function_declarations=tool_declarations)]

        self._gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            # Disable automatic function calling — ABD handles the loop manually
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        # Create a stateful chat session (preserves conversation history)
        self._chat = self._client.chats.create(
            model=config.GEMINI_MODEL,
            config=self._gen_config,
        )

        logger.info("GeminiClient initialised with model: %s", config.GEMINI_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_message(self, message: str | list[Any]) -> GenerateContentResponse:
        """Send a user message and return the raw Gemini response.

        Raises
        ------
        RuntimeError
            If the Gemini API returns an error or the response is blocked.
        """
        try:
            response = self._chat.send_message(message)
            return response
        except APIError as exc:
            raise RuntimeError(
                f"Gemini API error: {exc.message if hasattr(exc, 'message') else exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error calling Gemini: {exc}") from exc

    def send_tool_result(
        self,
        function_name: str,
        result: Any,
    ) -> GenerateContentResponse:
        """Send the result of a tool call back to Gemini so it can continue
        reasoning and produce a final natural-language response.

        Parameters
        ----------
        function_name:
            The name of the function that was called (must match the
            FunctionDeclaration name).
        result:
            The return value from the tool.  Must be JSON-serialisable.
        """
        # Build a Part containing the function response
        tool_response_part = types.Part.from_function_response(
            name=function_name,
            response=result if isinstance(result, dict) else {"result": result},
        )

        try:
            response = self._chat.send_message(tool_response_part)
            return response
        except APIError as exc:
            raise RuntimeError(
                f"Gemini API error sending tool result: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error sending tool result to Gemini: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers for inspecting responses
    # ------------------------------------------------------------------

    @staticmethod
    def get_text(response: GenerateContentResponse) -> str | None:
        """Extract plain text from a Gemini response, or None if there is none."""
        try:
            return response.text
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def get_function_calls(
        response: GenerateContentResponse,
    ) -> list[dict[str, Any]]:
        """Return a list of function-call dictionaries from a Gemini response.

        Each dict has:
            ``name``  — the function name (str)
            ``args``  — the arguments dict

        Uses the top-level ``response.function_calls`` property introduced
        in the google-genai SDK — no need to walk ``response.parts`` manually.
        """
        calls = []
        # response.function_calls is a list[FunctionCall] or None
        fc_list = response.function_calls
        if fc_list:
            for fc in fc_list:
                if fc.name:
                    calls.append(
                        {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                    )
        return calls

    def reset_session(self) -> None:
        """Start a fresh conversation (clears history)."""
        self._chat = self._client.chats.create(
            model=config.GEMINI_MODEL,
            config=self._gen_config,
        )
        logger.info("GeminiClient session reset.")
