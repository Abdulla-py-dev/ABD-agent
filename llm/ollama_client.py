"""
llm/ollama_client.py — Ollama Local LLM Client for ABD V2
============================================================
Connects to a locally running Ollama server and provides an interface
that mirrors GeminiClient so brain/agent.py can swap providers without
modification.

Capabilities:
  - Multi-turn conversation with history management (Ollama is stateless
    per request, so we maintain a ``messages`` list internally).
  - Native tool/function calling via Ollama's ``tools`` parameter.
  - Optional streaming: yields tokens as they arrive for a responsive
    terminal feel.
  - Clear error hierarchy with actionable messages.

Supported Ollama API endpoint:  POST /api/chat
Ollama docs: https://github.com/ollama/ollama/blob/main/docs/api.md

Usage (non-streaming)::

    client = OllamaClient(tool_schemas=TOOL_SCHEMAS, system_prompt=PROMPT)
    response = client.send_message("Hello!")
    print(client.get_text(response))

Usage (streaming)::

    for token in client.send_message_stream("Explain this file."):
        print(token, end="", flush=True)
    print()  # newline after stream ends
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

import requests

import config

logger = logging.getLogger("abd.ollama")


# ------------------------------------------------------------------
# Error types
# ------------------------------------------------------------------

class OllamaConnectionError(RuntimeError):
    """Raised when Ollama server is unreachable."""

class OllamaModelError(RuntimeError):
    """Raised when the requested model is not installed."""

class OllamaTimeoutError(RuntimeError):
    """Raised when the request exceeds the configured timeout."""

class OllamaResponseError(RuntimeError):
    """Raised when the response cannot be parsed."""


# ------------------------------------------------------------------
# Response wrapper
# ------------------------------------------------------------------

@dataclass
class OllamaResponse:
    """Wraps a complete Ollama /api/chat response.

    Mirrors the subset of the Gemini response interface that
    brain/agent.py uses, so the agent loop is provider-agnostic.
    """
    # The full message dict from Ollama (role, content, tool_calls)
    message: dict[str, Any] = field(default_factory=dict)
    # Whether the model finished generating (done=True in Ollama response)
    done: bool = True
    # Raw response body for debugging
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str | None:
        """Plain-text content from the assistant, or None."""
        content = self.message.get("content", "")
        return content if content else None

    @property
    def function_calls(self) -> list[dict[str, Any]]:
        """List of tool_call dicts, each with ``name`` and ``args``."""
        tool_calls = self.message.get("tool_calls", [])
        result = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            # Ollama may return arguments as a JSON string or a dict
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if name:
                result.append({"name": name, "args": arguments})
        return result


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class OllamaClient:
    """Ollama /api/chat client for ABD V2.

    Parameters
    ----------
    tool_schemas:
        List of tool schema dicts (provider-neutral JSON Schema format).
        Pass an empty list to disable tool calling.
    system_prompt:
        The system instruction string injected at the start of every
        conversation.
    model:
        Ollama model tag (default from config).
    base_url:
        Ollama server base URL (default from config).
    stream:
        If True, ``send_message`` collects tokens and invokes
        ``on_token`` for each one as they arrive.
        If False, blocks until the full response is ready.
    timeout:
        HTTP request timeout in seconds.
    on_token:
        Optional callable ``(token: str) -> None`` invoked for every
        streamed token. The client never writes to stdout directly;
        all display is the caller's responsibility.
    """

    def __init__(
        self,
        tool_schemas: list[dict[str, Any]],
        system_prompt: str,
        model: str | None = None,
        base_url: str | None = None,
        stream: bool | None = None,
        timeout: int | None = None,
        on_token: Any | None = None,
    ) -> None:
        self._model = model or config.OLLAMA_MODEL
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._stream = stream if stream is not None else config.OLLAMA_STREAM
        self._timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT
        self._tool_schemas = tool_schemas
        self._system_prompt = system_prompt
        # Callable invoked with each streamed token; None means no streaming output
        self._on_token = on_token

        # Conversation history — Ollama requires the full history each call
        self._messages: list[dict[str, Any]] = []

        # Prepend the system message
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

        logger.info(
            "OllamaClient initialised | model=%s base_url=%s stream=%s",
            self._model, self._base_url, self._stream,
        )

    # ------------------------------------------------------------------
    # Public API (mirrors GeminiClient)
    # ------------------------------------------------------------------

    def send_message(self, message: str) -> OllamaResponse:
        """Send a user message and return the complete response.

        If streaming is enabled this method still returns a complete
        ``OllamaResponse`` — it prints tokens to stdout as they arrive.
        Use ``send_message_stream`` if you need the token iterator directly.

        Raises
        ------
        OllamaConnectionError
            Ollama server is not running.
        OllamaModelError
            The model is not installed on the Ollama server.
        OllamaTimeoutError
            Request exceeded the configured timeout.
        OllamaResponseError
            Response could not be parsed.
        """
        self._messages.append({"role": "user", "content": message})

        if self._stream:
            # Collect streamed tokens; invoke on_token callback for each one
            full_content = ""
            final_response: OllamaResponse | None = None
            for token, response in self._stream_request():
                if self._on_token and token:
                    self._on_token(token)
                full_content += token
                final_response = response

            # Reconstruct a proper OllamaResponse with full content
            if final_response is None:
                final_response = OllamaResponse(
                    message={"role": "assistant", "content": full_content},
                    done=True,
                )
            else:
                # Merge streamed content into the message
                if not final_response.message.get("content"):
                    final_response.message["content"] = full_content

            self._append_assistant_message(final_response)
            return final_response
        else:
            response = self._blocking_request()
            self._append_assistant_message(response)
            return response

    def send_message_stream(self, message: str) -> Iterator[str]:
        """Send a user message and yield response tokens one by one.

        Stores the final complete message in history after the stream ends.

        Yields
        ------
        str
            Individual text tokens as they arrive from Ollama.
        """
        self._messages.append({"role": "user", "content": message})
        full_content = ""
        final_response: OllamaResponse | None = None

        for token, response in self._stream_request():
            full_content += token
            final_response = response
            yield token

        if final_response is None:
            final_response = OllamaResponse(
                message={"role": "assistant", "content": full_content},
                done=True,
            )
        else:
            if not final_response.message.get("content"):
                final_response.message["content"] = full_content

        self._append_assistant_message(final_response)

    def send_tool_result(
        self,
        function_name: str,
        result: Any,
    ) -> OllamaResponse:
        """Send the result of a tool call back to the model.

        Appends a ``tool`` role message to history and sends the full
        conversation to Ollama so it can continue reasoning.

        Parameters
        ----------
        function_name:
            The name of the tool that was executed.
        result:
            The tool's return value (must be JSON-serialisable).
        """
        # Ollama expects tool results as a "tool" role message
        tool_message: dict[str, Any] = {
            "role": "tool",
            "content": json.dumps(
                result if isinstance(result, dict) else {"result": result},
                ensure_ascii=False,
                default=str,
            ),
        }
        self._messages.append(tool_message)

        if self._stream:
            full_content = ""
            final_response: OllamaResponse | None = None
            for token, response in self._stream_request():
                if self._on_token and token:
                    self._on_token(token)
                full_content += token
                final_response = response

            if final_response is None:
                final_response = OllamaResponse(
                    message={"role": "assistant", "content": full_content},
                    done=True,
                )
            else:
                if not final_response.message.get("content"):
                    final_response.message["content"] = full_content

            self._append_assistant_message(final_response)
            return final_response
        else:
            response = self._blocking_request()
            self._append_assistant_message(response)
            return response

    def set_stream_callback(self, callback: Any | None) -> None:
        """Set or clear the streaming token callback.

        Parameters
        ----------
        callback:
            A callable ``(token: str) -> None`` that is invoked for each
            streamed token, or ``None`` to disable streaming output.
        """
        self._on_token = callback

    def reset_session(self) -> None:
        """Clear conversation history and start a fresh session."""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
        logger.info("OllamaClient session reset.")

    # ------------------------------------------------------------------
    # Helpers for inspecting responses (mirrors GeminiClient interface)
    # ------------------------------------------------------------------

    @staticmethod
    def get_text(response: OllamaResponse) -> str | None:
        """Extract plain text from a response, or None if there is none."""
        return response.text

    @staticmethod
    def get_function_calls(response: OllamaResponse) -> list[dict[str, Any]]:
        """Return a list of function-call dicts from a response.

        Each dict has:
            ``name``  — the function name (str)
            ``args``  — the arguments dict
        """
        return response.function_calls

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _build_payload(self, stream: bool) -> dict[str, Any]:
        """Build the /api/chat request body."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": stream,
        }
        if self._tool_schemas:
            # Ollama expects tools in OpenAI-compatible format
            payload["tools"] = self._tool_schemas
        return payload

    def _post(self, payload: dict[str, Any], stream: bool) -> requests.Response:
        """Make a POST to /api/chat and return the raw Response.

        Raises our custom error types for common failure modes.
        """
        url = f"{self._base_url}/api/chat"
        try:
            resp = requests.post(
                url,
                json=payload,
                stream=stream,
                timeout=self._timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self._timeout}s. "
                "Try increasing OLLAMA_TIMEOUT or using a smaller model."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Ollama HTTP request failed: {exc}"
            ) from exc

        if resp.status_code == 404:
            raise OllamaModelError(
                f"Model '{self._model}' is not installed on Ollama. "
                f"Install it with: ollama pull {self._model}"
            )
        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        return resp

    def _blocking_request(self) -> OllamaResponse:
        """Send a non-streaming request and return a complete OllamaResponse."""
        payload = self._build_payload(stream=False)
        resp = self._post(payload, stream=False)

        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                f"Ollama returned invalid JSON: {exc}"
            ) from exc

        message = body.get("message", {})
        done = body.get("done", True)

        logger.debug("Ollama response | done=%s message=%s", done, message)
        return OllamaResponse(message=message, done=done, raw=body)

    def _stream_request(self) -> Generator[tuple[str, OllamaResponse], None, None]:
        """Send a streaming request and yield (token, partial_response) pairs.

        The final yielded response contains the complete message including
        any tool_calls that were accumulated across chunks.
        """
        payload = self._build_payload(stream=True)
        resp = self._post(payload, stream=True)

        accumulated_content = ""
        accumulated_tool_calls: list[dict] = []
        last_body: dict[str, Any] = {}

        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON Ollama stream chunk: %s", line)
                    continue

                last_body = chunk
                msg = chunk.get("message", {})

                # Accumulate text content
                content_delta = msg.get("content", "")
                if content_delta:
                    accumulated_content += content_delta

                # Accumulate tool calls (usually arrive in the final chunk)
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    accumulated_tool_calls.extend(tool_calls)

                done = chunk.get("done", False)

                # Build the current response snapshot
                current_response = OllamaResponse(
                    message={
                        "role": "assistant",
                        "content": accumulated_content,
                        "tool_calls": accumulated_tool_calls,
                    },
                    done=done,
                    raw=chunk,
                )

                if content_delta:
                    yield content_delta, current_response

                if done:
                    # Yield a final response even if no new content token
                    if not content_delta:
                        yield "", current_response
                    break

        except requests.exceptions.ChunkedEncodingError as exc:
            raise OllamaConnectionError(
                f"Ollama stream was interrupted: {exc}"
            ) from exc

    def _append_assistant_message(self, response: OllamaResponse) -> None:
        """Add the assistant's reply to conversation history."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.message.get("content", ""),
        }
        # Include tool_calls in history if present (required by Ollama)
        tool_calls = response.message.get("tool_calls", [])
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)
