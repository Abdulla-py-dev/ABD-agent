"""
tests/test_ollama_client.py — Unit Tests for the Ollama Client (ABD V2)
=========================================================================
Tests OllamaClient behaviour using mock HTTP responses so that no real
Ollama server or model is needed to run the test suite.

Coverage:
  - Successful plain-text chat response
  - Successful tool/function-call response
  - Connection refused (Ollama not running)
  - Model not found (404)
  - Request timeout
  - Invalid JSON response
  - Session reset clears history
  - send_tool_result appends correct message and continues conversation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# conftest.py sets env vars before any import — just import the client
import config  # noqa: F401 — ensures env is loaded
from llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelError,
    OllamaTimeoutError,
    OllamaResponseError,
    OllamaResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs) -> OllamaClient:
    """Return an OllamaClient with streaming disabled (simpler for tests)."""
    return OllamaClient(
        tool_schemas=[],
        system_prompt="You are ABD, a helpful assistant.",
        stream=False,
        **kwargs,
    )


def _mock_response(body: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response for a non-streaming request."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.text = json.dumps(body)
    return mock_resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> OllamaClient:
    return _make_client()


# ---------------------------------------------------------------------------
# 1. Successful plain-text response
# ---------------------------------------------------------------------------

class TestPlainTextResponse:

    def test_returns_text_from_response(self, client):
        body = {
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            response = client.send_message("Hi")

        assert response.done is True
        assert client.get_text(response) == "Hello! How can I help?"

    def test_history_grows_after_send(self, client):
        body = {
            "message": {"role": "assistant", "content": "Sure!"},
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            client.send_message("Do something")

        # system + user + assistant = 3 messages
        assert len(client._messages) == 3
        assert client._messages[-1]["role"] == "assistant"
        assert client._messages[-1]["content"] == "Sure!"

    def test_function_calls_empty_for_plain_response(self, client):
        body = {
            "message": {"role": "assistant", "content": "Done."},
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            response = client.send_message("hello")

        assert client.get_function_calls(response) == []


# ---------------------------------------------------------------------------
# 2. Tool / function-call response
# ---------------------------------------------------------------------------

class TestToolCallResponse:

    def test_parses_tool_calls(self, client):
        body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"directory": ""},
                        }
                    }
                ],
            },
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            response = client.send_message("List my files")

        calls = client.get_function_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_files"
        assert calls[0]["args"] == {"directory": ""}

    def test_tool_call_with_json_string_arguments(self, client):
        """Ollama sometimes returns arguments as a JSON string — we parse it."""
        body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "main.py"}',  # string, not dict
                        }
                    }
                ],
            },
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            response = client.send_message("Read main.py")

        calls = client.get_function_calls(response)
        assert calls[0]["args"] == {"path": "main.py"}

    def test_send_tool_result_appends_tool_message(self, client):
        # First, prime history with a user message
        first_body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}],
            },
            "done": True,
        }
        second_body = {
            "message": {"role": "assistant", "content": "I found 3 files."},
            "done": True,
        }

        with patch("requests.post", side_effect=[
            _mock_response(first_body),
            _mock_response(second_body),
        ]):
            client.send_message("List files")
            response = client.send_tool_result(
                "list_files",
                {"success": True, "items": [], "message": "3 items found"},
            )

        # History: system, user, assistant (tool call), tool, assistant (final)
        roles = [m["role"] for m in client._messages]
        assert "tool" in roles
        assert client.get_text(response) == "I found 3 files."


# ---------------------------------------------------------------------------
# 3. Connection / infrastructure errors
# ---------------------------------------------------------------------------

class TestConnectionErrors:

    def test_connection_refused_raises_connection_error(self, client):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                client.send_message("hello")
        assert "ollama serve" in str(exc_info.value).lower() or "connect" in str(exc_info.value).lower()

    def test_model_not_found_raises_model_error(self, client):
        mock_resp = _mock_response({"error": "model not found"}, status_code=404)
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaModelError) as exc_info:
                client.send_message("hello")
        assert "ollama pull" in str(exc_info.value).lower()

    def test_timeout_raises_timeout_error(self, client):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.Timeout("timed out")):
            with pytest.raises(OllamaTimeoutError) as exc_info:
                client.send_message("hello")
        assert "timeout" in str(exc_info.value).lower()

    def test_non_200_raises_response_error(self, client):
        mock_resp = _mock_response({"error": "internal error"}, status_code=500)
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaResponseError) as exc_info:
                client.send_message("hello")
        assert "500" in str(exc_info.value)

    def test_invalid_json_raises_response_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaResponseError):
                client.send_message("hello")


# ---------------------------------------------------------------------------
# 4. Session management
# ---------------------------------------------------------------------------

class TestSessionReset:

    def test_reset_clears_history(self, client):
        body = {
            "message": {"role": "assistant", "content": "Hi!"},
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            client.send_message("Hello")

        # Confirm history grew
        assert len(client._messages) > 1

        client.reset_session()

        # After reset: only the system message should remain
        assert len(client._messages) == 1
        assert client._messages[0]["role"] == "system"

    def test_reset_then_new_conversation(self, client):
        body = {
            "message": {"role": "assistant", "content": "Fresh start!"},
            "done": True,
        }
        with patch("requests.post", return_value=_mock_response(body)):
            client.send_message("First message")

        client.reset_session()

        with patch("requests.post", return_value=_mock_response(body)):
            response = client.send_message("Second message")

        assert client.get_text(response) == "Fresh start!"
        # system + user + assistant = 3
        assert len(client._messages) == 3


# ---------------------------------------------------------------------------
# 5. OllamaResponse helper properties
# ---------------------------------------------------------------------------

class TestOllamaResponse:

    def test_text_property_returns_content(self):
        r = OllamaResponse(message={"content": "Hello!"}, done=True)
        assert r.text == "Hello!"

    def test_text_property_none_when_empty(self):
        r = OllamaResponse(message={"content": ""}, done=True)
        assert r.text is None

    def test_function_calls_property_empty(self):
        r = OllamaResponse(message={"content": "Done"}, done=True)
        assert r.function_calls == []

    def test_function_calls_property_populated(self):
        r = OllamaResponse(
            message={
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "x.py"}}}
                ],
            },
            done=True,
        )
        calls = r.function_calls
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["args"] == {"path": "x.py"}


# ---------------------------------------------------------------------------
# 6. Payload construction
# ---------------------------------------------------------------------------

class TestPayloadConstruction:

    def test_tools_included_when_schemas_provided(self):
        schemas = [{"type": "function", "function": {"name": "list_files"}}]
        client = OllamaClient(
            tool_schemas=schemas,
            system_prompt="You are ABD.",
            stream=False,
        )
        payload = client._build_payload(stream=False)
        assert "tools" in payload
        assert payload["tools"] == schemas

    def test_tools_omitted_when_no_schemas(self, client):
        payload = client._build_payload(stream=False)
        assert "tools" not in payload

    def test_system_message_in_history(self, client):
        assert client._messages[0]["role"] == "system"
        assert "ABD" in client._messages[0]["content"]


# ---------------------------------------------------------------------------
# 7. keep_alive configuration
# ---------------------------------------------------------------------------

class TestOllamaKeepAlive:

    def test_default_keep_alive_is_10m(self):
        assert config.OLLAMA_KEEP_ALIVE == "10m"

    def test_keep_alive_in_payload(self, client):
        payload = client._build_payload(stream=False)
        assert "keep_alive" in payload
        assert payload["keep_alive"] == config.OLLAMA_KEEP_ALIVE

    def test_keep_alive_override_via_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
        import importlib
        importlib.reload(config)
        try:
            assert config.OLLAMA_KEEP_ALIVE == "5m"
            client = _make_client()
            payload = client._build_payload(stream=False)
            assert payload["keep_alive"] == "5m"
        finally:
            monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
            importlib.reload(config)

    def test_keep_alive_zero_disables(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")
        import importlib
        importlib.reload(config)
        try:
            assert config.OLLAMA_KEEP_ALIVE == "0"
            client = _make_client()
            payload = client._build_payload(stream=False)
            assert payload["keep_alive"] == "0"
        finally:
            monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
            importlib.reload(config)


# ---------------------------------------------------------------------------
# 8. Telemetry context
# ---------------------------------------------------------------------------

class TestTelemetryContext:

    def test_set_telemetry_context_stores_intent_and_schemas(self):
        client = _make_client()
        assert client._last_intent is None
        assert client._last_selected_schema_names == []

        client.set_telemetry_context("APP", ["open_application", "list_supported_apps"])
        assert client._last_intent == "APP"
        assert client._last_selected_schema_names == [
            "open_application",
            "list_supported_apps",
        ]

    def test_set_telemetry_context_overwrites_previous(self):
        client = _make_client()
        client.set_telemetry_context("CONVERSATIONAL", [])
        client.set_telemetry_context("FILE", ["list_files", "read_file"])
        assert client._last_intent == "FILE"
        assert client._last_selected_schema_names == ["list_files", "read_file"]

    def test_set_telemetry_context_empty_schemas(self):
        client = _make_client()
        client.set_telemetry_context("CONVERSATIONAL", [])
        assert client._last_selected_schema_names == []

    def test_log_ollama_metrics_includes_intent_and_schemas(self, client, caplog):
        import logging
        client.set_telemetry_context("APP", ["open_application"])
        body = {
            "total_duration": 2_000_000_000,
            "load_duration": 500_000_000,
            "prompt_eval_count": 245,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 10,
            "eval_duration": 500_000_000,
        }
        with caplog.at_level(logging.DEBUG, logger="abd.ollama"):
            client._log_ollama_metrics(body)
        record = caplog.records[-1]
        assert "intent=APP" in record.message
        assert "schemas=1" in record.message
        assert "open_application" in record.message
        assert "prompt_tokens=245" in record.message
        assert "load=0.50s" in record.message
        assert "total=2.00s" in record.message

    def test_log_ollama_metrics_handles_missing_context(self, client, caplog):
        import logging
        body = {
            "total_duration": 1_000_000_000,
            "load_duration": 200_000_000,
            "prompt_eval_count": 40,
            "prompt_eval_duration": 300_000_000,
            "eval_count": 5,
            "eval_duration": 200_000_000,
        }
        with caplog.at_level(logging.DEBUG, logger="abd.ollama"):
            client._log_ollama_metrics(body)
        record = caplog.records[-1]
        assert "intent=unknown" in record.message
        assert "schemas=0" in record.message
        assert "tools=[]" in record.message
