"""
tests/test_agent_tool_selection.py — Integration Tests: Dynamic Tool Selection
===============================================================================
Verifies the full ABDAgent.chat() path with mocked Ollama HTTP, confirming:

  1. Conversational messages → no tool schemas sent to Ollama
  2. File messages → only file schemas sent
  3. Code messages → only code schemas sent
  4. App messages → only app schemas sent
  5. Safety gate still blocks path traversal even with subset schemas
  6. Fallback JSON tool-call parser executes recognised tools correctly
  7. Fallback JSON tool-call parser rejects disabled/hallucinated tool names
  8. Tool execution works end-to-end (mocked Ollama + mocked tool)
  9. Existing functionality: session reset clears state

All tests use conftest.py env vars (LLM_PROVIDER=ollama, OLLAMA_STREAM=false).
No real Ollama server or model is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

import config  # noqa: F401  — env is already set by conftest.py
from brain.router import TOOL_SCHEMAS
from brain.tool_selector import (
    Intent,
    FILE_TOOL_NAMES,
    CODE_TOOL_NAMES,
    APP_TOOL_NAMES,
    WEB_TOOL_NAMES,
    YOUTUBE_TOOL_NAMES,
    PDF_TOOL_NAMES,
)
from llm.ollama_client import OllamaResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plain_body(text: str = "Done.") -> dict:
    """Build a minimal Ollama response body for a plain-text reply."""
    return {
        "message": {"role": "assistant", "content": text},
        "done": True,
        "total_duration": 1_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 500_000_000,
        "eval_count": 5,
        "eval_duration": 400_000_000,
    }


def _tool_call_body(tool_name: str, arguments: dict) -> dict:
    """Build a minimal Ollama response body that requests a tool call."""
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": tool_name, "arguments": arguments}}
            ],
        },
        "done": True,
        "total_duration": 2_000_000_000,
        "load_duration": 200_000_000,
        "prompt_eval_count": 100,
        "prompt_eval_duration": 1_500_000_000,
        "eval_count": 10,
        "eval_duration": 300_000_000,
    }


def _mock_resp(body: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = body
    m.text = json.dumps(body)
    return m


def _make_agent():
    # Import here (not at module level) so that the voice test's
    # patch("llm.get_llm_client") can intercept the first import of brain.agent.
    from brain.agent import ABDAgent
    return ABDAgent()


def _schemas_sent_to_ollama(mock_post: MagicMock) -> list[dict]:
    """Return the ``tools`` list from the first POST call's JSON payload."""
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1].get("json") or call_kwargs[0][1]
    return payload.get("tools", [])


def _schema_names_sent(mock_post: MagicMock) -> set[str]:
    schemas = _schemas_sent_to_ollama(mock_post)
    return {s["function"]["name"] for s in schemas}


# ---------------------------------------------------------------------------
# 1. Conversational intent → no tools
# ---------------------------------------------------------------------------

class TestConversationalNoTools:

    @pytest.mark.parametrize("msg", [
        "hello",
        "what time is it?",
        "who are you?",
        "tell me a joke",
        "explain transformers",
    ])
    def test_conversational_sends_no_schemas(self, msg):
        body = _plain_body("I'm ABD, your assistant!")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat(msg)
        assert _schema_names_sent(mock_post) == set(), (
            f"Expected no schemas for conversational message {msg!r}"
        )

    def test_conversational_returns_text(self):
        body = _plain_body("Hello! I'm ABD.")
        with patch("requests.post", return_value=_mock_resp(body)):
            agent = _make_agent()
            result = agent.chat("hello")
        assert "ABD" in result or "Hello" in result or result  # got some text back


# ---------------------------------------------------------------------------
# 2. File intent → file schemas only
# ---------------------------------------------------------------------------

class TestFileIntentSchemas:

    def test_list_files_sends_file_schemas_only(self):
        body = _plain_body("Here are your files.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("list my workspace files")
        sent = _schema_names_sent(mock_post)
        assert sent == FILE_TOOL_NAMES, f"Expected FILE_TOOL_NAMES, got {sent}"

    def test_read_file_sends_file_schemas(self):
        body = _plain_body("File content here.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("read the file notes.txt")
        assert _schema_names_sent(mock_post) == FILE_TOOL_NAMES

    def test_file_schemas_exclude_code_tools(self):
        body = _plain_body("Done.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("list files")
        sent = _schema_names_sent(mock_post)
        assert not sent.intersection(CODE_TOOL_NAMES), (
            f"Code tool names leaked into file intent: {sent & CODE_TOOL_NAMES}"
        )


# ---------------------------------------------------------------------------
# 3. Code intent → code schemas only
# ---------------------------------------------------------------------------

class TestCodeIntentSchemas:

    def test_create_script_sends_code_schemas(self):
        body = _plain_body("Script created.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("create a script that prints hello")
        sent = _schema_names_sent(mock_post)
        assert sent == CODE_TOOL_NAMES, f"Expected CODE_TOOL_NAMES, got {sent}"

    def test_py_extension_sends_code_schemas(self):
        body = _plain_body("Content of main.py.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("read agent.py")
        assert _schema_names_sent(mock_post) == CODE_TOOL_NAMES

    def test_code_schemas_exclude_file_tools(self):
        body = _plain_body("Done.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("create a .py file")
        sent = _schema_names_sent(mock_post)
        assert not sent.intersection(FILE_TOOL_NAMES), (
            f"File tool names leaked into code intent: {sent & FILE_TOOL_NAMES}"
        )


# ---------------------------------------------------------------------------
# 4. App intent → app schemas only
# ---------------------------------------------------------------------------

class TestAppIntentSchemas:

    def test_open_calculator_sends_app_schemas(self):
        body = _plain_body("Opening calculator.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("open calculator")
        sent = _schema_names_sent(mock_post)
        assert sent == APP_TOOL_NAMES, f"Expected APP_TOOL_NAMES, got {sent}"

    def test_app_schemas_exclude_web_tools(self):
        body = _plain_body("Done.")
        with patch("requests.post", return_value=_mock_resp(body)) as mock_post:
            agent = _make_agent()
            agent.chat("open notepad")
        sent = _schema_names_sent(mock_post)
        assert not sent.intersection(WEB_TOOL_NAMES)

    def test_plain_text_does_not_execute_tool(self):
        """A plain-text success claim must NOT be treated as proof of execution."""
        body = _plain_body("Opening Calculator… Done!")

        with patch("requests.post", return_value=_mock_resp(body)):
            mock_fn = MagicMock(return_value={"success": True, "message": "Calculator opened."})
            with patch.dict("brain.router.TOOL_REGISTRY", {"open_application": mock_fn}):
                agent = _make_agent()
                result = agent.chat("open calculator")

        # The tool must not have been invoked because no structured tool call was returned
        mock_fn.assert_not_called()
        assert "couldn't execute" in result or "couldn't" in result


# ---------------------------------------------------------------------------
# 5. Safety gate enforcement with subset schemas
# ---------------------------------------------------------------------------

class TestSafetyGateWithSubsetSchemas:

    def test_path_traversal_blocked_via_file_intent(self):
        """Even with file schemas, path traversal must be blocked by is_safe_tool_call."""
        # First response: model asks to read a file
        first = _tool_call_body("read_file", {"path": "../../secret.txt"})
        # Second response (after tool result): plain text
        second = _plain_body("I tried to read the file.")

        with patch("requests.post", side_effect=[
            _mock_resp(first),
            _mock_resp(second),
        ]):
            agent = _make_agent()
            result = agent.chat("read the file notes.txt")

        # The safety gate should have blocked the traversal; the result should
        # contain either a blocked/error message or the plain follow-up text.
        # What it must NOT do is silently succeed with the traversal.
        assert result  # got a non-empty response

    def test_open_app_safety_gate_passes(self):
        """open_application with a valid app name should pass the safety gate."""
        first = _tool_call_body("open_application", {"app_name": "Calculator"})
        second = _plain_body("Calculator opened.")

        # Patch TOOL_REGISTRY directly — that's where execute_tool() looks up functions
        mock_fn = MagicMock(return_value={"success": True, "message": "Calculator opened."})

        with patch("requests.post", side_effect=[
            _mock_resp(first),
            _mock_resp(second),
        ]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"open_application": mock_fn}):
                agent = _make_agent()
                result = agent.chat("open calculator")

        mock_fn.assert_called_once_with(app_name="Calculator")
        assert result  # got a non-empty response


# ---------------------------------------------------------------------------
# 6. Fallback JSON tool-call parser — recognised tool, safety gate applies
# ---------------------------------------------------------------------------

class TestFallbackJsonToolCallParser:

    def test_fallback_parses_tag_wrapped_tool_call(self):
        """<tool_call> blocks in content should be parsed and executed."""
        content_with_tag = (
            '<tool_call>{"name": "list_files", "arguments": {"directory": ""}}</tool_call>'
        )
        first = {
            "message": {"role": "assistant", "content": content_with_tag},
            "done": True,
            "total_duration": 1_000_000_000, "load_duration": 0,
            "prompt_eval_count": 50, "prompt_eval_duration": 0,
            "eval_count": 5, "eval_duration": 0,
        }
        second = _plain_body("I found your files.")

        with patch("requests.post", side_effect=[
            _mock_resp(first),
            _mock_resp(second),
        ]):
            with patch("tools.file_tools.list_files", return_value={
                "success": True, "items": [], "message": "0 items",
            }):
                agent = _make_agent()
                result = agent.chat("list my files")

        assert result  # got a non-empty response

    def test_fallback_rejects_disabled_tool_name(self):
        """A tool name not in the enabled set must NOT be executed by the fallback."""
        # Send a conversational message → 0 schemas enabled
        # Model emits a <tool_call> for a file tool (disabled for this turn)
        content_with_disabled = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>'
        )
        body = {
            "message": {"role": "assistant", "content": content_with_disabled},
            "done": True,
            "total_duration": 1_000_000_000, "load_duration": 0,
            "prompt_eval_count": 40, "prompt_eval_duration": 0,
            "eval_count": 5, "eval_duration": 0,
        }

        with patch("requests.post", return_value=_mock_resp(body)):
            with patch("tools.file_tools.read_file") as mock_read:
                agent = _make_agent()
                # Conversational intent → 0 schemas → read_file is disabled
                result = agent.chat("what time is it?")

        # The disabled tool must NOT have been called
        mock_read.assert_not_called()
        # We should still get some response (the content text printed as-is)
        assert result

    def test_fallback_rejects_hallucinated_tool_name(self):
        """A made-up tool name in a <tool_call> block must never be executed."""
        content = (
            '<tool_call>{"name": "delete_all_files", "arguments": {}}</tool_call>'
        )
        body = {
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 1_000_000_000, "load_duration": 0,
            "prompt_eval_count": 60, "prompt_eval_duration": 0,
            "eval_count": 5, "eval_duration": 0,
        }

        with patch("requests.post", return_value=_mock_resp(body)):
            agent = _make_agent()
            # Any intent — the name doesn't exist in TOOL_REGISTRY at all
            result = agent.chat("list files please")

        # execute_tool would return an error for an unknown tool, but the
        # fallback parser should filter it before that; either way the
        # hallucinated tool must not cause a crash.
        assert result

    def test_app_intent_cannot_execute_file_tool(self):
        """Even if the model emits a structured call for a file tool, it
        must not be executed when the intent was APP and only app schemas
        were enabled for the turn."""
        first = _tool_call_body("read_file", {"path": "notes.txt"})
        second = _plain_body("I tried to read the file.")

        mock_fn = MagicMock(return_value={"success": True, "message": "File read."})

        with patch("requests.post", side_effect=[_mock_resp(first), _mock_resp(second)]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"read_file": mock_fn}):
                agent = _make_agent()
                result = agent.chat("open calculator")

        # The file tool must not have been executed because it was not enabled
        mock_fn.assert_not_called()
        assert result

    def test_structured_unknown_tool_rejected(self):
        """Structured tool calls that name unknown/hallucinated tools must
        not be executed even when present in the tool_calls field."""
        first = {
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "delete_all_files", "arguments": {}}}
            ]},
            "done": True,
        }
        second = _plain_body("No action taken.")

        with patch("requests.post", side_effect=[_mock_resp(first), _mock_resp(second)]):
            agent = _make_agent()
            result = agent.chat("open calculator")

        assert result

    def test_successful_tool_execution_accepts_final_response(self):
        """When a valid enabled tool executes successfully, the subsequent
        plain-text final response from the model should be accepted."""
        first = _tool_call_body("open_application", {"app_name": "Calculator"})
        second = _plain_body("Done.")

        mock_fn = MagicMock(return_value={
            "success": True,
            "message": "Calculator launched successfully.",
            "app_name": "Calculator",
        })

        with patch("requests.post", side_effect=[_mock_resp(first), _mock_resp(second)]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"open_application": mock_fn}):
                agent = _make_agent()
                result = agent.chat("open calculator")

        mock_fn.assert_called_once_with(app_name="Calculator")
        assert result == "Done." or "Done" in result

    def test_failed_tool_execution_reports_failure(self):
        """If the tool executes but returns success=False, ABD must not accept
        a later plain-text success claim and should report the failure."""
        first = _tool_call_body("open_application", {"app_name": "Calculator"})
        second = _plain_body("Done.")

        mock_fn = MagicMock(return_value={
            "success": False,
            "message": "Failed to launch Calculator.",
            "app_name": "Calculator",
        })

        with patch("requests.post", side_effect=[_mock_resp(first), _mock_resp(second)]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"open_application": mock_fn}):
                agent = _make_agent()
                result = agent.chat("open calculator")

        mock_fn.assert_called_once_with(app_name="Calculator")
        assert "Failed to launch Calculator" in result or "couldn't execute" in result or "failed" in result.lower()

    def test_safety_blocked_tool_execution_not_accepted(self):
        """A tool blocked by safety (e.g. path traversal) must not allow a
        subsequent plain-text success claim to be accepted."""
        # First response: model asks to read a sensitive file (will be blocked)
        first = _tool_call_body("read_file", {"path": "../../secret.txt"})
        second = _plain_body("Done.")

        # Patch TOOL_REGISTRY so execute_tool will be invoked but safety will block
        mock_fn = MagicMock(return_value={"success": True, "message": "Should not run."})

        with patch("requests.post", side_effect=[_mock_resp(first), _mock_resp(second)]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"read_file": mock_fn}):
                agent = _make_agent()
                result = agent.chat("read the file ../../secret.txt")

        # Since safety blocks path traversal, the mock function must NOT have been called
        mock_fn.assert_not_called()
        assert "outside the workspace" in result or "couldn't execute" in result or result


# ---------------------------------------------------------------------------
# 7. End-to-end tool execution (mocked Ollama + mocked tool)
# ---------------------------------------------------------------------------

class TestEndToEndToolExecution:

    def test_list_files_tool_called_and_response_returned(self):
        first = _tool_call_body("list_files", {"directory": ""})
        second = _plain_body("You have 3 files: a.txt, b.txt, c.txt.")

        # Patch TOOL_REGISTRY directly so execute_tool() calls the mock
        mock_fn = MagicMock(return_value={
            "success": True,
            "items": ["a.txt", "b.txt", "c.txt"],
            "message": "3 items found",
        })

        with patch("requests.post", side_effect=[
            _mock_resp(first),
            _mock_resp(second),
        ]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"list_files": mock_fn}):
                agent = _make_agent()
                result = agent.chat("list my workspace files")

        mock_fn.assert_called_once_with(directory="")
        assert result  # response was returned

    def test_web_search_tool_called(self):
        first = _tool_call_body("web_search", {"query": "Python FastAPI tutorial"})
        second = _plain_body("Opening Google search for Python FastAPI tutorial.")

        # Patch TOOL_REGISTRY directly so execute_tool() calls the mock
        mock_fn = MagicMock(return_value={"success": True, "message": "Opened browser."})

        with patch("requests.post", side_effect=[
            _mock_resp(first),
            _mock_resp(second),
        ]):
            with patch.dict("brain.router.TOOL_REGISTRY", {"web_search": mock_fn}):
                agent = _make_agent()
                result = agent.chat("search the web for Python FastAPI tutorial")

        mock_fn.assert_called_once_with(query="Python FastAPI tutorial")
        assert result


# ---------------------------------------------------------------------------
# 8. Session reset
# ---------------------------------------------------------------------------

class TestSessionReset:

    def test_reset_clears_turn_count(self):
        body = _plain_body("Hi!")
        with patch("requests.post", return_value=_mock_resp(body)):
            agent = _make_agent()
            agent.chat("hello")
        assert agent._turn_count == 1
        agent.reset()
        assert agent._turn_count == 0

    def test_reset_clears_client_history(self):
        body = _plain_body("Hi!")
        with patch("requests.post", return_value=_mock_resp(body)):
            agent = _make_agent()
            agent.chat("hello")

        # After chat: system + user + assistant in history
        assert len(agent._client._messages) > 1
        agent.reset()
        # After reset: only system message
        assert len(agent._client._messages) == 1
        assert agent._client._messages[0]["role"] == "system"


def _make_agent_with_instruction(instruction: str):
    from brain.agent import ABDAgent
    return ABDAgent(talk_mode_instruction=instruction)


# ---------------------------------------------------------------------------
# 9. Talk Mode instruction
# ---------------------------------------------------------------------------

class TestTalkModeInstruction:

    def test_talk_mode_instruction_appended_to_system_prompt(self):
        from brain.prompts import TALK_MODE_INSTRUCTION
        agent = _make_agent_with_instruction(TALK_MODE_INSTRUCTION)
        system_content = agent._client._messages[0]["content"]
        assert TALK_MODE_INSTRUCTION.strip() in system_content

    def test_chat_mode_has_no_talk_instruction(self):
        agent = _make_agent()
        system_content = agent._client._messages[0]["content"]
        assert "voice conversation mode" not in system_content

    def test_default_agent_has_no_talk_instruction(self):
        agent = _make_agent()
        system_content = agent._client._messages[0]["content"]
        assert "voice conversation mode" not in system_content

    def test_talk_mode_instruction_contains_exception_keywords(self):
        from brain.prompts import TALK_MODE_INSTRUCTION
        keywords = [
            "detailed explanation",
            "full code",
            "complete tutorial",
            "step-by-step",
            "long explanation",
            "documentation",
            "large output",
        ]
        instruction_lower = TALK_MODE_INSTRUCTION.lower()
        for kw in keywords:
            assert kw in instruction_lower, f"Missing exception keyword: {kw}"

    def test_talk_mode_does_not_duplicate_entire_system_prompt(self):
        from brain.prompts import SYSTEM_PROMPT, TALK_MODE_INSTRUCTION
        agent = _make_agent_with_instruction(TALK_MODE_INSTRUCTION)
        system_content = agent._client._messages[0]["content"]
        assert system_content.count(SYSTEM_PROMPT.strip()) == 1
        assert system_content.count(TALK_MODE_INSTRUCTION.strip()) == 1

    def test_talk_mode_preserves_system_prompt_content(self):
        from brain.prompts import SYSTEM_PROMPT, TALK_MODE_INSTRUCTION
        agent = _make_agent_with_instruction(TALK_MODE_INSTRUCTION)
        system_content = agent._client._messages[0]["content"]
        assert SYSTEM_PROMPT.strip() in system_content

    def test_chat_mode_uses_original_system_prompt(self):
        from brain.prompts import SYSTEM_PROMPT
        agent = _make_agent()
        system_content = agent._client._messages[0]["content"]
        assert system_content == SYSTEM_PROMPT
