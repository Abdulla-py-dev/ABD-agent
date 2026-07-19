"""
tests/test_tool_selector.py — Unit Tests for brain/tool_selector.py
=====================================================================
Tests classify_intent() and select_tool_schemas() exhaustively.
No LLM or network required — pure Python unit tests.

Coverage:
  - All 8 intent categories
  - CODE vs FILE precedence (code keywords beat file keywords)
  - YOUTUBE vs APP precedence ("open youtube" → YOUTUBE, not APP)
  - WEB vs APP precedence (URLs → WEB, not APP)
  - Ambiguous action phrases → AMBIGUOUS_ACTION
  - select_tool_schemas() returns correct names per intent
  - AMBIGUOUS_ACTION returns all schemas
  - CONVERSATIONAL returns empty list
  - Tool names in returned schemas match TOOL_REGISTRY exactly
"""

from __future__ import annotations

import pytest

# conftest.py sets env vars before any import
import config  # noqa: F401
from brain.tool_selector import (
    Intent,
    classify_intent,
    select_tool_schemas,
    FILE_TOOL_NAMES,
    CODE_TOOL_NAMES,
    APP_TOOL_NAMES,
    WEB_TOOL_NAMES,
    YOUTUBE_TOOL_NAMES,
    PDF_TOOL_NAMES,
)
from brain.router import TOOL_SCHEMAS, TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names(schemas: list[dict]) -> set[str]:
    """Extract tool names from a schema list."""
    return {s["function"]["name"] for s in schemas}


# ---------------------------------------------------------------------------
# 1. classify_intent — CONVERSATIONAL
# ---------------------------------------------------------------------------

class TestConversationalIntent:

    @pytest.mark.parametrize("msg", [
        "hello",
        "hi there",
        "what time is it?",
        "who are you?",
        "what can you do?",
        "tell me a joke",
        "explain quantum computing",
        "what is 2 + 2?",
        "how are you",
        "thanks",
        "good morning",
    ])
    def test_conversational_messages(self, msg):
        assert classify_intent(msg) is Intent.CONVERSATIONAL


# ---------------------------------------------------------------------------
# 2. classify_intent — FILE
# ---------------------------------------------------------------------------

class TestFileIntent:

    @pytest.mark.parametrize("msg", [
        "list my workspace files",
        "list files",
        "what files do I have",
        "read the file notes.txt",
        "create file called report.txt",
        "search for files",
        "find a file named data.txt",
        "show files in workspace",
        "edit file config.txt",
        "make a new file",
    ])
    def test_file_messages(self, msg):
        assert classify_intent(msg) is Intent.FILE

    def test_txt_extension_routes_to_file(self):
        assert classify_intent("open notes.txt") is Intent.FILE

    def test_markdown_extension_routes_to_file(self):
        assert classify_intent("read README.md") is Intent.FILE


# ---------------------------------------------------------------------------
# 3. classify_intent — CODE (and CODE > FILE precedence)
# ---------------------------------------------------------------------------

class TestCodeIntent:

    @pytest.mark.parametrize("msg", [
        "create a script that prints hello",
        "write a python script",
        "read the code file main.py",
        "create a function in JavaScript",
        "list all code files",
        "generate a program to sort numbers",
        "edit code file utils.py",
        "show me the source code",
        "write a class for a bank account",
    ])
    def test_code_messages(self, msg):
        assert classify_intent(msg) is Intent.CODE

    @pytest.mark.parametrize("ext, msg", [
        (".py",   "read agent.py"),
        (".js",   "show me app.js"),
        (".ts",   "create a new .ts file"),
        (".java", "open Main.java"),
        (".cpp",  "create calculator.cpp"),
        (".html", "create index.html"),
        (".css",  "edit styles.css"),
        (".sh",   "show build.sh"),
        (".sql",  "read schema.sql"),
        (".yaml", "edit config.yaml"),
        (".json", "read package.json"),
    ])
    def test_code_extensions(self, ext, msg):
        assert classify_intent(msg) is Intent.CODE, f"Expected CODE for {msg!r}"

    def test_code_beats_file_with_py_extension(self):
        """'read a python file' has both .py (code) and 'file' keyword — CODE wins."""
        assert classify_intent("read the python file main.py") is Intent.CODE

    def test_code_beats_file_for_script_creation(self):
        """'create a script' should route to CODE, not FILE."""
        assert classify_intent("create a script that reads a file") is Intent.CODE

    def test_code_beats_file_for_workspace_code_files(self):
        assert classify_intent("list all code files in workspace") is Intent.CODE


# ---------------------------------------------------------------------------
# 4. classify_intent — APP
# ---------------------------------------------------------------------------

class TestAppIntent:

    @pytest.mark.parametrize("msg", [
        "open calculator",
        "open notepad",
        "open spotify",
        "open whatsapp",
        "open vs code",
        "launch the calculator",
        "start notepad",
        "list supported apps",
        "what apps can you open",
        "which applications can you open",
    ])
    def test_app_messages(self, msg):
        assert classify_intent(msg) is Intent.APP


# ---------------------------------------------------------------------------
# 5. classify_intent — WEB
# ---------------------------------------------------------------------------

class TestWebIntent:

    @pytest.mark.parametrize("msg", [
        "open https://github.com",
        "open http://example.com",
        "visit www.python.org",
        "search the web for Python tutorials",
        "google docker networking",
        "web search for machine learning",
        "open website github.com",
        "look up FastAPI documentation",
        "search online for Linux commands",
    ])
    def test_web_messages(self, msg):
        assert classify_intent(msg) is Intent.WEB


# ---------------------------------------------------------------------------
# 6. classify_intent — YOUTUBE (and YOUTUBE > APP precedence)
# ---------------------------------------------------------------------------

class TestYouTubeIntent:

    @pytest.mark.parametrize("msg", [
        "search youtube for python tutorials",
        "open youtube",
        "youtube search for machine learning",
        "find a video on youtube",
        "search yt for docker",
        "open the youtube app",
    ])
    def test_youtube_messages(self, msg):
        assert classify_intent(msg) is Intent.YOUTUBE

    def test_youtube_beats_app_for_open_youtube(self):
        """'open youtube' contains 'open' which could trigger APP, but YOUTUBE wins."""
        assert classify_intent("open youtube") is Intent.YOUTUBE

    def test_youtube_beats_web_for_youtube_search(self):
        """YouTube search is more specific than generic web search."""
        assert classify_intent("search youtube for python") is Intent.YOUTUBE


# ---------------------------------------------------------------------------
# 7. classify_intent — PDF
# ---------------------------------------------------------------------------

class TestPdfIntent:

    @pytest.mark.parametrize("msg", [
        "read and summarise this PDF: C:/papers/ml.pdf",
        "load the pdf",
        "open this pdf document",
        "summarize the pdf",
        "what is in the loaded pdf",
        "get active pdf info",
        "read the .pdf file",
        "the pdf document is too large",
    ])
    def test_pdf_messages(self, msg):
        assert classify_intent(msg) is Intent.PDF


# ---------------------------------------------------------------------------
# 8. classify_intent — AMBIGUOUS_ACTION
# ---------------------------------------------------------------------------

class TestAmbiguousActionIntent:

    @pytest.mark.parametrize("msg", [
        "do it",
        "execute",
        "go ahead",
        "proceed",
        "make it happen",
        "fix it",
        "do the thing",
        "run it",
    ])
    def test_ambiguous_action_messages(self, msg):
        assert classify_intent(msg) is Intent.AMBIGUOUS_ACTION


# ---------------------------------------------------------------------------
# 9. select_tool_schemas — correct names returned per intent
# ---------------------------------------------------------------------------

class TestSelectToolSchemas:

    def test_conversational_returns_empty(self):
        schemas = select_tool_schemas(Intent.CONVERSATIONAL, TOOL_SCHEMAS)
        assert schemas == []

    def test_file_returns_file_tools(self):
        schemas = select_tool_schemas(Intent.FILE, TOOL_SCHEMAS)
        assert _names(schemas) == FILE_TOOL_NAMES

    def test_code_returns_code_tools(self):
        schemas = select_tool_schemas(Intent.CODE, TOOL_SCHEMAS)
        assert _names(schemas) == CODE_TOOL_NAMES

    def test_app_returns_app_tools(self):
        schemas = select_tool_schemas(Intent.APP, TOOL_SCHEMAS)
        assert _names(schemas) == APP_TOOL_NAMES

    def test_web_returns_web_tools(self):
        schemas = select_tool_schemas(Intent.WEB, TOOL_SCHEMAS)
        assert _names(schemas) == WEB_TOOL_NAMES

    def test_youtube_returns_youtube_tools(self):
        schemas = select_tool_schemas(Intent.YOUTUBE, TOOL_SCHEMAS)
        assert _names(schemas) == YOUTUBE_TOOL_NAMES

    def test_pdf_returns_pdf_tools(self):
        schemas = select_tool_schemas(Intent.PDF, TOOL_SCHEMAS)
        assert _names(schemas) == PDF_TOOL_NAMES

    def test_ambiguous_action_returns_all_schemas(self):
        schemas = select_tool_schemas(Intent.AMBIGUOUS_ACTION, TOOL_SCHEMAS)
        assert len(schemas) == len(TOOL_SCHEMAS)

    def test_returned_names_exist_in_registry(self):
        """Every schema name returned by select_tool_schemas must be in TOOL_REGISTRY."""
        for intent in Intent:
            schemas = select_tool_schemas(intent, TOOL_SCHEMAS)
            for s in schemas:
                name = s["function"]["name"]
                assert name in TOOL_REGISTRY, (
                    f"Schema name {name!r} returned for {intent.name} "
                    f"is not in TOOL_REGISTRY"
                )

    def test_schemas_are_subset_of_all_schemas(self):
        """Selected schemas must always be a subset of TOOL_SCHEMAS."""
        for intent in Intent:
            selected = select_tool_schemas(intent, TOOL_SCHEMAS)
            assert all(s in TOOL_SCHEMAS for s in selected), (
                f"Intent {intent.name} returned a schema not in TOOL_SCHEMAS"
            )

    def test_empty_input_schemas_returns_empty(self):
        """If the full schema list is empty, all selections should be empty."""
        for intent in Intent:
            result = select_tool_schemas(intent, [])
            assert result == [] or intent is Intent.AMBIGUOUS_ACTION
