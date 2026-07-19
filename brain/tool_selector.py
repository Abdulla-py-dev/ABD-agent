"""
brain/tool_selector.py — Dynamic Tool-Schema Selector for ABD V2
=================================================================
Classifies user intent with a fast keyword scan (O(n), zero LLM calls)
and returns only the relevant subset of TOOL_SCHEMAS for that turn.

Sending 16 tool schemas on every request forces the model to tokenise and
attend to ~800+ extra tokens during prefill even for trivial questions.
Selecting a relevant subset — or no tools at all for conversation — cuts
prompt_eval_count dramatically and speeds up every response.

Intent classes and their tool subsets
--------------------------------------
CONVERSATIONAL   → [] (no tools)
FILE             → list_files, search_files, read_file, create_file, edit_file
CODE             → read_code_file, create_code_file, edit_code_file,
                   get_workspace_code_files
                   (CODE wins over FILE when code-specific keywords are present)
APP              → open_application, list_supported_apps
WEB              → open_url, web_search
YOUTUBE          → youtube_search, open_youtube
PDF              → load_pdf, get_pdf_text, get_active_pdf_info
AMBIGUOUS_ACTION → all tools (safe fallback)

Precedence (high → low):
  YOUTUBE > WEB > PDF > APP > CODE > FILE > AMBIGUOUS_ACTION > CONVERSATIONAL

Usage::

    from brain.tool_selector import classify_intent, select_tool_schemas
    from brain.router import TOOL_SCHEMAS

    intent  = classify_intent("open calculator")
    schemas = select_tool_schemas(intent, TOOL_SCHEMAS)
"""

from __future__ import annotations

import logging
import re
from enum import Enum, auto
from typing import Sequence

logger = logging.getLogger("abd.tool_selector")


# ---------------------------------------------------------------------------
# Intent labels
# ---------------------------------------------------------------------------

class Intent(Enum):
    CONVERSATIONAL   = auto()
    FILE             = auto()
    CODE             = auto()
    APP              = auto()
    WEB              = auto()
    YOUTUBE          = auto()
    PDF              = auto()
    AMBIGUOUS_ACTION = auto()


# ---------------------------------------------------------------------------
# Tool name groups (must match brain/router.py TOOL_REGISTRY keys exactly)
# ---------------------------------------------------------------------------

FILE_TOOL_NAMES: frozenset[str] = frozenset({
    "list_files",
    "search_files",
    "read_file",
    "create_file",
    "edit_file",
})

CODE_TOOL_NAMES: frozenset[str] = frozenset({
    "read_code_file",
    "create_code_file",
    "edit_code_file",
    "get_workspace_code_files",
})

APP_TOOL_NAMES: frozenset[str] = frozenset({
    "open_application",
    "list_supported_apps",
})

WEB_TOOL_NAMES: frozenset[str] = frozenset({
    "open_url",
    "web_search",
})

YOUTUBE_TOOL_NAMES: frozenset[str] = frozenset({
    "youtube_search",
    "open_youtube",
})

PDF_TOOL_NAMES: frozenset[str] = frozenset({
    "load_pdf",
    "get_pdf_text",
    "get_active_pdf_info",
})

# Maps intent → the names of tools to include
_INTENT_TOOL_NAMES: dict[Intent, frozenset[str]] = {
    Intent.CONVERSATIONAL:   frozenset(),
    Intent.FILE:             FILE_TOOL_NAMES,
    Intent.CODE:             CODE_TOOL_NAMES,
    Intent.APP:              APP_TOOL_NAMES,
    Intent.WEB:              WEB_TOOL_NAMES,
    Intent.YOUTUBE:          YOUTUBE_TOOL_NAMES,
    Intent.PDF:              PDF_TOOL_NAMES,
    Intent.AMBIGUOUS_ACTION: frozenset(),  # handled specially → all tools
}


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------

# Each set is checked against the normalised (lower-case, stripped) message.
# Use whole-word patterns where needed to avoid false matches.

_YOUTUBE_KEYWORDS: frozenset[str] = frozenset({
    "youtube", "yt ", " yt", "^yt$", "watch on youtube",
    "youtube video", "youtube search",
})

_WEB_KEYWORDS: frozenset[str] = frozenset({
    "http://", "https://", "www.",
    "open url", "open link", "open website", "open webpage",
    "open browser", "browse to",
    "web search", "search the web", "search online", "search internet",
    "google ", " google", "google for", "google search",
    "look up",
    "visit ", "go to website", "go to http",
})

_PDF_KEYWORDS: frozenset[str] = frozenset({
    ".pdf", "pdf file", "pdf document",
    "load pdf", "read pdf", "open pdf", "summarise pdf", "summarize pdf",
    "pdf summary", "pdf text", "pdf content", "active pdf",
    "loaded pdf", "the pdf", "this pdf", "my pdf",
})

_APP_KEYWORDS: frozenset[str] = frozenset({
    "open app", "open application", "launch app", "launch application",
    "start app", "run app",
    # Generic verb + known app names (supports "launch X", "start X", "run X")
    "launch ", "start ",
    "open calculator", "open notepad", "open spotify", "open whatsapp",
    "open vs code", "open vscode", "open chrome", "open firefox",
    "open edge", "open word", "open excel", "open powerpoint",
    "open paint", "open task manager",
    "list apps", "list supported apps", "what apps", "which apps",
    "what applications", "which applications", "apps can you open",
    "apps you can open", "applications can you open",
})

# Code-specific keywords — CODE wins over FILE when these appear
_CODE_KEYWORDS: frozenset[str] = frozenset({
    # explicit code-file keywords
    "code file", "source file", "source code",
    "create code", "write code", "generate code",
    "read code", "edit code", "show code",
    "create a script", "write a script", "generate a script",
    "create a program", "write a program", "generate a program",
    "python script", "python file", "python program",
    "create a function", "write a function",
    "create a class", "write a class",
    "list code files", "workspace code",
    # extension-based triggers (file read/create by extension)
    ".py", ".js", ".ts", ".java", ".cpp", ".c ", ".c,", ".h ",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".jsx", ".tsx", ".vue", ".sql", ".sh",
    ".bat", ".ps1", ".yaml", ".yml", ".json", ".toml",
    # context clues that strongly imply code
    "function ", "class ", "import ", "def ", "async ",
    "module", "library", "package",
})

# FILE keywords — only reached if CODE didn't match
_FILE_KEYWORDS: frozenset[str] = frozenset({
    "file", "folder", "directory", "workspace",
    "list files", "list folder", "list my files",
    "show files", "what files",
    "read file", "read the file", "open file",
    "create file", "make file", "new file",
    "edit file", "modify file", "update file", "change file",
    "search file", "find file",
    "notes.txt", ".txt", ".md", ".log", ".csv",
})

# Ambiguous action phrases that suggest the user wants ABD to *do* something
# but the intent is unclear — safe fallback is all tools
_AMBIGUOUS_ACTION_KEYWORDS: frozenset[str] = frozenset({
    "do it", "do that", "do something",
    "execute", "run it", "go ahead",
    "proceed", "continue", "make it happen",
    "fix it", "fix that",
    "make it", "do the thing",
})


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _matches(text: str, keywords: frozenset[str]) -> bool:
    """Return True if any keyword appears in *text* (substring match)."""
    return any(kw in text for kw in keywords)


def classify_intent(message: str) -> Intent:
    """Classify *message* into one of the ``Intent`` categories.

    Rules are evaluated in priority order; the first match wins:

    1. YOUTUBE  — highest priority (prevents "open youtube" → APP)
    2. WEB      — URL / search patterns
    3. PDF      — .pdf references or explicit pdf commands
    4. APP      — "open <app>" or "list apps"
    5. CODE     — code-specific extensions / verbs (beats FILE)
    6. FILE     — generic file / folder / workspace keywords
    7. AMBIGUOUS_ACTION — vague "do it / execute" phrases
    8. CONVERSATIONAL — default (no tools needed)

    Parameters
    ----------
    message:
        The raw user message string.

    Returns
    -------
    Intent
        The detected intent label.
    """
    norm = _normalise(message)

    if _matches(norm, _YOUTUBE_KEYWORDS):
        return Intent.YOUTUBE

    if _matches(norm, _WEB_KEYWORDS):
        return Intent.WEB

    if _matches(norm, _PDF_KEYWORDS):
        return Intent.PDF

    if _matches(norm, _APP_KEYWORDS):
        return Intent.APP

    # CODE is checked before FILE so that "create a .py file" → CODE, not FILE
    if _matches(norm, _CODE_KEYWORDS):
        return Intent.CODE

    if _matches(norm, _FILE_KEYWORDS):
        return Intent.FILE

    if _matches(norm, _AMBIGUOUS_ACTION_KEYWORDS):
        return Intent.AMBIGUOUS_ACTION

    return Intent.CONVERSATIONAL


def select_tool_schemas(
    intent: Intent,
    all_schemas: Sequence[dict],
) -> list[dict]:
    """Return the subset of *all_schemas* relevant for *intent*.

    Parameters
    ----------
    intent:
        The intent returned by ``classify_intent()``.
    all_schemas:
        The full ``TOOL_SCHEMAS`` list from ``brain/router.py``.

    Returns
    -------
    list[dict]
        Subset of schemas to send to Ollama for this turn.
        An empty list means "no tool calling" for this turn.
    """
    if intent is Intent.AMBIGUOUS_ACTION:
        # Safe fallback — send everything so ABD doesn't miss a needed tool
        logger.debug("Tool selector: AMBIGUOUS_ACTION → all %d schemas", len(all_schemas))
        return list(all_schemas)

    tool_names = _INTENT_TOOL_NAMES[intent]

    if not tool_names:
        # CONVERSATIONAL
        logger.debug("Tool selector: %s → 0 schemas (no tools)", intent.name)
        return []

    selected = [
        s for s in all_schemas
        if s.get("function", {}).get("name") in tool_names
    ]
    logger.debug(
        "Tool selector: %s → %d/%d schemas %s",
        intent.name, len(selected), len(all_schemas),
        sorted(s["function"]["name"] for s in selected),
    )
    return selected
