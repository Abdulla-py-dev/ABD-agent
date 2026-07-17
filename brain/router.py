"""
brain/router.py — Tool Registry and Dispatcher for ABD V2
==========================================================
Maps LLM function-call names → Python callable implementations.
Also defines tool schemas for both supported LLM providers.

Architecture:
  1. ``TOOL_REGISTRY``       — name → callable  (provider-agnostic)
  2. ``TOOL_DECLARATIONS``   — list of types.FunctionDeclaration  (Gemini)
  3. ``TOOL_SCHEMAS``        — list of plain dicts (Ollama / OpenAI format)
  4. ``execute_tool()``      — executes a tool after safety checks

The LLM factory in ``llm/__init__.py`` passes the correct schema format
to each provider's client.  The tool registry and execute_tool() function
are shared by both providers — the dispatcher is provider-agnostic.

Gemini schema note:
  - FunctionDeclaration uses ``parameters_json_schema`` (plain dict)
    with lowercase JSON Schema type names: "object", "string", "integer",
    "boolean" — NOT uppercase "OBJECT", "STRING", etc.
  - Import is: ``from google.genai import types``
"""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from tools.file_tools import list_files, search_files, read_file, create_file, edit_file
from tools.pdf_tools import load_pdf, get_pdf_text, get_active_pdf_info
from tools.app_tools import open_application, list_supported_apps
from tools.browser_tools import open_url, web_search
from tools.youtube_tools import youtube_search, open_youtube
from tools.code_tools import (
    read_code_file,
    create_code_file,
    edit_code_file,
    get_workspace_code_files,
)
from safety.permissions import is_safe_tool_call, log_action

logger = logging.getLogger("abd.router")


# ------------------------------------------------------------------
# 1. Tool Registry  (name → callable)
# ------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    # File management
    "list_files": list_files,
    "search_files": search_files,
    "read_file": read_file,
    "create_file": create_file,
    "edit_file": edit_file,
    # PDF tools
    "load_pdf": load_pdf,
    "get_pdf_text": get_pdf_text,
    "get_active_pdf_info": get_active_pdf_info,
    # App launcher
    "open_application": open_application,
    "list_supported_apps": list_supported_apps,
    # Browser tools
    "open_url": open_url,
    "web_search": web_search,
    # YouTube tools
    "youtube_search": youtube_search,
    "open_youtube": open_youtube,
    # Code tools
    "read_code_file": read_code_file,
    "create_code_file": create_code_file,
    "edit_code_file": edit_code_file,
    "get_workspace_code_files": get_workspace_code_files,
}


# ------------------------------------------------------------------
# 2. Gemini Tool Declarations  (JSON Schema function schemas)
#    Type names MUST be lowercase per JSON Schema spec:
#    "object", "string", "integer", "boolean", "array", "number"
# ------------------------------------------------------------------

def _str_prop(description: str) -> dict:
    return {"type": "string", "description": description}

def _int_prop(description: str) -> dict:
    return {"type": "integer", "description": description}

def _bool_prop(description: str) -> dict:
    return {"type": "boolean", "description": description}


TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [

    # ---- File Management -------------------------------------------

    types.FunctionDeclaration(
        name="list_files",
        description=(
            "List files and folders inside the workspace directory. "
            "Optionally specify a sub-directory to list."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "directory": _str_prop(
                    "Sub-directory path relative to workspace (optional). "
                    "Leave empty to list the workspace root."
                ),
            },
        },
    ),

    types.FunctionDeclaration(
        name="search_files",
        description=(
            "Search for files matching a name pattern inside the workspace. "
            "Supports wildcards like *.py."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": _str_prop(
                    "File name or pattern to search for (e.g. 'main.py', '*.txt')."
                ),
                "directory": _str_prop(
                    "Sub-directory to search in (optional, relative to workspace)."
                ),
            },
            "required": ["query"],
        },
    ),

    types.FunctionDeclaration(
        name="read_file",
        description=(
            "Read the full contents of a text or code file from the workspace. "
            "Returns the file content as a string."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop(
                    "Path to the file, relative to workspace (e.g. 'notes.txt', 'src/main.py')."
                ),
            },
            "required": ["path"],
        },
    ),

    types.FunctionDeclaration(
        name="create_file",
        description=(
            "Create a new file in the workspace with the given content. "
            "If the file already exists, set overwrite=true only after the user confirms."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop("File path relative to workspace (e.g. 'notes.txt')."),
                "content": _str_prop("The text content to write to the file."),
                "overwrite": _bool_prop(
                    "Set true to overwrite an existing file. "
                    "Only do this after explicit user confirmation."
                ),
            },
            "required": ["path", "content"],
        },
    ),

    types.FunctionDeclaration(
        name="edit_file",
        description=(
            "Overwrite an existing file in the workspace with new content. "
            "Only call this after the user has explicitly confirmed the edit."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop("Path to the file to edit (relative to workspace)."),
                "new_content": _str_prop("The complete new content to write to the file."),
            },
            "required": ["path", "new_content"],
        },
    ),

    # ---- PDF Tools --------------------------------------------------

    types.FunctionDeclaration(
        name="load_pdf",
        description=(
            "Load a PDF file and extract its text. "
            "Makes the PDF available for summarisation and Q&A. "
            "Use this when the user mentions a PDF file path or asks to read a PDF."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop(
                    "Path to the PDF file. Can be relative to workspace or an absolute path."
                ),
            },
            "required": ["path"],
        },
    ),

    types.FunctionDeclaration(
        name="get_pdf_text",
        description=(
            "Get a chunk of text from the currently loaded PDF. "
            "Use this to retrieve PDF content for summarisation or answering questions. "
            "Use chunk_index to page through large documents."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "chunk_index": _int_prop(
                    "Zero-based index of the text chunk to retrieve (default 0)."
                ),
            },
        },
    ),

    types.FunctionDeclaration(
        name="get_active_pdf_info",
        description="Get metadata about the currently loaded PDF (filename, page count, size).",
        parameters_json_schema={"type": "object", "properties": {}},
    ),

    # ---- Application Launcher ---------------------------------------

    types.FunctionDeclaration(
        name="open_application",
        description=(
            "Open a Windows application by name. "
            "Examples: 'Calculator', 'Notepad', 'VS Code', 'WhatsApp', 'Spotify'."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "app_name": _str_prop(
                    "Name of the application to open (e.g. 'Calculator', 'WhatsApp', 'VS Code')."
                ),
            },
            "required": ["app_name"],
        },
    ),

    types.FunctionDeclaration(
        name="list_supported_apps",
        description="List all applications that ABD can open.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),

    # ---- Browser Tools ----------------------------------------------

    types.FunctionDeclaration(
        name="open_url",
        description="Open a specific URL in the user's default web browser.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": _str_prop("The URL to open (e.g. 'https://github.com')."),
            },
            "required": ["url"],
        },
    ),

    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search Google for a query by opening a Google search URL in the browser. "
            "Use this when the user asks to 'search the web' or 'Google' something."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": _str_prop("The search query (e.g. 'Docker networking tutorial')."),
            },
            "required": ["query"],
        },
    ),

    # ---- YouTube Tools ----------------------------------------------

    types.FunctionDeclaration(
        name="youtube_search",
        description=(
            "Search YouTube for a query by opening a YouTube search URL in the browser. "
            "Use this when the user asks to search YouTube."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": _str_prop("The YouTube search query (e.g. 'Python FastAPI tutorial')."),
            },
            "required": ["query"],
        },
    ),

    types.FunctionDeclaration(
        name="open_youtube",
        description="Open the YouTube homepage in the default browser.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),

    # ---- Code Tools -------------------------------------------------

    types.FunctionDeclaration(
        name="read_code_file",
        description=(
            "Read a source-code file from the workspace. "
            "Returns content with language and line count metadata. "
            "Use this before asking Gemini to explain or improve code."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop("Path to the code file (relative to workspace)."),
            },
            "required": ["path"],
        },
    ),

    types.FunctionDeclaration(
        name="create_code_file",
        description=(
            "Create a new source-code file in the workspace. "
            "Use for generating new scripts, programs, or configuration files."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop("File path relative to workspace (e.g. 'calculator.py')."),
                "content": _str_prop("The complete source-code content to write."),
                "overwrite": _bool_prop(
                    "Set true to overwrite an existing file (only after user confirmation)."
                ),
            },
            "required": ["path", "content"],
        },
    ),

    types.FunctionDeclaration(
        name="edit_code_file",
        description=(
            "Overwrite an existing code file with new content. "
            "Only call this after explicit user confirmation."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": _str_prop("Path to the code file (relative to workspace)."),
                "new_content": _str_prop("The complete new source code."),
            },
            "required": ["path", "new_content"],
        },
    ),

    types.FunctionDeclaration(
        name="get_workspace_code_files",
        description="List all source-code files in the workspace directory.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]


# ------------------------------------------------------------------
# 3. Tool Schemas — provider-neutral (Ollama / OpenAI function format)
# ------------------------------------------------------------------
# These mirror TOOL_DECLARATIONS exactly but use the OpenAI-compatible
# format that Ollama and most local LLM servers understand.
# Add a new tool here AND in TOOL_DECLARATIONS to keep both in sync.
# ------------------------------------------------------------------

def _schema_str(description: str) -> dict:
    return {"type": "string", "description": description}

def _schema_int(description: str) -> dict:
    return {"type": "integer", "description": description}

def _schema_bool(description: str) -> dict:
    return {"type": "boolean", "description": description}


TOOL_SCHEMAS: list[dict] = [

    # ---- File Management -------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and folders inside the workspace directory. "
                "Optionally specify a sub-directory to list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": _schema_str(
                        "Sub-directory path relative to workspace (optional). "
                        "Leave empty to list the workspace root."
                    ),
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for files matching a name pattern inside the workspace. "
                "Supports wildcards like *.py."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _schema_str(
                        "File name or pattern to search for (e.g. 'main.py', '*.txt')."
                    ),
                    "directory": _schema_str(
                        "Sub-directory to search in (optional, relative to workspace)."
                    ),
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a text or code file from the workspace. "
                "Returns the file content as a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str(
                        "Path to the file, relative to workspace (e.g. 'notes.txt', 'src/main.py')."
                    ),
                },
                "required": ["path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": (
                "Create a new file in the workspace with the given content. "
                "If the file already exists, set overwrite=true only after the user confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str("File path relative to workspace (e.g. 'notes.txt')."),
                    "content": _schema_str("The text content to write to the file."),
                    "overwrite": _schema_bool(
                        "Set true to overwrite an existing file. "
                        "Only do this after explicit user confirmation."
                    ),
                },
                "required": ["path", "content"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Overwrite an existing file in the workspace with new content. "
                "Only call this after the user has explicitly confirmed the edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str("Path to the file to edit (relative to workspace)."),
                    "new_content": _schema_str("The complete new content to write to the file."),
                },
                "required": ["path", "new_content"],
            },
        },
    },

    # ---- PDF Tools --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "load_pdf",
            "description": (
                "Load a PDF file and extract its text. "
                "Makes the PDF available for summarisation and Q&A. "
                "Use this when the user mentions a PDF file path or asks to read a PDF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str(
                        "Path to the PDF file. Can be relative to workspace or an absolute path."
                    ),
                },
                "required": ["path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_pdf_text",
            "description": (
                "Get a chunk of text from the currently loaded PDF. "
                "Use this to retrieve PDF content for summarisation or answering questions. "
                "Use chunk_index to page through large documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_index": _schema_int(
                        "Zero-based index of the text chunk to retrieve (default 0)."
                    ),
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_active_pdf_info",
            "description": "Get metadata about the currently loaded PDF (filename, page count, size).",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ---- Application Launcher ---------------------------------------

    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": (
                "Open a Windows application by name. "
                "Examples: 'Calculator', 'Notepad', 'VS Code', 'WhatsApp', 'Spotify'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": _schema_str(
                        "Name of the application to open (e.g. 'Calculator', 'WhatsApp', 'VS Code')."
                    ),
                },
                "required": ["app_name"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "list_supported_apps",
            "description": "List all applications that ABD can open.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ---- Browser Tools ----------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a specific URL in the user's default web browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": _schema_str("The URL to open (e.g. 'https://github.com')."),
                },
                "required": ["url"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search Google for a query by opening a Google search URL in the browser. "
                "Use this when the user asks to 'search the web' or 'Google' something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _schema_str("The search query (e.g. 'Docker networking tutorial')."),
                },
                "required": ["query"],
            },
        },
    },

    # ---- YouTube Tools ----------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": (
                "Search YouTube for a query by opening a YouTube search URL in the browser. "
                "Use this when the user asks to search YouTube."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _schema_str("The YouTube search query (e.g. 'Python FastAPI tutorial')."),
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "open_youtube",
            "description": "Open the YouTube homepage in the default browser.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ---- Code Tools -------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "read_code_file",
            "description": (
                "Read a source-code file from the workspace. "
                "Returns content with language and line count metadata. "
                "Use this before asking the LLM to explain or improve code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str("Path to the code file (relative to workspace)."),
                },
                "required": ["path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "create_code_file",
            "description": (
                "Create a new source-code file in the workspace. "
                "Use for generating new scripts, programs, or configuration files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str("File path relative to workspace (e.g. 'calculator.py')."),
                    "content": _schema_str("The complete source-code content to write."),
                    "overwrite": _schema_bool(
                        "Set true to overwrite an existing file (only after user confirmation)."
                    ),
                },
                "required": ["path", "content"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "edit_code_file",
            "description": (
                "Overwrite an existing code file with new content. "
                "Only call this after explicit user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _schema_str("Path to the code file (relative to workspace)."),
                    "new_content": _schema_str("The complete new source code."),
                },
                "required": ["path", "new_content"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_workspace_code_files",
            "description": "List all source-code files in the workspace directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ------------------------------------------------------------------
# 4. Tool Executor
# ------------------------------------------------------------------

def execute_tool(name: str, args: dict[str, Any]) -> Any:
    """Look up and execute a tool by name after safety validation.

    Parameters
    ----------
    name:
        The function name from Gemini's function_call.
    args:
        The arguments dict from Gemini's function_call.

    Returns
    -------
    The tool's return value (always a JSON-serialisable dict).
    """
    if name not in TOOL_REGISTRY:
        log_action("UNKNOWN_TOOL", name)
        return {
            "success": False,
            "message": f"Unknown tool '{name}'. This is likely a Gemini hallucination.",
        }

    # Safety gate
    allowed, reason = is_safe_tool_call(name, args)
    if not allowed:
        return {"success": False, "message": reason}

    log_action("EXECUTE_TOOL", f"{name}({args})")
    logger.info("Executing tool: %s | args: %s", name, args)

    try:
        fn = TOOL_REGISTRY[name]
        result = fn(**args)
        return result
    except TypeError as exc:
        # Mismatched arguments — likely a Gemini schema issue
        logger.warning("Tool %s called with wrong args %s: %s", name, args, exc)
        return {
            "success": False,
            "message": f"Tool '{name}' received unexpected arguments: {exc}",
        }
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        return {
            "success": False,
            "message": f"Tool '{name}' encountered an error: {exc}",
        }
