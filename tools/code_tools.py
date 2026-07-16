"""
tools/code_tools.py — Coding Assistant Tools for ABD V1
========================================================
Provides tools for reading, creating, and analysing source-code files.
All file operations are workspace-restricted.
Code explanation and improvement suggestions are handled by the agent
(Gemini) once the file content is retrieved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from safety.permissions import validate_workspace_path, PathTraversalError, log_action
from tools.file_tools import read_file, create_file, edit_file
import config

logger = logging.getLogger("abd.code_tools")

# Extensions we treat as source code
_CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".sql", ".r", ".m", ".swift", ".kt", ".dart",
}


def _is_code_file(path: Path) -> bool:
    return path.suffix.lower() in _CODE_EXTENSIONS


# ------------------------------------------------------------------
# Tool: read_code_file
# ------------------------------------------------------------------

def read_code_file(path: str) -> dict[str, Any]:
    """Read a source-code file from the workspace.

    Wraps ``file_tools.read_file`` and adds code-specific metadata
    (language, line count) to the result.

    Parameters
    ----------
    path:
        Path to the code file (relative to workspace or absolute).

    Returns
    -------
    dict with keys: ``success``, ``path``, ``language``, ``line_count``,
                    ``content``, ``message``
    """
    result = read_file(path)
    if not result["success"]:
        return result

    file_path = Path(result["path"])
    language = file_path.suffix.lstrip(".").upper() or "text"
    content = result.get("content", "")
    line_count = content.count("\n") + 1 if content else 0

    result["language"] = language
    result["line_count"] = line_count

    if not _is_code_file(file_path):
        result["warning"] = (
            f"'{file_path.name}' doesn't have a recognised code extension. "
            "Content was returned, but it may not be source code."
        )

    log_action("READ_CODE", str(file_path))
    return result


# ------------------------------------------------------------------
# Tool: create_code_file
# ------------------------------------------------------------------

def create_code_file(
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a new source-code file in the workspace.

    Parameters
    ----------
    path:
        File path relative to workspace (e.g. "hello.py").
    content:
        The complete source-code content to write.
    overwrite:
        If True, overwrite an existing file (only after user confirmation).

    Returns
    -------
    dict with keys: ``success``, ``path``, ``language``,
                    ``requires_confirmation``, ``message``
    """
    result = create_file(path, content, overwrite=overwrite)

    if result.get("success"):
        file_path = Path(result["path"])
        result["language"] = file_path.suffix.lstrip(".").upper() or "text"

    log_action("CREATE_CODE", path)
    return result


# ------------------------------------------------------------------
# Tool: edit_code_file
# ------------------------------------------------------------------

def edit_code_file(path: str, new_content: str) -> dict[str, Any]:
    """Overwrite an existing code file with new content.

    This must only be called after explicit user confirmation.

    Parameters
    ----------
    path:
        File path relative to workspace.
    new_content:
        The complete new content to write.

    Returns
    -------
    dict with keys: ``success``, ``path``, ``message``
    """
    result = edit_file(path, new_content)
    if result.get("success"):
        file_path = Path(result["path"])
        result["language"] = file_path.suffix.lstrip(".").upper() or "text"
    log_action("EDIT_CODE", path)
    return result


# ------------------------------------------------------------------
# Tool: get_workspace_code_files
# ------------------------------------------------------------------

def get_workspace_code_files() -> dict[str, Any]:
    """List all source-code files in the workspace.

    Returns
    -------
    dict with keys: ``success``, ``files``, ``count``, ``message``
    """
    try:
        workspace = config.WORKSPACE_DIR
        code_files = []

        for file_path in sorted(workspace.rglob("*")):
            if file_path.is_file() and _is_code_file(file_path):
                relative = file_path.relative_to(workspace)
                code_files.append({
                    "path": str(relative),
                    "language": file_path.suffix.lstrip(".").upper(),
                    "size_bytes": file_path.stat().st_size,
                })

        return {
            "success": True,
            "files": code_files,
            "count": len(code_files),
            "message": (
                f"Found {len(code_files)} code file(s) in the workspace."
                if code_files
                else "No code files found in the workspace."
            ),
        }

    except Exception as exc:
        logger.exception("get_workspace_code_files error")
        return {"success": False, "message": f"Error listing code files: {exc}"}
