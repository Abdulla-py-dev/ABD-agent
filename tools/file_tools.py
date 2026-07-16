"""
tools/file_tools.py — File Management Tools for ABD V1
=======================================================
Provides tools for listing, searching, reading, creating, and editing files.
All file operations are restricted to the configured workspace directory.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

from safety.permissions import validate_workspace_path, PathTraversalError, log_action
import config

logger = logging.getLogger("abd.file_tools")

# Maximum characters to read from a single file before truncating
_MAX_READ_CHARS = 50_000


# ------------------------------------------------------------------
# Tool: list_files
# ------------------------------------------------------------------

def list_files(directory: str = "") -> dict[str, Any]:
    """List files and folders inside the workspace (or a sub-directory).

    Parameters
    ----------
    directory:
        Sub-directory path relative to workspace, or empty string for the
        workspace root.

    Returns
    -------
    dict with keys:
        ``success``  — bool
        ``path``     — the directory that was listed
        ``items``    — list of dicts with name, type (file/directory), size
        ``message``  — human-readable summary or error
    """
    try:
        if directory:
            target = validate_workspace_path(directory)
        else:
            target = config.WORKSPACE_DIR

        if not target.exists():
            return {"success": False, "message": f"Directory not found: {target}"}
        if not target.is_dir():
            return {"success": False, "message": f"Not a directory: {target}"}

        items = []
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue  # skip hidden files in listings
            item_info: dict[str, Any] = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
            if entry.is_file():
                item_info["size_bytes"] = entry.stat().st_size
            items.append(item_info)

        log_action("LIST_FILES", str(target))
        return {
            "success": True,
            "path": str(target),
            "items": items,
            "message": f"Found {len(items)} item(s) in {target.name or 'workspace'}.",
        }

    except PathTraversalError as exc:
        return {"success": False, "message": str(exc)}
    except PermissionError:
        return {"success": False, "message": "Permission denied reading directory."}
    except Exception as exc:
        logger.exception("list_files error")
        return {"success": False, "message": f"Unexpected error: {exc}"}


# ------------------------------------------------------------------
# Tool: search_files
# ------------------------------------------------------------------

def search_files(query: str, directory: str = "") -> dict[str, Any]:
    """Search for files matching a name pattern inside the workspace.

    Parameters
    ----------
    query:
        File name pattern to search for (supports wildcards like *.py).
    directory:
        Sub-directory to search in (relative to workspace), or empty for all.

    Returns
    -------
    dict with keys: ``success``, ``matches`` (list of relative paths), ``message``
    """
    try:
        if directory:
            search_root = validate_workspace_path(directory)
        else:
            search_root = config.WORKSPACE_DIR

        if not search_root.exists():
            return {"success": False, "message": f"Directory not found: {search_root}"}

        # Add wildcard if the query has no glob character
        pattern = query if ("*" in query or "?" in query) else f"*{query}*"

        matches = []
        for root, dirs, files in os.walk(search_root):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                    full_path = Path(root) / filename
                    relative = full_path.relative_to(config.WORKSPACE_DIR)
                    matches.append(str(relative))

        log_action("SEARCH_FILES", f"query={query} root={search_root}")
        if matches:
            return {
                "success": True,
                "matches": matches,
                "message": f"Found {len(matches)} file(s) matching '{query}'.",
            }
        else:
            return {
                "success": True,
                "matches": [],
                "message": f"No files found matching '{query}'.",
            }

    except PathTraversalError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        logger.exception("search_files error")
        return {"success": False, "message": f"Unexpected error: {exc}"}


# ------------------------------------------------------------------
# Tool: read_file
# ------------------------------------------------------------------

def read_file(path: str) -> dict[str, Any]:
    """Read a text or code file from the workspace and return its contents.

    Parameters
    ----------
    path:
        Path to the file (relative to workspace or absolute).

    Returns
    -------
    dict with keys: ``success``, ``path``, ``content``, ``truncated``, ``message``
    """
    try:
        resolved = validate_workspace_path(path)

        if not resolved.exists():
            return {"success": False, "message": f"File not found: {resolved.name}"}
        if not resolved.is_file():
            return {"success": False, "message": f"'{path}' is a directory, not a file."}

        # Try reading as text; fall back to binary-safe reading
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = resolved.read_bytes().decode("utf-8", errors="replace")

        truncated = False
        if len(content) > _MAX_READ_CHARS:
            content = content[:_MAX_READ_CHARS]
            truncated = True

        log_action("READ_FILE", str(resolved))
        return {
            "success": True,
            "path": str(resolved),
            "filename": resolved.name,
            "content": content,
            "truncated": truncated,
            "message": (
                f"Read {resolved.name} ({len(content)} chars"
                + (", truncated to 50 000 chars)" if truncated else ")")
                + "."
            ),
        }

    except PathTraversalError as exc:
        return {"success": False, "message": str(exc)}
    except PermissionError:
        return {"success": False, "message": "Permission denied reading file."}
    except Exception as exc:
        logger.exception("read_file error")
        return {"success": False, "message": f"Unexpected error: {exc}"}


# ------------------------------------------------------------------
# Tool: create_file
# ------------------------------------------------------------------

def create_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
    """Create a new file in the workspace.

    If the file already exists and ``overwrite`` is False the tool will
    return a confirmation request — the agent should ask the user before
    calling again with overwrite=True.

    Parameters
    ----------
    path:
        File path relative to workspace.
    content:
        Text content to write.
    overwrite:
        Set True to overwrite an existing file (after user confirmation).

    Returns
    -------
    dict with keys: ``success``, ``path``, ``requires_confirmation``, ``message``
    """
    try:
        resolved = validate_workspace_path(path)

        if resolved.exists() and not overwrite:
            return {
                "success": False,
                "requires_confirmation": True,
                "path": str(resolved),
                "message": (
                    f"File '{resolved.name}' already exists. "
                    "Ask the user for permission before overwriting."
                ),
            }

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        log_action("CREATE_FILE", str(resolved))
        return {
            "success": True,
            "path": str(resolved),
            "filename": resolved.name,
            "message": f"File '{resolved.name}' created successfully in the workspace.",
        }

    except PathTraversalError as exc:
        return {"success": False, "message": str(exc)}
    except PermissionError:
        return {"success": False, "message": "Permission denied writing file."}
    except Exception as exc:
        logger.exception("create_file error")
        return {"success": False, "message": f"Unexpected error: {exc}"}


# ------------------------------------------------------------------
# Tool: edit_file
# ------------------------------------------------------------------

def edit_file(path: str, new_content: str) -> dict[str, Any]:
    """Overwrite an existing file with new content.

    This tool should only be called after the user has explicitly confirmed
    the edit.  The agent is responsible for seeking confirmation.

    Parameters
    ----------
    path:
        File path relative to workspace.
    new_content:
        The complete new content to write (replaces the entire file).

    Returns
    -------
    dict with keys: ``success``, ``path``, ``message``
    """
    try:
        resolved = validate_workspace_path(path)

        if not resolved.exists():
            return {
                "success": False,
                "message": f"File '{resolved.name}' does not exist. Use create_file to create it.",
            }
        if not resolved.is_file():
            return {"success": False, "message": f"'{path}' is not a file."}

        resolved.write_text(new_content, encoding="utf-8")

        log_action("EDIT_FILE", str(resolved))
        return {
            "success": True,
            "path": str(resolved),
            "filename": resolved.name,
            "message": f"File '{resolved.name}' updated successfully.",
        }

    except PathTraversalError as exc:
        return {"success": False, "message": str(exc)}
    except PermissionError:
        return {"success": False, "message": "Permission denied writing file."}
    except Exception as exc:
        logger.exception("edit_file error")
        return {"success": False, "message": f"Unexpected error: {exc}"}
