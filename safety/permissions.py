"""
safety/permissions.py — ABD V1 Safety Layer
=============================================
Provides:
  - Workspace path validation and path-traversal prevention
  - User confirmation prompts for destructive or risky operations
  - An action logger that records what ABD does
  - A gate function used by the router before executing any tool
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import config

# ------------------------------------------------------------------
# Action logger — writes to the configured log file
# ------------------------------------------------------------------
_action_logger = logging.getLogger("abd.actions")

if not _action_logger.handlers:
    _file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    _action_logger.addHandler(_file_handler)
    _action_logger.setLevel(logging.INFO)
    _action_logger.propagate = False  # don't double-log to root logger


def log_action(action: str, details: str = "") -> None:
    """Write an action record to the action log file."""
    _action_logger.info("%s | %s", action, details)


# ------------------------------------------------------------------
# Path validation
# ------------------------------------------------------------------

class PathTraversalError(ValueError):
    """Raised when a requested path would escape the workspace."""


def validate_workspace_path(requested_path: str | Path) -> Path:
    """Resolve *requested_path* and confirm it is inside WORKSPACE_DIR.

    Parameters
    ----------
    requested_path:
        A relative or absolute path string provided by the user or agent.

    Returns
    -------
    Path
        The fully resolved, safe absolute path.

    Raises
    ------
    PathTraversalError
        If the path escapes the workspace directory.
    FileNotFoundError
        (not raised here — callers decide whether missing files are OK)
    """
    workspace = config.WORKSPACE_DIR
    path = Path(requested_path)

    # If given a relative path, anchor it to the workspace
    if not path.is_absolute():
        resolved = (workspace / path).resolve()
    else:
        resolved = path.resolve()

    # The resolved path must start with the workspace root
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PathTraversalError(
            f"Access denied: '{requested_path}' is outside the workspace directory "
            f"({workspace}). ABD can only access files inside the workspace."
        )

    return resolved


def is_within_workspace(path: str | Path) -> bool:
    """Return True if *path* is safely inside the workspace, False otherwise."""
    try:
        validate_workspace_path(path)
        return True
    except (PathTraversalError, Exception):
        return False


# ------------------------------------------------------------------
# User confirmation
# ------------------------------------------------------------------

def require_confirmation(prompt: str) -> bool:
    """Print *prompt* and ask the user to confirm with y/n.

    Returns True if the user confirmed, False otherwise.
    This is a blocking call — it waits for terminal input.
    """
    print(f"\n⚠️  [CONFIRMATION REQUIRED]\n   {prompt}")
    try:
        answer = input("   Type 'yes' to confirm, anything else to cancel: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n   Cancelled.")
        return False

    confirmed = answer in ("yes", "y")
    if not confirmed:
        print("   Action cancelled.")
    return confirmed


# ------------------------------------------------------------------
# Tool-call safety gate
# ------------------------------------------------------------------

# Tools that always require explicit user confirmation before running
_CONFIRMATION_REQUIRED_TOOLS: set[str] = {
    "edit_file",
    "create_file",  # only if file already exists (handled inside the tool)
    "create_code_file",
}

# Tools that are blocked entirely (shell execution, etc.)
_BLOCKED_TOOLS: set[str] = set()  # Reserved for V2 — none blocked in V1


def is_safe_tool_call(tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
    """Determine whether a tool call is safe to execute.

    Returns
    -------
    (allowed: bool, reason: str)
        allowed — True if execution should proceed.
        reason  — Human-readable explanation (shown when not allowed).
    """
    if tool_name in _BLOCKED_TOOLS:
        reason = f"Tool '{tool_name}' is blocked in ABD V1 for safety reasons."
        log_action("BLOCKED", f"tool={tool_name} args={args}")
        return False, reason

    # Validate paths for file-touching tools
    path_arg = args.get("path") or args.get("file_path") or args.get("directory")
    if path_arg and tool_name not in ("open_application", "open_url", "web_search",
                                       "youtube_search"):
        try:
            validate_workspace_path(path_arg)
        except PathTraversalError as exc:
            log_action("PATH_TRAVERSAL_BLOCKED", f"tool={tool_name} path={path_arg}")
            return False, str(exc)

    return True, ""
