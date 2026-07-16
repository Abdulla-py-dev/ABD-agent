"""
tests/test_safety.py — Unit Tests for the Safety Layer
========================================================
Tests path validation, path traversal prevention, and workspace checks.
Runs without requiring a real Gemini API key (conftest.py handles that).
"""

import pytest
from pathlib import Path
from unittest.mock import patch

import config  # now importable because conftest set GEMINI_API_KEY in env


# ------------------------------------------------------------------
# Per-test workspace fixture
# ------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """A fresh temporary workspace directory, wired into config."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    original = config.WORKSPACE_DIR
    config.WORKSPACE_DIR = ws
    yield ws
    config.WORKSPACE_DIR = original


# ------------------------------------------------------------------
# Tests: validate_workspace_path
# ------------------------------------------------------------------

class TestValidateWorkspacePath:

    def test_valid_relative_path(self, workspace):
        from safety.permissions import validate_workspace_path
        result = validate_workspace_path("notes.txt")
        assert result == workspace / "notes.txt"

    def test_valid_subdirectory_path(self, workspace):
        from safety.permissions import validate_workspace_path
        result = validate_workspace_path("src/main.py")
        assert result == workspace / "src" / "main.py"

    def test_path_traversal_with_dotdot(self, workspace):
        from safety.permissions import validate_workspace_path, PathTraversalError
        with pytest.raises(PathTraversalError):
            validate_workspace_path("../secret.txt")

    def test_path_traversal_nested(self, workspace):
        from safety.permissions import validate_workspace_path, PathTraversalError
        with pytest.raises(PathTraversalError):
            validate_workspace_path("subdir/../../outside.txt")

    def test_absolute_path_outside_workspace(self, workspace):
        from safety.permissions import validate_workspace_path, PathTraversalError
        with pytest.raises(PathTraversalError):
            validate_workspace_path("C:/Windows/System32/secret.dll")

    def test_absolute_path_inside_workspace(self, workspace):
        from safety.permissions import validate_workspace_path
        inside_path = str(workspace / "data.txt")
        result = validate_workspace_path(inside_path)
        assert result == workspace / "data.txt"


class TestIsWithinWorkspace:

    def test_valid_path_returns_true(self, workspace):
        from safety.permissions import is_within_workspace
        assert is_within_workspace("file.txt") is True

    def test_traversal_returns_false(self, workspace):
        from safety.permissions import is_within_workspace
        assert is_within_workspace("../etc/passwd") is False

    def test_external_absolute_returns_false(self, workspace):
        from safety.permissions import is_within_workspace
        assert is_within_workspace("C:/Windows/notepad.exe") is False


# ------------------------------------------------------------------
# Tests: is_safe_tool_call
# ------------------------------------------------------------------

class TestIsSafeToolCall:

    def test_list_files_is_safe(self, workspace):
        from safety.permissions import is_safe_tool_call
        allowed, reason = is_safe_tool_call("list_files", {"directory": ""})
        assert allowed is True

    def test_read_file_safe_path(self, workspace):
        from safety.permissions import is_safe_tool_call
        allowed, reason = is_safe_tool_call("read_file", {"path": "notes.txt"})
        assert allowed is True

    def test_read_file_traversal_blocked(self, workspace):
        from safety.permissions import is_safe_tool_call
        allowed, reason = is_safe_tool_call("read_file", {"path": "../../secret.txt"})
        assert allowed is False
        assert "outside the workspace" in reason

    def test_open_application_no_path_check(self, workspace):
        """App launcher doesn't use path validation — should always pass gate."""
        from safety.permissions import is_safe_tool_call
        allowed, reason = is_safe_tool_call("open_application", {"app_name": "Calculator"})
        assert allowed is True

    def test_web_search_no_path_check(self, workspace):
        """Web search doesn't use path validation — should always pass gate."""
        from safety.permissions import is_safe_tool_call
        allowed, reason = is_safe_tool_call("web_search", {"query": "Python tutorials"})
        assert allowed is True
