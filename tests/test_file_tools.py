"""
tests/test_file_tools.py — Unit Tests for File Management Tools
================================================================
Tests list_files, read_file, create_file, edit_file, search_files.
Runs without a real Gemini API key (conftest.py handles that).
"""

import pytest
from pathlib import Path

import config  # importable because conftest.py set GEMINI_API_KEY in env


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
# Tests: list_files
# ------------------------------------------------------------------

class TestListFiles:

    def test_empty_workspace(self, workspace):
        from tools.file_tools import list_files
        result = list_files()
        assert result["success"] is True
        visible_items = [i for i in result["items"] if not i["name"].startswith(".")]
        assert len(visible_items) == 0

    def test_lists_created_files(self, workspace):
        from tools.file_tools import list_files
        (workspace / "hello.txt").write_text("hello")
        (workspace / "world.py").write_text("print('hi')")

        result = list_files()
        assert result["success"] is True
        names = [i["name"] for i in result["items"]]
        assert "hello.txt" in names
        assert "world.py" in names

    def test_lists_subdirectory(self, workspace):
        from tools.file_tools import list_files
        subdir = workspace / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("# main")

        result = list_files("src")
        assert result["success"] is True
        names = [i["name"] for i in result["items"]]
        assert "main.py" in names

    def test_nonexistent_directory(self, workspace):
        from tools.file_tools import list_files
        result = list_files("does_not_exist")
        assert result["success"] is False
        assert "not found" in result["message"].lower()


# ------------------------------------------------------------------
# Tests: read_file
# ------------------------------------------------------------------

class TestReadFile:

    def test_read_existing_file(self, workspace):
        from tools.file_tools import read_file
        (workspace / "data.txt").write_text("Hello ABD!", encoding="utf-8")
        result = read_file("data.txt")
        assert result["success"] is True
        assert "Hello ABD!" in result["content"]

    def test_read_nonexistent_file(self, workspace):
        from tools.file_tools import read_file
        result = read_file("missing.txt")
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_read_traversal_blocked(self, workspace):
        from tools.file_tools import read_file
        result = read_file("../../etc/passwd")
        assert result["success"] is False
        # Either "outside" or "denied" depending on OS
        msg = result["message"].lower()
        assert "outside" in msg or "denied" in msg or "access" in msg

    def test_read_directory_rejected(self, workspace):
        from tools.file_tools import read_file
        (workspace / "mydir").mkdir()
        result = read_file("mydir")
        assert result["success"] is False


# ------------------------------------------------------------------
# Tests: create_file
# ------------------------------------------------------------------

class TestCreateFile:

    def test_create_new_file(self, workspace):
        from tools.file_tools import create_file
        result = create_file("hello.py", "print('Hello ABD!')")
        assert result["success"] is True
        assert (workspace / "hello.py").read_text() == "print('Hello ABD!')"

    def test_create_existing_without_overwrite(self, workspace):
        from tools.file_tools import create_file
        (workspace / "exists.txt").write_text("original")
        result = create_file("exists.txt", "new content", overwrite=False)
        assert result["success"] is False
        assert result.get("requires_confirmation") is True
        # Original content must be unchanged
        assert (workspace / "exists.txt").read_text() == "original"

    def test_create_existing_with_overwrite(self, workspace):
        from tools.file_tools import create_file
        (workspace / "exists.txt").write_text("original")
        result = create_file("exists.txt", "new content", overwrite=True)
        assert result["success"] is True
        assert (workspace / "exists.txt").read_text() == "new content"

    def test_create_traversal_blocked(self, workspace):
        from tools.file_tools import create_file
        result = create_file("../outside.txt", "bad content")
        assert result["success"] is False

    def test_create_nested_path(self, workspace):
        from tools.file_tools import create_file
        result = create_file("src/utils.py", "# utilities")
        assert result["success"] is True
        assert (workspace / "src" / "utils.py").exists()


# ------------------------------------------------------------------
# Tests: edit_file
# ------------------------------------------------------------------

class TestEditFile:

    def test_edit_existing_file(self, workspace):
        from tools.file_tools import edit_file
        (workspace / "code.py").write_text("original")
        result = edit_file("code.py", "updated content")
        assert result["success"] is True
        assert (workspace / "code.py").read_text() == "updated content"

    def test_edit_nonexistent_file(self, workspace):
        from tools.file_tools import edit_file
        result = edit_file("ghost.py", "content")
        assert result["success"] is False
        assert "does not exist" in result["message"].lower()

    def test_edit_traversal_blocked(self, workspace):
        from tools.file_tools import edit_file
        result = edit_file("../../target.txt", "evil")
        assert result["success"] is False


# ------------------------------------------------------------------
# Tests: search_files
# ------------------------------------------------------------------

class TestSearchFiles:

    def test_search_by_name(self, workspace):
        from tools.file_tools import search_files
        (workspace / "alpha.py").write_text("")
        (workspace / "beta.txt").write_text("")

        result = search_files("alpha")
        assert result["success"] is True
        assert any("alpha.py" in m for m in result["matches"])

    def test_search_by_extension(self, workspace):
        from tools.file_tools import search_files
        (workspace / "a.py").write_text("")
        (workspace / "b.py").write_text("")
        (workspace / "c.txt").write_text("")

        result = search_files("*.py")
        assert result["success"] is True
        assert len(result["matches"]) == 2

    def test_search_no_results(self, workspace):
        from tools.file_tools import search_files
        result = search_files("nonexistent_file_xyz_abc")
        assert result["success"] is True
        assert len(result["matches"]) == 0
