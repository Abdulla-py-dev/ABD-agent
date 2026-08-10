"""
tests/conftest.py — Pytest configuration for ABD V1 tests
==========================================================
Sets up fake environment variables BEFORE config.py is imported,
so tests can run without a real .env file or Gemini API key.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ------------------------------------------------------------------
# Patch environment variables before any ABD module is imported.
# This session-scoped autouse fixture ensures os.environ is set
# before config.py's module-level code runs.
# ------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def fake_env(tmp_path_factory):
    """Inject a fake GEMINI_API_KEY into os.environ for the whole test session.
    This prevents config.py from calling sys.exit(1).
    """
    fake_workspace = tmp_path_factory.mktemp("workspace_session")
    env_overrides = {
        "GEMINI_API_KEY": "fake-test-api-key-do-not-use",
        "GEMINI_MODEL": "gemini-2.0-flash",
        "WORKSPACE_DIR": str(fake_workspace),
        "LOG_FILE": str(fake_workspace / "test_actions.log"),
        "MAX_TOOL_ITERATIONS": "10",
        "PDF_CHUNK_SIZE": "8000",
        "OLLAMA_KEEP_ALIVE": "10m",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        # Now import config so it picks up the fake env vars
        if "config" in sys.modules:
            # Reload to pick up new env vars
            import importlib
            import config
            importlib.reload(config)
            config.WORKSPACE_DIR = fake_workspace
            config.GEMINI_API_KEY = "fake-test-api-key-do-not-use"
            config.LOG_FILE = fake_workspace / "test_actions.log"
        yield fake_workspace
