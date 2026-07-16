"""
conftest.py — Root-level pytest configuration for ABD V1
=========================================================
Sets fake environment variables at process start so that config.py
can import without a real .env file or Gemini API key.
"""

import os
import sys
from pathlib import Path

# Set fake env vars BEFORE any module-level imports happen.
# This runs when pytest first loads this conftest.
_FAKE_WORKSPACE = str(Path(__file__).parent / "workspace")

os.environ.setdefault("GEMINI_API_KEY", "fake-test-key-no-api-call-made")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash")
os.environ.setdefault("WORKSPACE_DIR", _FAKE_WORKSPACE)
os.environ.setdefault("LOG_FILE", str(Path(__file__).parent / "test_actions.log"))
os.environ.setdefault("MAX_TOOL_ITERATIONS", "10")
os.environ.setdefault("PDF_CHUNK_SIZE", "8000")
