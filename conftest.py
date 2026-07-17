"""
conftest.py — Root-level pytest configuration for ABD V2
=========================================================
Sets fake environment variables at process start so that config.py
can import without a real .env file, Gemini API key, or running Ollama.

V2: added Ollama env defaults and LLM_PROVIDER.
"""

import os
import sys
from pathlib import Path

# Set fake env vars BEFORE any module-level imports happen.
# This runs when pytest first loads this conftest.
_FAKE_WORKSPACE = str(Path(__file__).parent / "workspace")

# --- Core ---
os.environ.setdefault("WORKSPACE_DIR", _FAKE_WORKSPACE)
os.environ.setdefault("LOG_FILE", str(Path(__file__).parent / "test_actions.log"))
os.environ.setdefault("MAX_TOOL_ITERATIONS", "10")
os.environ.setdefault("PDF_CHUNK_SIZE", "8000")

# --- LLM Provider (tests default to Ollama so no API key is required) ---
os.environ.setdefault("LLM_PROVIDER", "ollama")

# --- Ollama (tests use a fake URL; requests are mocked in test files) ---
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5-coder:7b")
os.environ.setdefault("OLLAMA_STREAM", "false")   # disable streaming in tests
os.environ.setdefault("OLLAMA_TIMEOUT", "10")

# --- Gemini (kept for tests that need to import gemini_client) ---
os.environ.setdefault("GEMINI_API_KEY", "fake-test-key-no-api-call-made")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash")
