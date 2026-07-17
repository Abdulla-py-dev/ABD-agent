"""
config.py — ABD V2 Configuration
==================================
Loads all settings from the .env file and exposes them as typed constants.
Import this module anywhere in the project to access configuration.

V2 changes:
  - Added LLM_PROVIDER ("ollama" | "gemini") — default "ollama"
  - Added OLLAMA_BASE_URL and OLLAMA_MODEL
  - Added OLLAMA_STREAM flag
  - GEMINI_API_KEY is now optional when LLM_PROVIDER="ollama"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env file from the project root (same directory as config.py)
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


# ------------------------------------------------------------------
# LLM Provider selection  (V2)
# ------------------------------------------------------------------
# "ollama"  — local Ollama server (no API key needed)
# "gemini"  — Google Gemini API (requires GEMINI_API_KEY)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

if LLM_PROVIDER not in ("ollama", "gemini"):
    print(
        f"[ABD CONFIG ERROR] Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
        "Valid values are: 'ollama', 'gemini'."
    )
    sys.exit(1)


# ------------------------------------------------------------------
# Ollama settings  (V2)
# ------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

# Stream Ollama responses token-by-token for a faster terminal feel.
# Set OLLAMA_STREAM=false in .env to disable (useful for automated tests).
OLLAMA_STREAM: bool = os.getenv("OLLAMA_STREAM", "true").lower() not in ("false", "0", "no")

# Timeout in seconds for a single Ollama /api/chat request
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))


# ------------------------------------------------------------------
# Gemini API settings  (kept for optional fallback)
# ------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Default to the stable alias that always resolves to the latest Flash model.
# Override via GEMINI_MODEL in your .env file.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Gemini key is required only when it is the selected provider
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    print(
        "[ABD CONFIG ERROR] LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Set your API key: GEMINI_API_KEY=your_key_here\n"
        "  Get a free key at: https://aistudio.google.com/app/apikey\n"
        "  Or switch to local mode: LLM_PROVIDER=ollama"
    )
    sys.exit(1)

if LLM_PROVIDER == "ollama" and not GEMINI_API_KEY:
    # Not an error — just informational (logged, not printed, to keep startup clean)
    pass


# ------------------------------------------------------------------
# Workspace directory — ABD can only create/edit files here
# ------------------------------------------------------------------
_workspace_env = os.getenv("WORKSPACE_DIR", "workspace")
_workspace_path = Path(_workspace_env)

# Support both relative (to project root) and absolute paths
if not _workspace_path.is_absolute():
    WORKSPACE_DIR: Path = (_PROJECT_ROOT / _workspace_path).resolve()
else:
    WORKSPACE_DIR = _workspace_path.resolve()

# Create workspace if it doesn't exist
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
_log_env = os.getenv("LOG_FILE", "abd_actions.log")
_log_path = Path(_log_env)

if not _log_path.is_absolute():
    LOG_FILE: Path = (_PROJECT_ROOT / _log_path).resolve()
else:
    LOG_FILE = _log_path.resolve()


# ------------------------------------------------------------------
# Agent settings
# ------------------------------------------------------------------
MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))

# PDF chunking: max characters per chunk sent to LLM
PDF_CHUNK_SIZE: int = int(os.getenv("PDF_CHUNK_SIZE", "8000"))

# Project root (useful for other modules)
PROJECT_ROOT: Path = _PROJECT_ROOT
