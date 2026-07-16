"""
config.py — ABD V1 Configuration
=================================
Loads all settings from the .env file and exposes them as typed constants.
Import this module anywhere in the project to access configuration.
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
# Gemini API settings
# ------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Default to the stable alias that always resolves to the latest Flash model.
# Override via GEMINI_MODEL in your .env file.
# Known working aliases: gemini-flash-latest, gemini-pro-latest
# Known working versions: gemini-2.0-flash-lite (free tier, may hit quota)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not GEMINI_API_KEY:
    print(
        "[ABD CONFIG ERROR] GEMINI_API_KEY is not set.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Set your API key: GEMINI_API_KEY=your_key_here\n"
        "  Get a free key at: https://aistudio.google.com/app/apikey"
    )
    sys.exit(1)


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

# PDF chunking: max characters per chunk sent to Gemini
PDF_CHUNK_SIZE: int = int(os.getenv("PDF_CHUNK_SIZE", "8000"))

# Project root (useful for other modules)
PROJECT_ROOT: Path = _PROJECT_ROOT
