"""
tests/benchmark_tool_schemas.py — ABD V2 Tool-Schema Latency Benchmark
========================================================================
Sends the same simple prompt to Ollama under three tool-schema configurations
and records detailed timing from Ollama's own response fields:

  1. ALL_TOOLS   — all 16 schemas (current production behaviour)
  2. NO_TOOLS    — no schemas (plain conversation)
  3. FILE_TOOLS  — 5 file schemas (representative subset)

Run (requires a running Ollama instance):
    python tests/benchmark_tool_schemas.py

Output: a table of Ollama metrics + total HTTP round-trip time.

NOT a pytest test file — name does not start with "test_" so pytest skips it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: make the project importable from tests/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Minimal env so config.py doesn't abort
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5-coder:7b")
os.environ.setdefault("OLLAMA_STREAM", "false")
os.environ.setdefault("OLLAMA_TIMEOUT", "180")
os.environ.setdefault("GEMINI_API_KEY", "benchmark-placeholder")
os.environ.setdefault("WORKSPACE_DIR", str(_PROJECT_ROOT / "workspace"))
os.environ.setdefault("LOG_FILE", str(_PROJECT_ROOT / "test.log"))
os.environ.setdefault("MAX_TOOL_ITERATIONS", "10")
os.environ.setdefault("PDF_CHUNK_SIZE", "8000")
os.environ.setdefault("VOICE_ENABLED", "false")
os.environ.setdefault("VOICE_DEFAULT_MODE", "chat")

import config  # noqa: E402
import requests  # noqa: E402
from brain.router import TOOL_SCHEMAS  # noqa: E402

# ---------------------------------------------------------------------------
# Tool subset definitions
# ---------------------------------------------------------------------------

_FILE_TOOL_NAMES = {
    "list_files", "search_files", "read_file", "create_file", "edit_file"
}
_CODE_TOOL_NAMES = {
    "read_code_file", "create_code_file", "edit_code_file",
    "get_workspace_code_files",
}
_APP_TOOL_NAMES = {"open_application", "list_supported_apps"}
_WEB_TOOL_NAMES = {"open_url", "web_search"}
_YT_TOOL_NAMES  = {"youtube_search", "open_youtube"}
_PDF_TOOL_NAMES = {"load_pdf", "get_pdf_text", "get_active_pdf_info"}

def _subset(names: set[str]) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]

FILE_TOOLS = _subset(_FILE_TOOL_NAMES)
CODE_TOOLS = _subset(_CODE_TOOL_NAMES)
APP_TOOLS  = _subset(_APP_TOOL_NAMES)

SCENARIOS: list[tuple[str, list[dict]]] = [
    ("ALL_TOOLS (16)",  TOOL_SCHEMAS),
    ("NO_TOOLS   ( 0)", []),
    ("FILE_TOOLS ( 5)", FILE_TOOLS),
    ("CODE_TOOLS ( 4)", CODE_TOOLS),
    ("APP_TOOLS  ( 2)", APP_TOOLS),
]

PROMPT = "What is 2 + 2? Answer briefly."

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are ABD, a helpful Windows desktop assistant. "
        "Answer concisely."
    ),
}

# ---------------------------------------------------------------------------
# Single request with timing
# ---------------------------------------------------------------------------

def _run_once(scenario_name: str, tool_schemas: list[dict]) -> dict[str, Any]:
    """Send PROMPT to Ollama with the given schemas. Return metrics dict."""
    payload: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "messages": [SYSTEM_MSG, {"role": "user", "content": PROMPT}],
        "stream": False,
    }
    if tool_schemas:
        payload["tools"] = tool_schemas

    url = f"{config.OLLAMA_BASE_URL}/api/chat"

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=180)
    except requests.exceptions.ConnectionError:
        return {
            "scenario": scenario_name,
            "error": "Ollama not reachable — is `ollama serve` running?",
        }
    except requests.exceptions.Timeout:
        return {"scenario": scenario_name, "error": "Request timed out (>180s)"}
    t1 = time.perf_counter()

    if resp.status_code != 200:
        return {
            "scenario": scenario_name,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
        }

    try:
        body = resp.json()
    except json.JSONDecodeError as exc:
        return {"scenario": scenario_name, "error": f"Invalid JSON: {exc}"}

    ns = 1_000_000_000  # nanoseconds → seconds divisor

    result = {
        "scenario": scenario_name,
        "num_schemas": len(tool_schemas),
        "wall_clock_s": round(t1 - t0, 3),
        # Ollama's own timing fields (all in nanoseconds)
        "total_duration_s":      round(body.get("total_duration",      0) / ns, 3),
        "load_duration_s":       round(body.get("load_duration",       0) / ns, 3),
        "prompt_eval_count":     body.get("prompt_eval_count",    -1),
        "prompt_eval_duration_s":round(body.get("prompt_eval_duration", 0) / ns, 3),
        "eval_count":            body.get("eval_count",           -1),
        "eval_duration_s":       round(body.get("eval_duration",       0) / ns, 3),
        "response_text":         (body.get("message", {}).get("content") or "")[:120],
        "tool_called":           bool(body.get("message", {}).get("tool_calls")),
    }

    # Derived: tokens/s for generation phase
    if result["eval_count"] > 0 and result["eval_duration_s"] > 0:
        result["gen_tok_per_s"] = round(
            result["eval_count"] / result["eval_duration_s"], 1
        )
    else:
        result["gen_tok_per_s"] = None

    return result


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------

_COL = {
    "scenario":             ("Scenario",            24),
    "num_schemas":          ("#Tools",               7),
    "wall_clock_s":         ("Wall(s)",              8),
    "total_duration_s":     ("Total(s)",             9),
    "load_duration_s":      ("Load(s)",              8),
    "prompt_eval_count":    ("PmptTok",              8),
    "prompt_eval_duration_s":("PmptEval(s)",        12),
    "eval_count":           ("GenTok",               7),
    "eval_duration_s":      ("Gen(s)",               8),
    "gen_tok_per_s":        ("Tok/s",                7),
    "tool_called":          ("ToolCall",             9),
}


def _print_table(rows: list[dict]) -> None:
    headers = list(_COL.keys())
    widths  = [_COL[h][1] for h in headers]
    labels  = [_COL[h][0] for h in headers]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(l.ljust(w) for l, w in zip(labels, widths)) + " |"

    print(sep)
    print(hdr)
    print(sep)

    for row in rows:
        if "error" in row:
            print(f"| ERROR: {row['scenario']}: {row['error']}")
            continue
        cells = []
        for key, width in zip(headers, widths):
            val = row.get(key, "")
            if val is None:
                val = "n/a"
            cells.append(str(val).ljust(width)[:width])
        print("| " + " | ".join(cells) + " |")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 72)
    print("  ABD V2  —  Tool-Schema Latency Benchmark")
    print(f"  Model  : {config.OLLAMA_MODEL}")
    print(f"  Endpoint: {config.OLLAMA_BASE_URL}")
    print(f"  Prompt : {PROMPT!r}")
    print("=" * 72)
    print()

    results: list[dict] = []
    for name, schemas in SCENARIOS:
        print(f"  Running: {name} ...", end=" ", flush=True)
        row = _run_once(name, schemas)
        results.append(row)
        if "error" in row:
            print(f"ERROR — {row['error']}")
        else:
            print(f"done  ({row['wall_clock_s']:.1f}s wall)")

    print()
    _print_table(results)
    print()

    # Highlight the speedup between ALL_TOOLS and NO_TOOLS
    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        all_row  = next((r for r in valid if r["scenario"].startswith("ALL")), None)
        none_row = next((r for r in valid if r["scenario"].startswith("NO_")), None)
        if all_row and none_row:
            speedup = all_row["wall_clock_s"] / none_row["wall_clock_s"] if none_row["wall_clock_s"] else 0
            token_diff = (all_row.get("prompt_eval_count", 0) or 0) - (none_row.get("prompt_eval_count", 0) or 0)
            print(f"  > ALL_TOOLS vs NO_TOOLS wall-clock speedup : {speedup:.2f}x")
            print(f"  > Extra prompt tokens from tool schemas     : {token_diff}")
            print()

    print("  Benchmark complete.")
    print()


if __name__ == "__main__":
    main()
