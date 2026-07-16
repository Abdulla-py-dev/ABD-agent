"""
tools/pdf_tools.py — PDF Reading Tools for ABD V1
==================================================
Extracts text from PDF files using pypdf.
Supports session-level "active PDF" memory so follow-up questions work.
Large PDFs are automatically split into chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

from safety.permissions import validate_workspace_path, PathTraversalError, log_action
import config

logger = logging.getLogger("abd.pdf_tools")

# ------------------------------------------------------------------
# Session state — the currently loaded PDF
# ------------------------------------------------------------------
_active_pdf: dict[str, Any] = {
    "path": None,       # Path object
    "filename": None,   # str
    "text": None,       # full extracted text (str)
    "pages": 0,         # number of pages
}


def get_active_pdf() -> dict[str, Any]:
    """Return a copy of the active PDF session state."""
    return dict(_active_pdf)


def clear_active_pdf() -> None:
    """Clear the active PDF from session memory."""
    global _active_pdf
    _active_pdf = {"path": None, "filename": None, "text": None, "pages": 0}


# ------------------------------------------------------------------
# Internal helper — extract text from a PDF file
# ------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Return (full_text, page_count) from a PDF file.

    Raises RuntimeError if pypdf is not installed or the file can't be read.
    """
    if not _PYPDF_AVAILABLE:
        raise RuntimeError(
            "pypdf is not installed. Run: pip install pypdf"
        )

    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    texts = []
    for i, page in enumerate(pages):
        try:
            page_text = page.extract_text() or ""
            texts.append(f"[Page {i + 1}]\n{page_text.strip()}")
        except Exception as exc:
            texts.append(f"[Page {i + 1}] (could not extract text: {exc})")

    full_text = "\n\n".join(texts)
    return full_text, len(pages)


# ------------------------------------------------------------------
# Tool: load_pdf
# ------------------------------------------------------------------

def load_pdf(path: str) -> dict[str, Any]:
    """Load a PDF file into session memory and return an overview.

    This makes the PDF available for follow-up questions via
    ``query_active_pdf``.

    Parameters
    ----------
    path:
        Path to the PDF file (relative to workspace or absolute path that
        exists on the filesystem — PDFs may be outside the workspace for
        reading purposes, but writing is always restricted).

    Returns
    -------
    dict with keys: ``success``, ``filename``, ``pages``, ``preview``, ``message``
    """
    global _active_pdf

    pdf_path = Path(path)

    # Try workspace-relative first, then absolute
    if not pdf_path.is_absolute():
        # Attempt workspace path
        try:
            resolved = validate_workspace_path(path)
        except PathTraversalError:
            resolved = pdf_path.resolve()
    else:
        resolved = pdf_path.resolve()

    if not resolved.exists():
        return {"success": False, "message": f"PDF file not found: {path}"}
    if not resolved.is_file():
        return {"success": False, "message": f"'{path}' is not a file."}
    if resolved.suffix.lower() != ".pdf":
        return {"success": False, "message": f"'{resolved.name}' does not appear to be a PDF."}

    try:
        full_text, page_count = _extract_pdf_text(resolved)
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        logger.exception("PDF extraction error")
        return {"success": False, "message": f"Failed to read PDF: {exc}"}

    # Store in session
    _active_pdf = {
        "path": resolved,
        "filename": resolved.name,
        "text": full_text,
        "pages": page_count,
    }

    # Provide a brief preview (first 500 chars of content)
    preview_text = full_text[:500].strip() if full_text else "(No text extracted)"

    log_action("LOAD_PDF", str(resolved))
    return {
        "success": True,
        "filename": resolved.name,
        "pages": page_count,
        "total_chars": len(full_text),
        "preview": preview_text,
        "message": (
            f"PDF '{resolved.name}' loaded successfully. "
            f"{page_count} page(s), {len(full_text):,} characters extracted. "
            "You can now ask me to summarise it or answer questions about it."
        ),
    }


# ------------------------------------------------------------------
# Tool: get_pdf_text
# ------------------------------------------------------------------

def get_pdf_text(chunk_index: int = 0) -> dict[str, Any]:
    """Return a chunk of text from the currently loaded PDF.

    The text is split into chunks of ``config.PDF_CHUNK_SIZE`` characters.
    Use ``chunk_index`` to page through a large document.

    Parameters
    ----------
    chunk_index:
        Zero-based index of the chunk to return (default 0 = first chunk).

    Returns
    -------
    dict with keys: ``success``, ``filename``, ``chunk_index``,
                    ``total_chunks``, ``text``, ``message``
    """
    if not _active_pdf["text"]:
        return {
            "success": False,
            "message": "No PDF is currently loaded. Use load_pdf first.",
        }

    full_text: str = _active_pdf["text"]
    chunk_size = config.PDF_CHUNK_SIZE
    total_chunks = max(1, -(-len(full_text) // chunk_size))  # ceiling division

    if chunk_index < 0 or chunk_index >= total_chunks:
        return {
            "success": False,
            "message": (
                f"Invalid chunk_index {chunk_index}. "
                f"Valid range: 0 – {total_chunks - 1}."
            ),
        }

    start = chunk_index * chunk_size
    end = start + chunk_size
    chunk_text = full_text[start:end]

    return {
        "success": True,
        "filename": _active_pdf["filename"],
        "pages": _active_pdf["pages"],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "text": chunk_text,
        "message": (
            f"Returning chunk {chunk_index + 1} of {total_chunks} "
            f"from '{_active_pdf['filename']}'."
        ),
    }


# ------------------------------------------------------------------
# Tool: get_active_pdf_info
# ------------------------------------------------------------------

def get_active_pdf_info() -> dict[str, Any]:
    """Return metadata about the currently loaded PDF without the full text."""
    if not _active_pdf["path"]:
        return {
            "success": False,
            "message": "No PDF is currently loaded.",
        }
    return {
        "success": True,
        "filename": _active_pdf["filename"],
        "path": str(_active_pdf["path"]),
        "pages": _active_pdf["pages"],
        "total_chars": len(_active_pdf["text"]) if _active_pdf["text"] else 0,
        "message": f"Active PDF: '{_active_pdf['filename']}' ({_active_pdf['pages']} pages).",
    }
