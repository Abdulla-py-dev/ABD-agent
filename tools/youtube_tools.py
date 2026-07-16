"""
tools/youtube_tools.py — YouTube Tools for ABD V1
==================================================
Opens YouTube searches in the user's default browser.
"""

from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from typing import Any

from safety.permissions import log_action

logger = logging.getLogger("abd.youtube_tools")

_YOUTUBE_SEARCH_URL = "https://www.youtube.com/results?search_query="
_YOUTUBE_BASE_URL = "https://www.youtube.com"


def youtube_search(query: str) -> dict[str, Any]:
    """Search YouTube for *query* by opening a search URL in the default browser.

    Parameters
    ----------
    query:
        The search query string (e.g. "Python FastAPI tutorials").

    Returns
    -------
    dict with keys: ``success``, ``query``, ``url``, ``message``
    """
    query = query.strip()
    if not query:
        return {"success": False, "query": query, "message": "Search query cannot be empty."}

    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"{_YOUTUBE_SEARCH_URL}{encoded_query}"

    try:
        webbrowser.open(search_url)
        log_action("YOUTUBE_SEARCH", query)
        return {
            "success": True,
            "query": query,
            "url": search_url,
            "message": f"Opened YouTube search for '{query}' in your default browser.",
        }
    except Exception as exc:
        logger.exception("youtube_search error")
        return {
            "success": False,
            "query": query,
            "message": f"Failed to open YouTube search: {exc}",
        }


def open_youtube() -> dict[str, Any]:
    """Open the YouTube homepage in the default browser.

    Returns
    -------
    dict with keys: ``success``, ``url``, ``message``
    """
    try:
        webbrowser.open(_YOUTUBE_BASE_URL)
        log_action("OPEN_YOUTUBE", _YOUTUBE_BASE_URL)
        return {
            "success": True,
            "url": _YOUTUBE_BASE_URL,
            "message": "Opened YouTube in your default browser.",
        }
    except Exception as exc:
        logger.exception("open_youtube error")
        return {"success": False, "message": f"Failed to open YouTube: {exc}"}
