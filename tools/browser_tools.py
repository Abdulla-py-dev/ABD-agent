"""
tools/browser_tools.py — Web Browser Tools for ABD V1
======================================================
Opens URLs and performs web searches in the user's default browser.
Uses Python's built-in `webbrowser` module — no browser automation.
"""

from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from typing import Any

from safety.permissions import log_action

logger = logging.getLogger("abd.browser_tools")

# Base search URL
_GOOGLE_SEARCH_URL = "https://www.google.com/search?q="


def open_url(url: str) -> dict[str, Any]:
    """Open a URL in the user's default web browser.

    Parameters
    ----------
    url:
        The full URL to open (must start with http:// or https://).

    Returns
    -------
    dict with keys: ``success``, ``url``, ``message``
    """
    url = url.strip()

    # Basic validation: must be an http/https URL
    if not (url.startswith("http://") or url.startswith("https://")):
        # Try prepending https:// if it looks like a domain
        if "." in url and not url.startswith("//"):
            url = "https://" + url
        else:
            return {
                "success": False,
                "url": url,
                "message": f"'{url}' doesn't look like a valid URL. Please include https://",
            }

    try:
        webbrowser.open(url)
        log_action("OPEN_URL", url)
        return {
            "success": True,
            "url": url,
            "message": f"Opened {url} in your default browser.",
        }
    except Exception as exc:
        logger.exception("open_url error")
        return {
            "success": False,
            "url": url,
            "message": f"Failed to open URL: {exc}",
        }


def web_search(query: str) -> dict[str, Any]:
    """Search Google for *query* by opening a search URL in the default browser.

    Parameters
    ----------
    query:
        The search query string.

    Returns
    -------
    dict with keys: ``success``, ``query``, ``url``, ``message``
    """
    query = query.strip()
    if not query:
        return {"success": False, "query": query, "message": "Search query cannot be empty."}

    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"{_GOOGLE_SEARCH_URL}{encoded_query}"

    try:
        webbrowser.open(search_url)
        log_action("WEB_SEARCH", query)
        return {
            "success": True,
            "query": query,
            "url": search_url,
            "message": f"Opened Google search for '{query}' in your default browser.",
        }
    except Exception as exc:
        logger.exception("web_search error")
        return {
            "success": False,
            "query": query,
            "message": f"Failed to open browser for search: {exc}",
        }
