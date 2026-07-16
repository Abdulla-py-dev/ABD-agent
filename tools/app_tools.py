"""
tools/app_tools.py — Application Launcher for ABD V1
=====================================================
Opens whitelisted Windows applications safely using subprocess.
No arbitrary shell commands are allowed through this interface.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

from safety.permissions import log_action

logger = logging.getLogger("abd.app_tools")


# ------------------------------------------------------------------
# Application whitelist
# Each entry maps a set of aliases → launch strategy.
#
# Strategy types:
#   "exe"      — launch a specific executable directly
#   "start"    — use `start ""` shell command (for UWP/Store apps)
#   "uri"      — launch a URI handler (e.g. whatsapp://)
# ------------------------------------------------------------------

_APP_REGISTRY: dict[str, dict[str, Any]] = {
    # ---- System utilities -------------------------------------------
    "calculator": {
        "aliases": {"calculator", "calc"},
        "strategy": "exe",
        "command": "calc.exe",
        "display_name": "Calculator",
    },
    "notepad": {
        "aliases": {"notepad", "text editor", "note pad"},
        "strategy": "exe",
        "command": "notepad.exe",
        "display_name": "Notepad",
    },
    "paint": {
        "aliases": {"paint", "mspaint"},
        "strategy": "exe",
        "command": "mspaint.exe",
        "display_name": "Paint",
    },
    "file_explorer": {
        "aliases": {"file explorer", "explorer", "files", "file manager"},
        "strategy": "exe",
        "command": "explorer.exe",
        "display_name": "File Explorer",
    },
    "task_manager": {
        "aliases": {"task manager", "taskmgr"},
        "strategy": "exe",
        "command": "taskmgr.exe",
        "display_name": "Task Manager",
    },
    "cmd": {
        "aliases": {"command prompt", "cmd", "terminal"},
        "strategy": "exe",
        "command": "cmd.exe",
        "display_name": "Command Prompt",
    },
    "powershell": {
        "aliases": {"powershell", "power shell"},
        "strategy": "exe",
        "command": "powershell.exe",
        "display_name": "PowerShell",
    },
    # ---- Developer tools --------------------------------------------
    "vscode": {
        "aliases": {"vs code", "vscode", "visual studio code", "code editor"},
        "strategy": "exe",
        "command": "code",  # added to PATH by VS Code installer
        "display_name": "Visual Studio Code",
    },
    "git_bash": {
        "aliases": {"git bash", "gitbash"},
        "strategy": "path",
        "paths": [
            r"C:\Program Files\Git\git-bash.exe",
            r"C:\Program Files (x86)\Git\git-bash.exe",
        ],
        "display_name": "Git Bash",
    },
    # ---- Communication & productivity --------------------------------
    "whatsapp": {
        "aliases": {"whatsapp", "whats app"},
        "strategy": "uri",
        "command": "whatsapp://",
        "fallback_start": "WhatsApp",
        "display_name": "WhatsApp",
    },
    "teams": {
        "aliases": {"teams", "microsoft teams"},
        "strategy": "start",
        "command": "msteams",
        "display_name": "Microsoft Teams",
    },
    "outlook": {
        "aliases": {"outlook", "email", "mail"},
        "strategy": "exe",
        "command": "outlook",
        "display_name": "Microsoft Outlook",
    },
    "excel": {
        "aliases": {"excel", "spreadsheet"},
        "strategy": "exe",
        "command": "excel",
        "display_name": "Microsoft Excel",
    },
    "word": {
        "aliases": {"word", "microsoft word", "document editor"},
        "strategy": "exe",
        "command": "winword",
        "display_name": "Microsoft Word",
    },
    "powerpoint": {
        "aliases": {"powerpoint", "ppt", "presentation"},
        "strategy": "exe",
        "command": "powerpnt",
        "display_name": "Microsoft PowerPoint",
    },
    # ---- Media & browsers -------------------------------------------
    "chrome": {
        "aliases": {"chrome", "google chrome"},
        "strategy": "exe",
        "command": "chrome",
        "display_name": "Google Chrome",
    },
    "firefox": {
        "aliases": {"firefox", "mozilla firefox"},
        "strategy": "exe",
        "command": "firefox",
        "display_name": "Mozilla Firefox",
    },
    "edge": {
        "aliases": {"edge", "microsoft edge"},
        "strategy": "exe",
        "command": "msedge",
        "display_name": "Microsoft Edge",
    },
    "vlc": {
        "aliases": {"vlc", "media player", "video player"},
        "strategy": "exe",
        "command": "vlc",
        "display_name": "VLC Media Player",
    },
    "spotify": {
        "aliases": {"spotify", "music"},
        "strategy": "start",
        "command": "Spotify",
        "display_name": "Spotify",
    },
    # ---- Settings / control panel -----------------------------------
    "settings": {
        "aliases": {"settings", "windows settings", "system settings"},
        "strategy": "uri",
        "command": "ms-settings:",
        "display_name": "Windows Settings",
    },
    "control_panel": {
        "aliases": {"control panel", "control"},
        "strategy": "exe",
        "command": "control",
        "display_name": "Control Panel",
    },
}


def _resolve_app_name(app_name: str) -> dict[str, Any] | None:
    """Return the registry entry matching *app_name*, or None if not found."""
    query = app_name.strip().lower()
    for entry in _APP_REGISTRY.values():
        if query in entry["aliases"]:
            return entry
    return None


def _launch_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Attempt to launch the application described by *entry*.

    Returns a result dict with success/message.
    """
    strategy = entry.get("strategy", "exe")
    display_name = entry["display_name"]

    try:
        if strategy == "exe":
            cmd = entry["command"]
            # Check if the command is on PATH first
            if not shutil.which(cmd):
                return {
                    "success": False,
                    "message": (
                        f"{display_name} could not be found. "
                        "It may not be installed or not on the system PATH."
                    ),
                }
            subprocess.Popen([cmd], shell=False)

        elif strategy == "path":
            # Try a list of known absolute paths
            for candidate in entry.get("paths", []):
                if os.path.isfile(candidate):
                    subprocess.Popen([candidate], shell=False)
                    log_action("OPEN_APP", f"{display_name} via {candidate}")
                    return {
                        "success": True,
                        "message": f"{display_name} launched successfully.",
                    }
            return {
                "success": False,
                "message": f"{display_name} could not be found at any known location.",
            }

        elif strategy == "start":
            # Use `start ""` for UWP/Store apps or apps registered in PATH
            subprocess.Popen(
                ["cmd", "/c", "start", "", entry["command"]],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

        elif strategy == "uri":
            # Launch a URI scheme (e.g. whatsapp://, ms-settings:)
            import webbrowser
            webbrowser.open(entry["command"])

            # If a fallback start command is provided, we also try that
            # (WhatsApp via URI is less reliable than via start command)
            fallback = entry.get("fallback_start")
            if fallback:
                try:
                    subprocess.Popen(
                        ["cmd", "/c", "start", "", fallback],
                        shell=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                except Exception:
                    pass  # URI already tried; ignore fallback failures silently

        else:
            return {"success": False, "message": f"Unknown launch strategy: {strategy}"}

        log_action("OPEN_APP", display_name)
        return {"success": True, "message": f"{display_name} launched successfully."}

    except FileNotFoundError:
        return {
            "success": False,
            "message": (
                f"{display_name} could not be opened. "
                "The executable was not found on this system."
            ),
        }
    except PermissionError:
        return {
            "success": False,
            "message": f"Permission denied when trying to launch {display_name}.",
        }
    except Exception as exc:
        logger.exception("launch error for %s", display_name)
        return {"success": False, "message": f"Failed to launch {display_name}: {exc}"}


# ------------------------------------------------------------------
# Tool: open_application
# ------------------------------------------------------------------

def open_application(app_name: str) -> dict[str, Any]:
    """Open a Windows application by name.

    Parameters
    ----------
    app_name:
        The name of the application to open (e.g. "Calculator", "VS Code",
        "WhatsApp").

    Returns
    -------
    dict with keys: ``success``, ``app_name``, ``message``
    """
    entry = _resolve_app_name(app_name)

    if entry is None:
        # Build a helpful list of what IS supported
        supported = sorted({e["display_name"] for e in _APP_REGISTRY.values()})
        return {
            "success": False,
            "app_name": app_name,
            "message": (
                f"'{app_name}' is not in ABD's application whitelist. "
                f"Supported apps: {', '.join(supported)}."
            ),
        }

    result = _launch_entry(entry)
    result["app_name"] = app_name
    return result


# ------------------------------------------------------------------
# Tool: list_supported_apps
# ------------------------------------------------------------------

def list_supported_apps() -> dict[str, Any]:
    """Return the list of applications ABD can open."""
    apps = sorted({entry["display_name"] for entry in _APP_REGISTRY.values()})
    return {
        "success": True,
        "apps": apps,
        "message": f"ABD can open {len(apps)} applications: {', '.join(apps)}.",
    }
