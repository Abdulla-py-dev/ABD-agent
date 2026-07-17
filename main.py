"""
main.py — ABD V2 Entry Point
==============================
Runs the interactive terminal chat interface.

Start ABD with:
    python main.py

Options:
    --debug     Enable verbose debug logging to the console
    --reset     Start with a fresh conversation (no prior context)
    --version   Print ABD version and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

# Ensure config loads first (validates API key, creates workspace, etc.)
import config  # noqa: F401 — side-effects: API key validation, workspace creation

from brain.agent import ABDAgent

# --- Rich for beautiful terminal output ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.theme import Theme
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ------------------------------------------------------------------
# ABD version
# ------------------------------------------------------------------
ABD_VERSION = "2.0.0"


# ------------------------------------------------------------------
# Terminal UI helpers
# ------------------------------------------------------------------

if _RICH_AVAILABLE:
    _theme = Theme({
        "abd": "bold cyan",
        "user": "bold green",
        "error": "bold red",
        "dim": "dim white",
        "warning": "bold yellow",
    })
    console = Console(theme=_theme)

    def _print_banner() -> None:
        banner = Text()
        banner.append("  █████╗ ██████╗ ██████╗ \n", style="bold cyan")
        banner.append(" ██╔══██╗██╔══██╗██╔══██╗\n", style="bold cyan")
        banner.append(" ███████║██████╔╝██║  ██║\n", style="bold cyan")
        banner.append(" ██╔══██║██╔══██╗██║  ██║\n", style="bold cyan")
        banner.append(" ██║  ██║██████╔╝██████╔╝\n", style="bold cyan")
        banner.append(" ╚═╝  ╚═╝╚═════╝ ╚═════╝ \n", style="bold cyan")
        subtitle = Text(
            f"  Advanced Brain Desktop Assistant  ·  v{ABD_VERSION}  ·  "
            f"{'Ollama / ' + config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else 'Gemini / ' + config.GEMINI_MODEL}",
            style="dim white"
        )
        console.print(banner)
        console.print(subtitle)
        console.print()
        console.print(
            Panel(
                "[dim]Type your command below. Type [bold]'exit'[/bold] or [bold]'quit'[/bold] "
                "to close ABD.\nType [bold]'reset'[/bold] to start a new session. "
                "Type [bold]'help'[/bold] for example commands.[/dim]",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()

    def _print_user(text: str) -> None:
        console.print(f"[user]You:[/user] {text}")

    def _print_abd(text: str) -> None:
        console.print()
        console.print("[abd]ABD:[/abd]", end=" ")
        # Try rendering as Markdown for code blocks, lists, etc.
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)
        console.print()

    def _print_error(text: str) -> None:
        console.print(f"\n[error]ERROR:[/error] {text}\n")

    def _print_info(text: str) -> None:
        console.print(f"[dim]{text}[/dim]")

    def _print_rule(text: str = "") -> None:
        console.print(Rule(text, style="dim cyan"))

    def _get_input() -> str:
        try:
            return console.input("[user]You:[/user] ").strip()
        except EOFError:
            return "exit"

else:
    # Fallback: plain print/input if Rich is not installed
    def _print_banner() -> None:
        print("=" * 60)
        print(f"  ABD — Advanced Brain Desktop Assistant v{ABD_VERSION}")
        if config.LLM_PROVIDER == 'ollama':
            print(f"  Powered by Ollama / {config.OLLAMA_MODEL}")
        else:
            print(f"  Powered by Google Gemini / {config.GEMINI_MODEL}")
        print("=" * 60)
        print("  Type 'exit' to quit | 'reset' to start over")
        print()

    def _print_user(text: str) -> None:
        pass  # already echoed by terminal

    def _print_abd(text: str) -> None:
        print(f"\nABD: {text}\n")

    def _print_error(text: str) -> None:
        print(f"\n[ERROR] {text}\n")

    def _print_info(text: str) -> None:
        print(f"  {text}")

    def _print_rule(text: str = "") -> None:
        print(f"--- {text} ---" if text else "-" * 40)

    def _get_input() -> str:
        try:
            return input("You: ").strip()
        except EOFError:
            return "exit"


# ------------------------------------------------------------------
# Built-in commands (handled before sending to AI)
# ------------------------------------------------------------------

_HELP_TEXT = """
**Example commands you can try:**

**General**
- Hello ABD
- What can you do?

**Files**
- List my workspace files
- Read hello.py
- Create a file called notes.txt with "Meeting at 3pm"
- Search for Python files

**Coding**
- Create hello.py that prints "Hello ABD"
- Read main.py and explain what it does
- List all code files in workspace

**Apps**
- Open Calculator
- Open Notepad
- Open WhatsApp
- Open VS Code

**Web & YouTube**
- Search the web for Docker networking
- Search YouTube for Python FastAPI tutorials
- Open https://github.com

**PDFs**
- Read and summarise this PDF: C:/path/to/paper.pdf
- What is the methodology in the loaded PDF?

Type 'reset' to clear conversation history.
Type 'exit' or 'quit' to close ABD.
"""


def _handle_builtin(user_input: str, agent: ABDAgent) -> bool:
    """Handle built-in commands that don't need to go to Gemini.

    Returns True if the command was handled (skip sending to AI).
    """
    cmd = user_input.lower().strip()

    if cmd in ("exit", "quit", "bye", "goodbye"):
        _print_rule()
        _print_info("ABD shutting down. Goodbye!")
        sys.exit(0)

    if cmd == "reset":
        agent.reset()
        _print_info("Session reset. Starting fresh conversation.")
        return True

    if cmd in ("help", "?", "commands"):
        if _RICH_AVAILABLE:
            from rich.markdown import Markdown
            console.print(Markdown(_HELP_TEXT))
        else:
            print(_HELP_TEXT)
        return True

    if cmd in ("version", "--version"):
        if config.LLM_PROVIDER == "ollama":
            provider_info = f"Ollama / {config.OLLAMA_MODEL}"
        else:
            provider_info = f"Gemini / {config.GEMINI_MODEL}"
        _print_info(f"ABD v{ABD_VERSION} | Provider: {provider_info}")
        return True

    if cmd == "":
        return True  # ignore empty input silently

    return False


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Always log important ABD actions to file (via the action logger in safety/)
    # The action logger is configured independently in safety/permissions.py


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="abd",
        description="ABD — Advanced Brain Desktop Assistant",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args()

    if args.version:
        print(f"ABD v{ABD_VERSION}")
        sys.exit(0)

    _setup_logging(args.debug)

    # Print startup banner
    _print_banner()

    # Initialise the agent
    try:
        if config.LLM_PROVIDER == "ollama":
            _print_info(f"Initialising ABD with Ollama model: {config.OLLAMA_MODEL} …")
        else:
            _print_info(f"Initialising ABD with Gemini model: {config.GEMINI_MODEL} …")
        agent = ABDAgent()
        _print_info(f"Workspace: {config.WORKSPACE_DIR}")
        _print_info(f"Ready! Session started at {datetime.now().strftime('%H:%M:%S')}")
        _print_rule()
        print()
    except SystemExit:
        raise
    except Exception as exc:
        _print_error(f"Failed to initialise ABD: {exc}")
        sys.exit(1)

    # Main chat loop
    while True:
        try:
            user_input = _get_input()
        except KeyboardInterrupt:
            print()
            _print_info("\nInterrupted. Type 'exit' to quit.")
            continue

        # Handle built-in commands
        if _handle_builtin(user_input, agent):
            continue

        # Send to AI and display response
        try:
            # Determine if we should stream tokens directly to the terminal.
            # Streaming mode: Ollama + OLLAMA_STREAM=true.
            # In this path we print the ABD: prefix once, then feed each token
            # inline via a callback — the full response is NOT printed again.
            _streaming = (
                config.LLM_PROVIDER == "ollama"
                and config.OLLAMA_STREAM
            )

            if _streaming:
                # ── Streaming display path ─────────────────────────────────
                # Shared mutable state for the callback closure:
                #   first_token — True until the first token arrives
                #   live        — reference to the Live spinner instance
                _state: dict = {"first_token": True, "live": None}

                def _write_token(tok: str) -> None:
                    import sys as _sys
                    if _state["first_token"]:
                        _state["first_token"] = False
                        # Stop and clear the spinner on the very first token
                        if _state["live"] is not None:
                            _state["live"].stop()
                        # Print ABD: prefix exactly once
                        if _RICH_AVAILABLE:
                            console.print()
                            console.print("[abd]ABD:[/abd] ", end="")
                        else:
                            _sys.stdout.write("\nABD: ")
                            _sys.stdout.flush()
                    # Write token inline — sys.stdout avoids Rich buffering
                    _sys.stdout.write(tok)
                    _sys.stdout.flush()

                agent.set_stream_callback(_write_token)

                if _RICH_AVAILABLE:
                    from rich.live import Live as _Live
                    from rich.spinner import Spinner as _Spinner
                    _spin = _Spinner(
                        "dots",
                        text="[dim] ABD is thinking\u2026[/dim]",
                        style="dim cyan",
                    )
                    # transient=True: spinner area is erased when live.stop()
                    # is called, leaving a clean line for the ABD: prefix.
                    with _Live(
                        _spin,
                        console=console,
                        refresh_per_second=12,
                        transient=True,
                    ) as _live:
                        _state["live"] = _live
                        response = agent.chat(user_input)
                else:
                    # Non-Rich fallback: plain thinking message
                    print("ABD is thinking\u2026")
                    response = agent.chat(user_input)

                agent.set_stream_callback(None)

                if _state["first_token"]:
                    # No tokens were streamed (empty response or error path)
                    _print_abd(response)
                else:
                    print()   # end the streamed line
                    print()   # blank line between turns

            else:
                # ── Non-streaming display path (unchanged V1 behaviour) ────
                if _RICH_AVAILABLE:
                    with console.status("[dim]ABD is thinking…[/dim]", spinner="dots"):
                        response = agent.chat(user_input)
                else:
                    print("ABD is thinking…")
                    response = agent.chat(user_input)

                _print_abd(response)

        except KeyboardInterrupt:
            print()
            _print_info("Interrupted. Type 'exit' to quit or ask something else.")
        except Exception as exc:
            _print_error(f"An unexpected error occurred: {exc}")
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
