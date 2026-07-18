"""
voice/tts.py — Text-to-Speech for ABD V2 Phase 2
===================================================
Wraps pyttsx3 to speak ABD responses using a system voice.

Design principles:
  - Uses Windows SAPI5 (built-in, offline, no API key needed).
  - Prefers a male voice when available; falls back gracefully.
  - Never raises into the caller — TTS failures are logged and silently ignored
    so ABD continues to function (text is still printed to the terminal).
  - The class is designed so the engine can be swapped (ElevenLabs, Azure TTS,
    Coqui TTS, etc.) later without changing the VoiceManager interface.

pyttsx3 / COM threading note
-------------------------------
Windows SAPI5 is a COM (Component Object Model) API.  COM requires that a
thread that intends to call COM objects must first initialise a COM apartment
via CoInitialize/CoInitializeEx.  pyttsx3 wraps this through the ``comtypes``
library.

When ``pyttsx3.init()`` is called once at class construction time and the same
engine object is reused across speak() calls, SAPI5 behaves correctly as long
as all calls happen on the *same thread* that called init().

However, ``rich.console.status()`` (used in main.py for the "thinking…" spinner)
creates an internal background thread.  If the console.status context manager's
lifecycle overlaps with or changes the COM apartment state, the engine's SAPI
voice proxy can become invalid — causing ``runAndWait()`` to silently do nothing
or raise obscure COM errors.

**Fix**: create a fresh ``pyttsx3.init()`` engine for every ``speak()`` call.
This ensures COM is always initialised on the correct calling thread.  The
overhead of init() (~10-30 ms) is negligible relative to speech synthesis time.
The engine is explicitly terminated after each utterance to release COM resources.

Usage (inside voice_manager.py):
    tts = TextToSpeech()
    tts.speak("Hello! I am ABD, your assistant.")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config

logger = logging.getLogger("abd.voice.tts")

# Lazy import — pyttsx3 is optional.
_pyttsx3_available: bool = False
try:
    import pyttsx3  # type: ignore[import-untyped]
    _pyttsx3_available = True
except ImportError:
    logger.warning(
        "pyttsx3 library not found. "
        "Install it with: pip install pyttsx3"
    )


class TextToSpeech:
    """Speaks text aloud using the system's TTS engine (Windows SAPI5).

    All public methods return silently on failure — a TTS error never crashes
    ABD.  The response is still visible in the terminal even if voice is broken.

    Parameters
    ----------
    rate:
        Words per minute (default: config.VOICE_TTS_RATE).
    volume:
        Float 0.0–1.0 (default: config.VOICE_TTS_VOLUME).
    voice_preference:
        ``"male"``, ``"female"``, or ``"any"`` (default: config.VOICE_TTS_VOICE_PREFERENCE).
    """

    def __init__(
        self,
        rate: int | None = None,
        volume: float | None = None,
        voice_preference: str | None = None,
    ) -> None:
        self._rate = rate if rate is not None else config.VOICE_TTS_RATE
        self._volume = volume if volume is not None else config.VOICE_TTS_VOLUME
        self._voice_preference = (
            voice_preference if voice_preference is not None
            else config.VOICE_TTS_VOICE_PREFERENCE
        )

        # _voice_id is resolved once at init time (probe the available voices)
        # and reused on every speak() call to avoid re-selecting the voice.
        self._voice_id: Optional[str] = None
        self._ready: bool = False

        if not _pyttsx3_available:
            logger.warning("TTS disabled: pyttsx3 not installed.")
            return

        # Probe to verify pyttsx3 can initialise at all, and resolve the voice id.
        try:
            probe = pyttsx3.init()
            self._voice_id = self._resolve_voice_id(probe, self._voice_preference)
            probe.stop()
            # Note: do NOT call probe.runAndWait() here — that would speak nothing
            # but still block.  Just stop() releases the COM resources cleanly.
            self._ready = True
            logger.info(
                "TTS engine ready | rate=%d wpm | volume=%.1f | pref=%s | voice_id=%s",
                self._rate, self._volume, self._voice_preference, self._voice_id,
            )
        except Exception as exc:
            logger.error("TTS engine probe failed: %s", exc, exc_info=True)
            self._ready = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if TTS is ready to speak."""
        return _pyttsx3_available and self._ready

    def speak(self, text: str) -> None:
        """Speak *text* aloud.

        Creates a fresh pyttsx3 engine for every utterance to ensure the COM
        apartment is always initialised on the current calling thread, regardless
        of any background threads used by Rich spinners or other UI elements.

        If the engine is unavailable or encounters an error, the call returns
        silently — the text was already printed to the terminal by main.py.

        Parameters
        ----------
        text:
            The string to speak.  Markdown formatting is stripped before
            synthesis to avoid reading raw asterisks and backticks.
        """
        if not self.is_available():
            logger.debug("TTS speak() skipped: TTS not available.")
            return
        if not text or not text.strip():
            logger.debug("TTS speak() skipped: empty text.")
            return

        clean = _strip_markdown(text)
        if not clean.strip():
            logger.debug("TTS speak() skipped: text empty after Markdown strip.")
            return

        t_start = time.perf_counter()
        logger.info(
            "TTS speak() starting | chars=%d stripped_chars=%d",
            len(text), len(clean),
        )

        engine = None
        try:
            # Fresh engine per call — solves COM apartment threading issues.
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)

            engine.say(clean)

            t_synth_start = time.perf_counter()
            logger.debug("TTS engine.say() queued | pre-runAndWait=%.3fs", t_synth_start - t_start)

            engine.runAndWait()

            t_end = time.perf_counter()
            logger.info(
                "TTS speak() done | total=%.2fs | synthesis=%.2fs",
                t_end - t_start,
                t_end - t_synth_start,
            )

        except RuntimeError as exc:
            # runAndWait() can throw RuntimeError if the loop is already running
            # (rare on the main thread but possible if speak() is re-entered).
            logger.warning("TTS speak() RuntimeError: %s", exc)
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
        except Exception as exc:
            logger.error("TTS speak() failed: %s", exc, exc_info=True)
        finally:
            # Always release COM resources.
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_voice_id(self, engine, preference: str) -> Optional[str]:
        """Return the SAPI5 voice ID that best matches *preference*.

        Iterates SAPI5 voices and picks the first one whose name matches.
        Returns None to use the system default if no match is found.
        """
        try:
            voices = engine.getProperty("voices")
            if not voices:
                logger.warning("No TTS voices found; using system default.")
                return None

            pref = preference.lower().strip()

            if pref in ("male", "female"):
                # Strategy 1: gender attribute (available on some platforms)
                for v in voices:
                    gender = getattr(v, "gender", "") or ""
                    if pref in gender.lower():
                        logger.info("TTS voice selected by gender: %s", v.name)
                        return v.id

                # Strategy 2: common name keywords
                _MALE_KEYWORDS = ("david", "mark", "james")
                _FEMALE_KEYWORDS = ("zira", "hazel", "linda", "cortana")
                keywords = _FEMALE_KEYWORDS if pref == "female" else _MALE_KEYWORDS
                for v in voices:
                    if any(kw in v.name.lower() for kw in keywords):
                        logger.info("TTS voice selected by name keyword: %s", v.name)
                        return v.id

            # Fallback: first available voice
            logger.debug(
                "TTS: no %s voice found by keyword; using first voice: %s",
                pref, voices[0].name,
            )
            return voices[0].id

        except Exception as exc:
            logger.warning("TTS voice resolution failed (using default): %s", exc)
            return None


# ------------------------------------------------------------------
# Markdown stripping utility
# ------------------------------------------------------------------

import re as _re

# Patterns to strip before speaking.  Order matters — code blocks first.
_MD_PATTERNS: list[tuple] = [
    (_re.compile(r"```[\s\S]*?```"), ""),           # fenced code blocks
    (_re.compile(r"`[^`]+`"), ""),                   # inline code
    (_re.compile(r"\*\*(.+?)\*\*"), r"\1"),          # **bold**
    (_re.compile(r"\*(.+?)\*"), r"\1"),              # *italic*
    (_re.compile(r"__(.+?)__"), r"\1"),              # __bold__
    (_re.compile(r"_(.+?)_"), r"\1"),                # _italic_
    (_re.compile(r"#{1,6}\s*"), ""),                 # headings
    (_re.compile(r"\[(.+?)\]\(.*?\)"), r"\1"),       # [link text](url) → text
    (_re.compile(r"!\[.*?\]\(.*?\)"), ""),           # ![image](url) → skip
    (_re.compile(r"^[-*+]\s+", _re.MULTILINE), ""), # list bullets
    (_re.compile(r"^\d+\.\s+", _re.MULTILINE), ""), # numbered lists
    (_re.compile(r">+\s*", _re.MULTILINE), ""),      # blockquotes
    (_re.compile(r"-{3,}|={3,}|\*{3,}"), ""),       # horizontal rules
]


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting from *text* before TTS synthesis."""
    result = text
    for pattern, replacement in _MD_PATTERNS:
        result = pattern.sub(replacement, result)
    # Collapse multiple blank lines / leading spaces
    result = _re.sub(r"\n{3,}", "\n\n", result)
    result = _re.sub(r"[ \t]+", " ", result)
    return result.strip()
