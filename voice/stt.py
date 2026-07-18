"""
voice/stt.py — Speech-to-Text for ABD V2 Phase 2
===================================================
Wraps SpeechRecognition to convert microphone input into text.

Design principles:
  - Never raises into the caller.  All errors are caught and None is returned.
  - Microphone unavailability is detected and reported gracefully.
  - Uses Google Web Speech API for recognition (free, no key, works offline
    with `recognize_sphinx` as a future fallback if needed).
  - The class is designed so the engine can be swapped (Whisper, Azure, etc.)
    without changing the public interface.

Python 3.14 / PyAudioWPatch compatibility
------------------------------------------
PyAudio (the standard backend for SpeechRecognition's Microphone class) cannot
be compiled on Python 3.14 without Microsoft Visual C++ Build Tools.
PyAudioWPatch is a maintained fork with identical API that ships pre-built
``cp314-cp314-win_amd64`` wheels.

However, SpeechRecognition.Microphone.get_pyaudio() does a hard ``import pyaudio``
by module name.  PyAudioWPatch installs under the name ``pyaudiowpatch``, not
``pyaudio``.

The fix: inject ``pyaudiowpatch`` into ``sys.modules['pyaudio']`` before
SpeechRecognition is imported.  This is the standard Python shim pattern — it is
safe, fully supported, and requires no changes to installed packages or
SpeechRecognition's source code.  The shim only takes effect when PyAudio
(the real package) is NOT already installed; it is a no-op otherwise.

Usage (inside voice_manager.py):
    stt = SpeechToText()
    text = stt.listen()   # blocks until speech ends or timeout
    if text:
        agent.chat(text)
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import config

logger = logging.getLogger("abd.voice.stt")


# ---------------------------------------------------------------------------
# PyAudio shim — inject PyAudioWPatch as 'pyaudio' when needed (Python 3.14+)
# ---------------------------------------------------------------------------

def _install_pyaudio_shim() -> bool:
    """Register pyaudiowpatch under the 'pyaudio' module name if needed.

    SpeechRecognition.Microphone.get_pyaudio() does ``import pyaudio``
    by name.  PyAudioWPatch exposes the identical API under ``pyaudiowpatch``.

    This function runs once at module load time and:
      1. Returns immediately if real ``pyaudio`` is already importable (no-op).
      2. Tries to import ``pyaudiowpatch`` and registers it as ``sys.modules['pyaudio']``.
      3. Logs a clear message so the operator knows which backend is active.
      4. Returns True if a working pyaudio backend is now in sys.modules.
         Returns False if neither pyaudio nor pyaudiowpatch is available.

    This function never raises — errors are caught and logged.
    """
    # Case 1: real pyaudio already available — nothing to do.
    if "pyaudio" in sys.modules:
        return True

    try:
        import pyaudio  # noqa: F401 — standard package present
        logger.debug("PyAudio backend: using standard pyaudio.")
        return True
    except ImportError:
        pass  # expected on Python 3.14 without Build Tools

    # Case 2: try PyAudioWPatch as a drop-in.
    try:
        import pyaudiowpatch  # type: ignore[import-untyped]
        sys.modules["pyaudio"] = pyaudiowpatch  # shim
        logger.info(
            "PyAudio shim active: pyaudiowpatch %s registered as 'pyaudio'. "
            "SpeechRecognition will use PyAudioWPatch transparently.",
            getattr(pyaudiowpatch, "__version__", "unknown"),
        )
        return True
    except ImportError:
        logger.warning(
            "Neither pyaudio nor pyaudiowpatch is installed. "
            "Microphone input will be unavailable. "
            "Install with: pip install PyAudioWPatch"
        )
        return False
    except Exception as exc:
        logger.error("Unexpected error while installing PyAudio shim: %s", exc)
        return False


# Install the shim before importing SpeechRecognition so that
# Microphone.get_pyaudio() finds 'pyaudio' in sys.modules on first call.
_pyaudio_backend_available: bool = _install_pyaudio_shim()


# ---------------------------------------------------------------------------
# Lazy import of SpeechRecognition
# ---------------------------------------------------------------------------

_sr_available: bool = False
try:
    import speech_recognition as sr  # type: ignore[import-untyped]
    _sr_available = True
except ImportError:
    logger.warning(
        "SpeechRecognition library not found. "
        "Install it with: pip install SpeechRecognition PyAudioWPatch"
    )


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------

class STTError(Exception):
    """Raised internally — never propagated to the caller."""


class SpeechToText:
    """Listens to the microphone and converts speech to text.

    All public methods return None rather than raising on error, so a voice
    failure never crashes the ABD application.

    Parameters
    ----------
    timeout:
        Seconds to wait for the user to start speaking (default: config value).
    phrase_time_limit:
        Maximum seconds of speech to capture per utterance (default: config value).
    """

    def __init__(
        self,
        timeout: int | None = None,
        phrase_time_limit: int | None = None,
    ) -> None:
        self._timeout = timeout if timeout is not None else config.VOICE_STT_TIMEOUT
        self._phrase_limit = (
            phrase_time_limit if phrase_time_limit is not None
            else config.VOICE_STT_PHRASE_LIMIT
        )
        self._recognizer: Optional[object] = None
        self._microphone_available: bool = False
        # Tracks how many consecutive listen() calls returned None (no speech).
        # Exposed so Talk Mode can throttle retry prompts.
        self.consecutive_silence: int = 0

        if not _sr_available:
            logger.warning("STT disabled: SpeechRecognition not installed.")
            return

        if not _pyaudio_backend_available:
            logger.warning(
                "STT disabled: no PyAudio backend found. "
                "Install PyAudioWPatch: pip install PyAudioWPatch"
            )
            return

        self._recognizer = sr.Recognizer()
        self._microphone_available = self._check_microphone()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if STT is ready to listen."""
        return _sr_available and _pyaudio_backend_available and self._microphone_available

    def listen(self) -> Optional[str]:
        """Block until the user speaks, then return the recognised text.

        Updates ``self.consecutive_silence``: increments on None, resets on
        successful recognition so callers can throttle retry prompts.

        Returns
        -------
        str | None
            The recognised text, or None if:
            - No microphone is available
            - The user was silent / unintelligible
            - The recognition service is unreachable
            - Any other error
        """
        if not self.is_available():
            logger.warning("STT listen() called but STT is not available.")
            return None

        t_listen_start = time.perf_counter()
        try:
            result = self._listen_and_recognise()
        except Exception as exc:
            logger.error("Unexpected STT error: %s", exc, exc_info=True)
            self.consecutive_silence += 1
            return None

        t_listen_end = time.perf_counter()
        if result:
            logger.info(
                "STT listen() succeeded | text=%r | elapsed=%.2fs",
                result, t_listen_end - t_listen_start,
            )
            self.consecutive_silence = 0
        else:
            self.consecutive_silence += 1
            logger.debug(
                "STT listen() returned None | elapsed=%.2fs | consecutive_silence=%d",
                t_listen_end - t_listen_start, self.consecutive_silence,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _listen_and_recognise(self) -> Optional[str]:
        """Core listen + recognise logic — may raise, caller handles."""
        recognizer = self._recognizer

        t0 = time.perf_counter()
        with sr.Microphone() as source:  # type: ignore[attr-defined]
            t_mic_open = time.perf_counter()
            logger.debug("Microphone opened | open_latency=%.3fs", t_mic_open - t0)

            logger.debug("Calibrating for ambient noise (0.5 s)…")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)  # type: ignore[union-attr]
            t_calibrated = time.perf_counter()
            logger.debug("Ambient noise calibrated | calibration=%.3fs", t_calibrated - t_mic_open)

            logger.debug(
                "Listening… (timeout=%ds, phrase_limit=%ds)",
                self._timeout, self._phrase_limit,
            )
            try:
                audio = recognizer.listen(  # type: ignore[union-attr]
                    source,
                    timeout=self._timeout,
                    phrase_time_limit=self._phrase_limit,
                )
                t_audio_captured = time.perf_counter()
                logger.debug(
                    "Audio captured | capture=%.3fs (excl. calibration)",
                    t_audio_captured - t_calibrated,
                )
            except sr.WaitTimeoutError:  # type: ignore[attr-defined]
                logger.debug(
                    "STT: no speech detected within %ds timeout.", self._timeout
                )
                return None

        # --- Recognise ---
        t_recog_start = time.perf_counter()
        try:
            text = recognizer.recognize_google(audio)  # type: ignore[union-attr]
            t_recog_end = time.perf_counter()
            logger.info(
                "STT recognised: %r | recognition_latency=%.2fs",
                text, t_recog_end - t_recog_start,
            )
            return text.strip() if text else None
        except sr.UnknownValueError:  # type: ignore[attr-defined]
            logger.debug("STT: speech was unintelligible.")
            return None
        except sr.RequestError as exc:  # type: ignore[attr-defined]
            logger.error("STT recognition service error: %s", exc)
            return None

    def _check_microphone(self) -> bool:
        """Return True if at least one microphone device is accessible."""
        try:
            mic_names = sr.Microphone.list_microphone_names()  # type: ignore[attr-defined]
            if mic_names:
                logger.debug("Microphones found: %s", mic_names)
                return True
            logger.warning("No microphone devices found.")
            return False
        except Exception as exc:
            logger.warning("Could not enumerate microphones: %s", exc)
            return False
