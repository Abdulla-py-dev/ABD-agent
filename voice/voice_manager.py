"""
voice/voice_manager.py — Voice Coordinator for ABD V2 Phase 2
==============================================================
Coordinates SpeechToText and TextToSpeech, providing a clean interface
that main.py can use without knowing any voice engine internals.

Flow in Talk Mode:
  1. main.py calls VoiceManager.listen_and_transcribe()
  2. VoiceManager delegates to SpeechToText.listen()
  3. Recognised text → ABDAgent.chat() (in main.py)
  4. main.py calls VoiceManager.speak(response_text)
  5. VoiceManager delegates to TextToSpeech.speak()

All failures are swallowed here — voice errors never crash ABD.

Usage:
    vm = VoiceManager()
    if vm.is_available():
        text = vm.listen_and_transcribe()
        if text:
            response = agent.chat(text)
            vm.speak(response)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("abd.voice.manager")


class VoiceManager:
    """High-level voice coordinator for ABD Talk Mode.

    Initialises STT and TTS on construction.  If either subsystem fails to
    initialise, it is marked unavailable; the other subsystem continues to work
    if possible.

    Parameters
    ----------
    stt_timeout:
        Override the config STT timeout (seconds to wait for speech to start).
    stt_phrase_limit:
        Override the config phrase time limit in seconds.
    tts_rate:
        Override the config TTS words-per-minute rate.
    tts_volume:
        Override the config TTS volume (0.0–1.0).
    tts_voice_preference:
        Override the config voice preference ("male", "female", "any").
    """

    def __init__(
        self,
        stt_timeout: int | None = None,
        stt_phrase_limit: int | None = None,
        tts_rate: int | None = None,
        tts_volume: float | None = None,
        tts_voice_preference: str | None = None,
    ) -> None:
        # Import here so that missing libraries degrade to a warning rather
        # than a module-level ImportError.
        from voice.stt import SpeechToText
        from voice.tts import TextToSpeech

        self._stt = SpeechToText(
            timeout=stt_timeout,
            phrase_time_limit=stt_phrase_limit,
        )
        self._tts = TextToSpeech(
            rate=tts_rate,
            volume=tts_volume,
            voice_preference=tts_voice_preference,
        )

        logger.info(
            "VoiceManager ready | stt_available=%s | tts_available=%s",
            self._stt.is_available(),
            self._tts.is_available(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if at least STT is functional.

        ABD can enter Talk Mode as long as it can receive spoken input.
        If TTS is unavailable, responses are printed but not spoken.
        """
        return self._stt.is_available()

    def stt_available(self) -> bool:
        """Return True if Speech-to-Text is ready."""
        return self._stt.is_available()

    def tts_available(self) -> bool:
        """Return True if Text-to-Speech is ready."""
        return self._tts.is_available()

    def listen_and_transcribe(self) -> Optional[str]:
        """Listen to the microphone and return recognised text.

        Blocks until speech ends, timeout expires, or an error occurs.

        Returns
        -------
        str | None
            The recognised text string, or None if:
            - No speech was detected within the timeout
            - The speech was unintelligible
            - STT is unavailable
            - Any error occurred
        """
        try:
            result = self._stt.listen()
            if result:
                logger.debug("Transcribed: %r", result)
            else:
                logger.debug("No transcription returned.")
            return result
        except Exception as exc:
            # Belt-and-suspenders: SpeechToText.listen() already catches
            # all exceptions, but we add a second guard here.
            logger.error("VoiceManager.listen_and_transcribe() unexpected error: %s", exc)
            return None

    def speak(self, text: str) -> None:
        """Speak *text* using the TTS engine.

        If TTS is unavailable, the call returns silently.  The text is already
        displayed in the terminal by main.py regardless.

        Parameters
        ----------
        text:
            The full response text to speak (not individual streamed tokens).
        """
        if not text or not text.strip():
            return
        try:
            self._tts.speak(text)
        except Exception as exc:
            # Belt-and-suspenders guard (tts.speak() already catches all).
            logger.error("VoiceManager.speak() unexpected error: %s", exc)

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def status_summary(self) -> str:
        """Return a human-readable status string for the startup banner."""
        stt_status = "✓ ready" if self._stt.is_available() else "✗ unavailable"
        tts_status = "✓ ready" if self._tts.is_available() else "✗ unavailable (text only)"
        return f"STT {stt_status} | TTS {tts_status}"
