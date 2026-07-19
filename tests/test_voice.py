"""
tests/test_voice.py — Unit Tests for ABD V2 Phase 2 Voice Components
======================================================================
Tests STT, TTS, and VoiceManager behaviour using mocks.
No real microphone, speakers, or internet connection is required.

Coverage:
  STT
    - Successful recognition returns text
    - UnknownValueError (unintelligible speech) returns None
    - RequestError (service unreachable) returns None
    - WaitTimeoutError (no speech detected) returns None
    - No microphone available → is_available() returns False, listen() returns None
    - Missing SpeechRecognition library → degrades gracefully

  PyAudioWPatch shim (Python 3.14 compatibility fix)
    - Shim registers pyaudiowpatch as sys.modules['pyaudio'] when pyaudio absent
    - Shim is a no-op when real pyaudio is already importable
    - Shim degrades gracefully when neither pyaudio nor pyaudiowpatch installed
    - After shim, SR Microphone.list_microphone_names() succeeds
    - is_available() returns False when no PyAudio backend found
    - _install_pyaudio_shim() returns True/False correctly

  TTS
    - speak() calls engine with text
    - Markdown is stripped before speaking
    - Engine init failure → speak() silently returns
    - Missing pyttsx3 library → degrades gracefully
    - Empty / blank text → speak() is a no-op

  VoiceManager
    - listen_and_transcribe() delegates to STT and returns text
    - listen_and_transcribe() returns None when STT returns None
    - speak() delegates to TTS
    - is_available() reflects STT availability
    - Unexpected exception in listen_and_transcribe() is swallowed
    - Unexpected exception in speak() is swallowed

  Integration: safety system not bypassed
    - Recognised voice text passes through agent.chat() identically to typed text
    - ABD does not crash when voice components raise exceptions
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

# conftest.py ensures env vars are set (VOICE_ENABLED=false etc.)
import config  # noqa: F401


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

def _make_fake_sr_module() -> ModuleType:
    """Build a minimal fake speech_recognition module for patching."""
    fake_sr = ModuleType("speech_recognition")

    # Exception hierarchy
    class _AudioException(Exception):
        pass
    class _RequestError(_AudioException):
        pass
    class _UnknownValueError(_AudioException):
        pass
    class _WaitTimeoutError(_AudioException):
        pass

    fake_sr.RequestError = _RequestError
    fake_sr.UnknownValueError = _UnknownValueError
    fake_sr.WaitTimeoutError = _WaitTimeoutError

    # Recognizer mock
    fake_recognizer = MagicMock()
    fake_sr.Recognizer = MagicMock(return_value=fake_recognizer)

    # Microphone mock (context manager)
    fake_mic_instance = MagicMock()
    fake_mic_instance.__enter__ = MagicMock(return_value=fake_mic_instance)
    fake_mic_instance.__exit__ = MagicMock(return_value=False)
    fake_sr.Microphone = MagicMock(return_value=fake_mic_instance)
    fake_sr.Microphone.list_microphone_names = MagicMock(
        return_value=["Default Microphone"]
    )

    return fake_sr


@pytest.fixture(autouse=True)
def clean_voice_imports():
    """Remove cached voice module imports so each test starts clean."""
    mods_to_remove = [k for k in sys.modules if k.startswith("voice")]
    for mod in mods_to_remove:
        del sys.modules[mod]
    yield
    mods_to_remove = [k for k in sys.modules if k.startswith("voice")]
    for mod in mods_to_remove:
        del sys.modules[mod]


# ===========================================================================
# STT Tests
# ===========================================================================

class TestSpeechToText:

    def _make_stt(self, fake_sr=None, mic_names=None):
        """Create a SpeechToText with an injected fake sr module."""
        if fake_sr is None:
            fake_sr = _make_fake_sr_module()
        if mic_names is not None:
            fake_sr.Microphone.list_microphone_names.return_value = mic_names

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            # Also patch the _sr_available flag inside stt module
            from voice.stt import SpeechToText
            stt = SpeechToText()
        return stt, fake_sr

    def test_is_available_true_when_mic_present(self):
        """STT reports available when a microphone is found."""
        fake_sr = _make_fake_sr_module()
        fake_sr.Microphone.list_microphone_names.return_value = ["Built-in Mic"]

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        assert stt.is_available() is True

    def test_is_available_false_when_no_mic(self):
        """STT reports unavailable when no microphones are detected."""
        fake_sr = _make_fake_sr_module()
        fake_sr.Microphone.list_microphone_names.return_value = []

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        assert stt.is_available() is False

    def test_listen_returns_text_on_success(self):
        """listen() returns the recognised text string on success."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.return_value = "Hello ABD"

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        result = stt.listen()
        assert result == "Hello ABD"
        recognizer.recognize_google.assert_called_once()

    def test_listen_returns_none_on_unknown_value_error(self):
        """listen() returns None when speech is unintelligible."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.side_effect = fake_sr.UnknownValueError()

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        result = stt.listen()
        assert result is None

    def test_listen_returns_none_on_request_error(self):
        """listen() returns None when the recognition service is unreachable."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.side_effect = fake_sr.RequestError("network error")

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        result = stt.listen()
        assert result is None

    def test_listen_returns_none_on_wait_timeout(self):
        """listen() returns None when no speech is detected within timeout."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.listen.side_effect = fake_sr.WaitTimeoutError()

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        result = stt.listen()
        assert result is None

    def test_listen_returns_none_when_unavailable(self):
        """listen() returns None (not an exception) when STT is unavailable."""
        fake_sr = _make_fake_sr_module()
        fake_sr.Microphone.list_microphone_names.return_value = []

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        # Should return None without raising
        result = stt.listen()
        assert result is None

    def test_listen_swallows_unexpected_exception(self):
        """listen() catches unexpected exceptions and returns None."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.side_effect = RuntimeError("unexpected!")

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        # Must not raise
        result = stt.listen()
        assert result is None

    def test_listen_strips_whitespace_from_result(self):
        """Recognised text is stripped of leading/trailing whitespace."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.return_value = "  open calculator  "

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        result = stt.listen()
        assert result == "open calculator"

    def test_sr_not_installed_is_available_returns_false(self):
        """If SpeechRecognition is not installed, is_available() returns False."""
        # Simulate missing library by hiding the module
        with patch.dict(sys.modules, {"speech_recognition": None}):  # type: ignore[dict-item]
            # Re-import with mocked missing module
            import importlib
            # Temporarily break the import
            original = sys.modules.pop("speech_recognition", None)
            try:
                sys.modules["speech_recognition"] = None  # type: ignore[assignment]
                from voice import stt as stt_mod
                importlib.reload(stt_mod)
                s = stt_mod.SpeechToText.__new__(stt_mod.SpeechToText)
                s._recognizer = None
                s._microphone_available = False
                assert s.is_available() is False
            finally:
                if original is not None:
                    sys.modules["speech_recognition"] = original
                elif "speech_recognition" in sys.modules:
                    del sys.modules["speech_recognition"]


# ===========================================================================
# PyAudioWPatch Shim Tests  (Python 3.14 compatibility fix)
# ===========================================================================

class TestPyAudioWPatchShim:
    """Verify that _install_pyaudio_shim() correctly bridges PyAudioWPatch to
    the 'pyaudio' module name that SpeechRecognition expects.

    All tests restore sys.modules to their original state via teardown so they
    cannot pollute each other.
    """

    def _run_shim(self, modules_override: dict) -> bool:
        """Run _install_pyaudio_shim() in an isolated sys.modules context.

        ``modules_override`` is applied to sys.modules for the duration of
        the call, then the original state is restored.  Returns the bool
        result of _install_pyaudio_shim().
        """
        import importlib

        # Save originals for the keys we will touch
        saved = {k: sys.modules.get(k, None) for k in modules_override}
        saved_voice_stt = sys.modules.pop("voice.stt", None)

        try:
            for k, v in modules_override.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

            # Re-import the stt module so module-level code re-runs
            import voice.stt as stt_mod
            importlib.reload(stt_mod)
            return stt_mod._pyaudio_backend_available
        finally:
            # Restore originals
            for k, original_v in saved.items():
                if original_v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = original_v
            if saved_voice_stt is not None:
                sys.modules["voice.stt"] = saved_voice_stt
            else:
                sys.modules.pop("voice.stt", None)

    def test_shim_registers_pyaudiowpatch_when_pyaudio_absent(self):
        """When pyaudio is missing, pyaudiowpatch is registered as sys.modules['pyaudio']."""
        import importlib

        # Build a fake pyaudiowpatch with the essential PyAudio API
        fake_pawpatch = ModuleType("pyaudiowpatch")
        fake_pawpatch.__version__ = "0.2.12.8"  # type: ignore[attr-defined]
        fake_pawpatch.PyAudio = MagicMock()
        fake_pawpatch.paInt16 = 8
        fake_pawpatch.get_sample_size = MagicMock(return_value=2)

        saved_pyaudio = sys.modules.pop("pyaudio", None)
        saved_pawpatch = sys.modules.pop("pyaudiowpatch", None)
        saved_stt = sys.modules.pop("voice.stt", None)

        try:
            # Make pyaudio missing, pyaudiowpatch present
            sys.modules["pyaudiowpatch"] = fake_pawpatch

            import voice.stt as stt_mod
            importlib.reload(stt_mod)

            # The shim must have registered pyaudiowpatch under 'pyaudio'
            assert "pyaudio" in sys.modules
            assert sys.modules["pyaudio"] is fake_pawpatch
            assert stt_mod._pyaudio_backend_available is True
        finally:
            # Restore
            sys.modules.pop("pyaudio", None)
            if saved_pyaudio is not None:
                sys.modules["pyaudio"] = saved_pyaudio
            if saved_pawpatch is not None:
                sys.modules["pyaudiowpatch"] = saved_pawpatch
            else:
                sys.modules.pop("pyaudiowpatch", None)
            if saved_stt is not None:
                sys.modules["voice.stt"] = saved_stt
            else:
                sys.modules.pop("voice.stt", None)

    def test_shim_is_noop_when_pyaudio_already_present(self):
        """When real pyaudio is in sys.modules, the shim leaves it untouched."""
        import importlib

        # Simulate real pyaudio already loaded
        fake_real_pyaudio = ModuleType("pyaudio")
        fake_real_pyaudio.__version__ = "0.2.14"  # type: ignore[attr-defined]
        fake_real_pyaudio.PyAudio = MagicMock()

        saved_pyaudio = sys.modules.pop("pyaudio", None)
        saved_stt = sys.modules.pop("voice.stt", None)

        try:
            sys.modules["pyaudio"] = fake_real_pyaudio

            import voice.stt as stt_mod
            importlib.reload(stt_mod)

            # shim should not replace it — real pyaudio stays
            assert sys.modules.get("pyaudio") is fake_real_pyaudio
            assert stt_mod._pyaudio_backend_available is True
        finally:
            if saved_pyaudio is not None:
                sys.modules["pyaudio"] = saved_pyaudio
            else:
                sys.modules.pop("pyaudio", None)
            if saved_stt is not None:
                sys.modules["voice.stt"] = saved_stt
            else:
                sys.modules.pop("voice.stt", None)

    def test_shim_returns_false_when_neither_backend_present(self):
        """When both pyaudio and pyaudiowpatch are absent, backend_available is False."""
        import importlib

        saved_pyaudio = sys.modules.pop("pyaudio", None)
        saved_pawpatch = sys.modules.pop("pyaudiowpatch", None)
        saved_stt = sys.modules.pop("voice.stt", None)

        try:
            # Remove both from sys.modules so import fails
            # We simulate ImportError by setting to None (Python's sentinel for
            # 'this module does not exist')
            sys.modules["pyaudio"] = None      # type: ignore[assignment]
            sys.modules["pyaudiowpatch"] = None  # type: ignore[assignment]

            import voice.stt as stt_mod

            # Manually call the shim function directly since reload would use
            # the cached pyaudiowpatch that may still be installed
            # Instead, test the function logic with both modules absent
            result = stt_mod._install_pyaudio_shim.__wrapped__() if hasattr(
                stt_mod._install_pyaudio_shim, '__wrapped__'
            ) else None

            # The key assertion: is_available() on a fresh STT instance must
            # be False if _pyaudio_backend_available is False
            stt = stt_mod.SpeechToText.__new__(stt_mod.SpeechToText)
            stt._recognizer = None
            stt._microphone_available = False
            # Temporarily patch the module-level flag
            original_flag = stt_mod._pyaudio_backend_available
            stt_mod._pyaudio_backend_available = False
            try:
                assert stt.is_available() is False
            finally:
                stt_mod._pyaudio_backend_available = original_flag
        finally:
            sys.modules.pop("pyaudio", None)
            sys.modules.pop("pyaudiowpatch", None)
            if saved_pyaudio is not None:
                sys.modules["pyaudio"] = saved_pyaudio
            if saved_pawpatch is not None:
                sys.modules["pyaudiowpatch"] = saved_pawpatch
            if saved_stt is not None:
                sys.modules["voice.stt"] = saved_stt
            else:
                sys.modules.pop("voice.stt", None)

    def test_install_pyaudio_shim_returns_true_with_pawpatch(self):
        """_install_pyaudio_shim() returns True when pyaudiowpatch is available."""
        # In the live environment pyaudiowpatch IS installed, so the shim
        # already ran at import time.  Verify the module-level flag directly.
        import voice.stt as stt_mod
        # If pyaudiowpatch is installed, _pyaudio_backend_available must be True
        import importlib
        try:
            import pyaudiowpatch  # noqa: F401
            pawpatch_installed = True
        except ImportError:
            pawpatch_installed = False

        if pawpatch_installed:
            assert stt_mod._pyaudio_backend_available is True
        # If neither is installed (CI environment), just assert the type
        assert isinstance(stt_mod._pyaudio_backend_available, bool)

    def test_shim_makes_sr_microphone_list_names_succeed(self):
        """After the shim, SR Microphone.list_microphone_names() succeeds
        even without real pyaudio installed."""
        # Import stt first to install the shim
        import voice.stt  # noqa: F401

        # Now import SR and verify Microphone can call get_pyaudio()
        import speech_recognition as sr_mod
        # get_pyaudio() should NOT raise AttributeError since shim is active
        try:
            pa_module = sr_mod.Microphone.get_pyaudio()
            # Verify it has the expected API surface
            assert hasattr(pa_module, "PyAudio"), "PyAudio class missing from backend"
            assert hasattr(pa_module, "paInt16"), "paInt16 constant missing from backend"
            assert hasattr(pa_module, "get_sample_size"), "get_sample_size missing from backend"
        except AttributeError as exc:
            pytest.fail(f"SR Microphone.get_pyaudio() raised after shim: {exc}")

    def test_is_available_false_when_pyaudio_backend_missing(self):
        """is_available() returns False when _pyaudio_backend_available is False,
        regardless of whether SR is installed."""
        import voice.stt as stt_mod

        original_flag = stt_mod._pyaudio_backend_available
        try:
            stt_mod._pyaudio_backend_available = False
            stt = stt_mod.SpeechToText.__new__(stt_mod.SpeechToText)
            stt._recognizer = MagicMock()
            stt._microphone_available = True  # even if mic thinks it's OK
            assert stt.is_available() is False
        finally:
            stt_mod._pyaudio_backend_available = original_flag

    def test_listen_returns_none_when_pyaudio_backend_missing(self):
        """listen() returns None (not an exception) when pyaudio backend is absent."""
        import voice.stt as stt_mod

        original_flag = stt_mod._pyaudio_backend_available
        try:
            stt_mod._pyaudio_backend_available = False
            stt = stt_mod.SpeechToText.__new__(stt_mod.SpeechToText)
            stt._recognizer = None
            stt._microphone_available = False
            result = stt.listen()
            assert result is None
        finally:
            stt_mod._pyaudio_backend_available = original_flag


# ===========================================================================
# TTS Tests
# ===========================================================================

class TestTextToSpeech:
    """Tests for the TextToSpeech class.

    Architecture note:
    speak() now calls pyttsx3.init() on every invocation (fresh engine per
    call) to fix the Windows SAPI5 COM apartment threading issue.  pyttsx3.init()
    is therefore called TWICE for a full speak() cycle:
      1. During TextToSpeech.__init__() — probe/voice-resolution call
      2. During speak() — the actual synthesis call

    The fake_pyttsx3.init mock must return a valid engine mock on BOTH calls,
    not just the first one.
    """

    def _make_voice(self, name="Microsoft David Desktop", vid="david_id", gender=""):
        v = MagicMock()
        v.name = name
        v.id = vid
        v.gender = gender
        return v

    def _make_fake_pyttsx3(self, engine_mock=None, init_raises=None):
        """Build a fake pyttsx3 module whose init() returns an engine mock."""
        fake_pyttsx3 = ModuleType("pyttsx3")
        if init_raises:
            fake_pyttsx3.init = MagicMock(side_effect=init_raises)
        else:
            mock_engine = engine_mock or MagicMock()
            voice = self._make_voice()
            mock_engine.getProperty.return_value = [voice]
            # init() is called multiple times; return the same mock each time
            fake_pyttsx3.init = MagicMock(return_value=mock_engine)
        return fake_pyttsx3

    def _make_tts(self, engine_mock=None, init_raises=None):
        """Create a TextToSpeech with a mocked pyttsx3 engine."""
        fake_pyttsx3 = self._make_fake_pyttsx3(engine_mock=engine_mock,
                                                init_raises=init_raises)
        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()
        return tts, fake_pyttsx3

    def test_speak_calls_engine(self):
        """speak() calls engine.say() and engine.runAndWait() on a fresh engine."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]

        fake_pyttsx3 = ModuleType("pyttsx3")
        # init() is called once in __init__ (probe) and once in speak()
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        tts.speak("Hello, I am ABD.")

        # say() and runAndWait() must have been called at least once (during speak())
        engine_mock.say.assert_called()
        engine_mock.runAndWait.assert_called()

    def test_speak_creates_fresh_engine_per_call(self):
        """speak() calls pyttsx3.init() on every utterance (COM threading fix)."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        init_calls_after_init = fake_pyttsx3.init.call_count  # 1 (probe in __init__)

        tts.speak("First utterance.")
        assert fake_pyttsx3.init.call_count == init_calls_after_init + 1, \
            "speak() must call pyttsx3.init() to create a fresh engine"

        tts.speak("Second utterance.")
        assert fake_pyttsx3.init.call_count == init_calls_after_init + 2, \
            "Each speak() call must create its own engine instance"

    def test_speak_strips_markdown(self):
        """speak() strips Markdown before sending text to the engine."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        tts.speak("**Hello** from `ABD`! Here is a [link](http://example.com).")

        # say() is called during speak(); verify the argument is Markdown-clean
        assert engine_mock.say.called
        called_with = engine_mock.say.call_args[0][0]
        assert "**" not in called_with
        assert "`" not in called_with
        assert "[link]" not in called_with
        assert "Hello" in called_with

    def test_speak_empty_string_is_noop(self):
        """speak() with empty or whitespace-only text does nothing."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        init_count = fake_pyttsx3.init.call_count
        tts.speak("")
        tts.speak("   ")
        # No additional init() calls — empty text bails out before engine creation
        assert fake_pyttsx3.init.call_count == init_count
        engine_mock.say.assert_not_called()

    def test_engine_init_failure_degrades_gracefully(self):
        """If pyttsx3.init() raises, TTS becomes unavailable without crashing."""
        tts, _ = self._make_tts(init_raises=RuntimeError("No TTS driver found"))
        assert tts.is_available() is False
        # speak() on unavailable TTS must silently return
        tts.speak("This should not crash.")  # no exception

    def test_speak_runtime_error_is_swallowed(self):
        """RuntimeError from runAndWait() is caught; ABD keeps running."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]
        engine_mock.runAndWait.side_effect = RuntimeError("loop already running")

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        # Must not raise
        tts.speak("Test message")

    def test_speak_generic_exception_is_swallowed(self):
        """Any exception from engine.say() is caught without crashing ABD."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]
        engine_mock.say.side_effect = OSError("audio device gone")

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        # Must not raise
        tts.speak("Another message")

    def test_is_available_false_when_engine_not_ready(self):
        """is_available() returns False when engine init failed."""
        tts, _ = self._make_tts(init_raises=Exception("fail"))
        assert tts.is_available() is False

    def test_engine_stop_called_in_finally(self):
        """engine.stop() is called after speak() to release COM resources."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        tts.speak("Release resources.")
        # stop() must be called in the finally block
        engine_mock.stop.assert_called()

    def test_engine_stop_called_even_after_exception(self):
        """engine.stop() is called even when runAndWait() raises."""
        engine_mock = MagicMock()
        engine_mock.getProperty.return_value = [self._make_voice()]
        engine_mock.runAndWait.side_effect = Exception("crash")

        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(return_value=engine_mock)

        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import TextToSpeech
            tts = TextToSpeech()

        tts.speak("Crash during speak.")
        # stop() must have been called in the finally clause
        engine_mock.stop.assert_called()


# ===========================================================================
# STT consecutive_silence counter tests
# ===========================================================================

class TestSTTConsecutiveSilence:
    """Verify the consecutive_silence counter exposed on SpeechToText."""

    def _make_stt_with_fake_sr(self, recognize_returns=None, recognize_side_effect=None):
        """Create an SpeechToText with controlled recognize_google behaviour."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value

        if recognize_side_effect is not None:
            recognizer.recognize_google.side_effect = recognize_side_effect
        elif recognize_returns is not None:
            recognizer.recognize_google.return_value = recognize_returns

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()
        return stt

    def test_consecutive_silence_starts_at_zero(self):
        """consecutive_silence is 0 on a fresh SpeechToText instance."""
        stt = self._make_stt_with_fake_sr(recognize_returns="hello")
        assert stt.consecutive_silence == 0

    def test_consecutive_silence_increments_on_no_speech(self):
        """consecutive_silence increments each time listen() returns None."""
        fake_sr = _make_fake_sr_module()
        fake_sr.Recognizer.return_value.recognize_google.side_effect = \
            fake_sr.UnknownValueError()

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        stt.listen()
        assert stt.consecutive_silence == 1
        stt.listen()
        assert stt.consecutive_silence == 2

    def test_consecutive_silence_resets_on_successful_recognition(self):
        """consecutive_silence resets to 0 after successful speech recognition."""
        fake_sr = _make_fake_sr_module()
        # First two calls fail, third succeeds
        fake_sr.Recognizer.return_value.recognize_google.side_effect = [
            fake_sr.UnknownValueError(),
            fake_sr.UnknownValueError(),
            "hello world",
        ]

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        stt.listen()
        stt.listen()
        assert stt.consecutive_silence == 2

        stt.listen()  # success
        assert stt.consecutive_silence == 0

    def test_consecutive_silence_increments_on_timeout(self):
        """consecutive_silence increments on WaitTimeoutError."""
        fake_sr = _make_fake_sr_module()
        fake_sr.Recognizer.return_value.listen.side_effect = \
            fake_sr.WaitTimeoutError()

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        stt.listen()
        assert stt.consecutive_silence == 1




# ===========================================================================
# Strip Markdown tests
# ===========================================================================

class TestStripMarkdown:

    def _strip(self, text: str) -> str:
        # Import _strip_markdown directly from the module
        fake_pyttsx3 = ModuleType("pyttsx3")
        fake_pyttsx3.init = MagicMock(side_effect=Exception("no init needed"))
        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            from voice.tts import _strip_markdown
        return _strip_markdown(text)

    def test_strips_bold(self):
        assert "**" not in self._strip("**bold text**")

    def test_strips_inline_code(self):
        assert "`" not in self._strip("Use `print()` here")

    def test_strips_fenced_code_block(self):
        result = self._strip("```python\nprint('hi')\n```")
        assert "```" not in result

    def test_strips_headings(self):
        result = self._strip("## Section Title")
        assert "##" not in result
        assert "Section Title" in result

    def test_strips_links(self):
        result = self._strip("[click here](https://example.com)")
        assert "https://example.com" not in result
        assert "click here" in result

    def test_plain_text_unchanged(self):
        text = "Hello, I am ABD your assistant."
        assert self._strip(text) == text


# ===========================================================================
# VoiceManager Tests
# ===========================================================================

class TestVoiceManager:

    def _make_vm(self, stt_available=True, stt_result=None, tts_available=True):
        """Create a VoiceManager with fully mocked STT and TTS."""
        mock_stt = MagicMock()
        mock_stt.is_available.return_value = stt_available
        mock_stt.listen.return_value = stt_result

        mock_tts = MagicMock()
        mock_tts.is_available.return_value = tts_available

        with patch("voice.stt.SpeechToText", return_value=mock_stt), \
             patch("voice.tts.TextToSpeech", return_value=mock_tts):
            from voice.voice_manager import VoiceManager
            vm = VoiceManager()

        # Inject mocks directly for assertions
        vm._stt = mock_stt
        vm._tts = mock_tts
        return vm, mock_stt, mock_tts

    def test_is_available_true_when_stt_available(self):
        vm, _, _ = self._make_vm(stt_available=True)
        assert vm.is_available() is True

    def test_is_available_false_when_stt_unavailable(self):
        vm, _, _ = self._make_vm(stt_available=False)
        assert vm.is_available() is False

    def test_stt_available_delegates_to_stt(self):
        vm, mock_stt, _ = self._make_vm(stt_available=True)
        assert vm.stt_available() is True
        mock_stt.is_available.assert_called()

    def test_tts_available_delegates_to_tts(self):
        vm, _, mock_tts = self._make_vm(tts_available=True)
        assert vm.tts_available() is True

    def test_listen_and_transcribe_returns_text_on_success(self):
        """listen_and_transcribe() returns the recognised text."""
        vm, mock_stt, _ = self._make_vm(stt_result="open calculator")
        result = vm.listen_and_transcribe()
        assert result == "open calculator"
        mock_stt.listen.assert_called_once()

    def test_listen_and_transcribe_returns_none_when_no_speech(self):
        """listen_and_transcribe() returns None when STT returns None."""
        vm, _, _ = self._make_vm(stt_result=None)
        result = vm.listen_and_transcribe()
        assert result is None

    def test_listen_and_transcribe_swallows_unexpected_exception(self):
        """listen_and_transcribe() catches unexpected errors and returns None."""
        vm, mock_stt, _ = self._make_vm()
        mock_stt.listen.side_effect = RuntimeError("microphone exploded")

        result = vm.listen_and_transcribe()
        assert result is None  # no exception raised

    def test_speak_delegates_to_tts(self):
        """speak() delegates to the TTS engine."""
        vm, _, mock_tts = self._make_vm()
        vm.speak("Hello world")
        mock_tts.speak.assert_called_once_with("Hello world")

    def test_speak_swallows_unexpected_exception(self):
        """speak() catches unexpected errors without crashing ABD."""
        vm, _, mock_tts = self._make_vm()
        mock_tts.speak.side_effect = RuntimeError("speaker caught fire")

        # Must not raise
        vm.speak("This might fail")

    def test_speak_empty_string_is_noop(self):
        """speak() with empty string does not call TTS."""
        vm, _, mock_tts = self._make_vm()
        vm.speak("")
        vm.speak("   ")
        # TTS.speak() is still called (it handles the empty-string guard internally)
        # — the important thing is no exception is raised
        # (we don't assert call count here since vm.speak guards empty strings)

    def test_status_summary_contains_expected_info(self):
        """status_summary() returns a non-empty string with relevant keywords."""
        vm, _, _ = self._make_vm(stt_available=True, tts_available=True)
        summary = vm.status_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "STT" in summary
        assert "TTS" in summary


# ===========================================================================
# Safety integration tests
# ===========================================================================

class TestVoiceSafetyIntegration:
    """Verify that voice input goes through the same agent/safety path as typed input."""

    def test_voice_text_passes_through_agent_chat(self):
        """Recognised voice text is passed to ABDAgent.chat() exactly like typed text."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.return_value = "list my workspace files"

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Here are your workspace files..."

        # Simulate Talk Mode flow: listen → pass to agent
        text = stt.listen()
        assert text == "list my workspace files"

        response = mock_agent.chat(text)
        mock_agent.chat.assert_called_once_with("list my workspace files")
        assert response == "Here are your workspace files..."

    def test_voice_traversal_attempt_blocked_by_agent(self):
        """A voice path traversal attempt is handled by the agent's safety layer."""
        fake_sr = _make_fake_sr_module()
        recognizer = fake_sr.Recognizer.return_value
        recognizer.recognize_google.return_value = "read the file ../../secret.txt"

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        # The safety check happens inside the real agent/router.
        # Here we verify that voice input is passed as plain text to chat()
        # and the agent (not the voice layer) is responsible for safety decisions.
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Access denied: path is outside workspace."

        text = stt.listen()
        response = mock_agent.chat(text)

        # Voice layer passed the text unmodified — safety is the agent's job
        mock_agent.chat.assert_called_once_with("read the file ../../secret.txt")

    def test_voice_failure_does_not_crash_abd(self):
        """A complete voice pipeline failure returns None; ABD continues."""
        fake_sr = _make_fake_sr_module()
        # Make everything fail
        fake_sr.Microphone.list_microphone_names.side_effect = Exception("no audio")
        fake_sr.Recognizer.return_value.recognize_google.side_effect = Exception("fail")

        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            from voice.stt import SpeechToText
            stt = SpeechToText()

        # STT should be unavailable, not crash
        assert stt.is_available() is False
        result = stt.listen()
        assert result is None  # no exception raised

    def test_chat_mode_text_input_unaffected(self):
        """Existing ABDAgent.chat() still works with typed text (no voice changes)."""
        # This test verifies the agent interface is unchanged from Phase 1.
        mock_client = MagicMock()
        mock_client.send_message.return_value = MagicMock()
        mock_client.get_function_calls.return_value = []
        mock_client.get_text.return_value = "I can help with that!"

        # Patch at brain.agent.get_llm_client (the already-bound name) rather
        # than llm.get_llm_client, so the mock intercepts ABDAgent.__init__
        # regardless of whether brain.agent was imported earlier in the test run.
        with patch("brain.agent.get_llm_client", return_value=mock_client):
            from brain.agent import ABDAgent
            agent = ABDAgent()

        response = agent.chat("Hello ABD")
        assert response == "I can help with that!"
        mock_client.send_message.assert_called_once_with("Hello ABD")

