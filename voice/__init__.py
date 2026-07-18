"""
voice/__init__.py — ABD V2 Phase 2 Voice Package
=================================================
Provides Speech-to-Text (STT) and Text-to-Speech (TTS) for ABD Talk Mode.

Public surface:
    from voice.voice_manager import VoiceManager

Voice input is just another input channel. All recognised text passes through
the existing ABDAgent.chat() / safety/permissions.py pipeline unchanged.
"""
