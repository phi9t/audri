"""Audri: model-independent streaming speech with explicit audio formats."""

from .audio import AudioFormat, AudioFrame, PCMFramer
from .speak_tool import SpeakTool
from .tts.driver import TTSDriver
from .tts.session import SessionConfig, SessionMetrics
from .tts.types import CancellationToken, SynthesisRequest, VoiceReference

__version__ = "0.2.0"
__all__ = ["AudioFormat", "AudioFrame", "PCMFramer", "CancellationToken",
           "SynthesisRequest", "VoiceReference", "SpeakTool", "TTSDriver",
           "SessionConfig", "SessionMetrics"]
