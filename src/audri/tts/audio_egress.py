"""Audio egress — PCM frame consumer.

LocalAudioEgress is a stub that raises NotImplementedError; local audio
playback via sounddevice is not included in this build (headless-only).

Use NullAudioEgress (or the HeadlessDriver in speak_demo.py) for headless
operation that writes WAV files instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "int16"
FRAME_SAMPLES = 480      # 20ms @ 24kHz


class LocalAudioEgress:
    """Stub — local audio playback is not available in this (headless) build.

    Instantiation succeeds so that SpeakTool wiring doesn't break, but
    calling run() raises NotImplementedError immediately.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def run(self, pcm_queue: asyncio.Queue[Optional[bytes]]) -> None:
        raise NotImplementedError(
            "LocalAudioEgress is not available in headless builds. "
            "Use AUDRI_HEADLESS=1 (NullAudioEgress) or the HeadlessDriver."
        )
