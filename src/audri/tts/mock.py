"""Diagnostic tones for transport tests, not speech or model benchmarking."""

from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncIterator

from audri.audio import AudioFormat, AudioFrame, PCMFramer
from .driver import DriverCapabilities, TTSDriver
from .types import CancellationToken, SynthesisRequest


class ToneDriver(TTSDriver):
    capabilities = DriverCapabilities("diagnostic-tone", native_streaming=True,
                                       cancellation="cooperative")

    def __init__(self, sample_rate: int = 24_000, duration_s: float = 0.25,
                 frequency: float = 440.0, frame_ms: float = 20.0) -> None:
        self.format = AudioFormat(sample_rate)
        if not math.isfinite(duration_s) or not 0 < duration_s <= 60:
            raise ValueError("duration_s must be finite and in (0, 60]")
        if not math.isfinite(frequency) or not 0 < frequency < sample_rate / 2:
            raise ValueError("frequency must be positive and below Nyquist")
        self.duration_s = duration_s
        self.frequency = frequency
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        PCMFramer(self.format, frame_ms)

    async def stream(self, request: SynthesisRequest, *, cancel: CancellationToken | None = None
                     ) -> AsyncIterator[AudioFrame]:
        if cancel:
            cancel.raise_if_cancelled()
        if not request.text.strip():
            return
        count = max(1, round(self.sample_rate * self.duration_s))
        framer = PCMFramer(self.format, self.frame_ms)
        fade = max(1, min(count // 2, self.sample_rate // 200))
        for start in range(0, count, framer.frame_samples):
            if cancel:
                cancel.raise_if_cancelled()
            samples = []
            for index in range(start, min(count, start + framer.frame_samples)):
                envelope = min(1.0, index / fade, (count - index - 1) / fade)
                value = 4096 * envelope * math.sin(2 * math.pi * self.frequency * index / self.sample_rate)
                samples.append(round(value))
            yield AudioFrame(struct.pack(f"<{len(samples)}h", *samples), self.format)
            await asyncio.sleep(0)
