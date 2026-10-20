"""The small driver interface. Importing it never loads a model framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from audri.audio import AudioFrame
from .types import CancellationToken, SynthesisRequest

# Legacy constants apply only to old 24 kHz byte consumers. New code reads
# AudioFrame.format; Fish uses its own codec rate (normally 44.1 kHz).
SAMPLE_RATE = 24_000
FRAME_SAMPLES = 480
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True, slots=True)
class DriverCapabilities:
    name: str
    native_streaming: bool
    voice_references: bool = False
    instructions: bool = False
    cancellation: str = "transport"


class TTSDriver(ABC):
    """A reusable backend; every stream owns its cancellation and resources.

    Consumers must close the async iterator on early exit (aclosing). Session
    orchestration does this automatically. Closing a transport cannot promise
    that an upstream model server has stopped GPU computation.
    """

    capabilities = DriverCapabilities("custom", native_streaming=False)

    @abstractmethod
    def stream(self, request: SynthesisRequest, *, cancel: CancellationToken | None = None
               ) -> AsyncIterator[AudioFrame]:
        raise NotImplementedError

    async def synth(self, text: str, voice: str = "default", language: str = "English",
                    instruction: str | None = None) -> AsyncIterator[bytes]:
        """Compatibility wrapper. Prefer stream to preserve format metadata."""
        from contextlib import aclosing
        async with aclosing(self.stream(SynthesisRequest(text, voice, language, instruction))) as stream:
            async for frame in stream:
                # Kept for old scripts; not the format of a concurrent stream.
                self.sample_rate = frame.format.sample_rate
                yield frame.pcm

    async def aclose(self) -> None:
        """Release backend resources after sessions have been closed."""

    async def __aenter__(self) -> TTSDriver:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def __getattr__(name: str) -> Any:
    # Preserve existing public imports without importing Torch into Fish clients.
    if name in {"Qwen3TTSDriver", "HFQwen3TTSDriver"}:
        from .qwen import Qwen3TTSDriver
        return Qwen3TTSDriver
    if name == "FishSpeechDriver":
        from .fish import FishSpeechDriver
        return FishSpeechDriver
    if name == "VLLMOmniTTSDriver":
        raise ImportError("The old VLLMOmniTTSDriver was an unimplemented stub. "
                          "Use FishSpeechDriver with the native Fish API; "
                          "Omni serving protocols require their own adapter.")
    raise AttributeError(name)
