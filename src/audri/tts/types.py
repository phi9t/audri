"""Model-independent requests and request-scoped cancellation."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import math


class CancellationToken:
    """An event owned by one utterance, never shared as driver-global state."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError


@dataclass(frozen=True, slots=True)
class VoiceReference:
    """Encoded reference audio and its transcript; never a remote filesystem path."""

    audio: bytes
    text: str
    speaker: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes) or not 1 <= len(self.audio) <= 10 * 1024 * 1024:
            raise ValueError("reference audio must contain 1 byte to 10 MiB of encoded audio")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 20_000:
            raise ValueError("reference text must contain 1 to 20000 characters")
        if self.speaker is not None and (type(self.speaker) is not int or not 0 <= self.speaker <= 15):
            raise ValueError("reference speaker must be an integer from 0 to 15")

    def as_dict(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {
            "audio": base64.b64encode(self.audio).decode("ascii"), "text": self.text,
        }
        if self.speaker is not None:
            result["speaker"] = self.speaker
        return result

    def to_json(self) -> dict[str, str | int]:
        return self.as_dict()


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    voice: str = "default"
    language: str = "English"
    instruction: str | None = None
    reference_id: str | None = None
    references: tuple[VoiceReference, ...] = ()
    seed: int | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    chunk_length: int = 200
    normalize: bool = True
    single_chunk: bool = False

    def __post_init__(self) -> None:
        # Empty text is permitted for session templates. A backend emits no
        # frames for it. Conversation-wide text limits belong to the session.
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        for name in ("voice", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f"{name} must contain 1 to 256 characters")
        if self.instruction is not None and (not isinstance(self.instruction, str)
                                             or len(self.instruction) > 4096):
            raise ValueError("instruction must be a string of at most 4096 characters")
        if self.reference_id is not None and (not isinstance(self.reference_id, str)
                                             or not self.reference_id.strip()
                                             or len(self.reference_id) > 256):
            raise ValueError("reference_id must contain 1 to 256 characters")
        if not isinstance(self.references, tuple) or len(self.references) > 16:
            raise ValueError("references must be a tuple of at most 16 VoiceReference values")
        if any(not isinstance(ref, VoiceReference) for ref in self.references):
            raise TypeError("references must contain VoiceReference values")
        if sum(len(ref.audio) for ref in self.references) > 32 * 1024 * 1024:
            raise ValueError("total reference audio cannot exceed 32 MiB")
        if self.reference_id is not None and self.references:
            raise ValueError("choose reference_id or inline references, not both")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 16384:
            raise ValueError("max_new_tokens must be between 1 and 16384")
        if type(self.chunk_length) is not int or not 100 <= self.chunk_length <= 300:
            raise ValueError("chunk_length must be between 100 and 300")
        for name, lower, upper in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0),
                                   ("repetition_penalty", 0.0, 2.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"{name} must be finite and in ({lower}, {upper}]")
        if type(self.normalize) is not bool or type(self.single_chunk) is not bool:
            raise ValueError("normalize and single_chunk must be booleans")
