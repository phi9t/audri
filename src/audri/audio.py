"""Unambiguous, dependency-free PCM values shared by every Audri component.

PCM is signed 16-bit little-endian, interleaved by channel. Sample rate travels
with every frame; it is never inferred from a global constant or a model name.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or not 8_000 <= self.sample_rate <= 192_000:
            raise ValueError("sample_rate must be an integer between 8000 and 192000 Hz")
        if type(self.channels) is not int or not 1 <= self.channels <= 8:
            raise ValueError("channels must be an integer between 1 and 8")
        if type(self.sample_width) is not int or self.sample_width != 2:
            raise ValueError("Audri frames require signed 16-bit little-endian PCM")

    @property
    def bytes_per_sample(self) -> int:
        """Bytes in one sample instant across all channels (WAV block align)."""
        return self.channels * self.sample_width

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.bytes_per_sample

    @property
    def encoding(self) -> str:
        return "pcm_s16le"

    def as_dict(self) -> dict[str, int | str]:
        return {"sample_rate": self.sample_rate, "channels": self.channels,
                "sample_width": self.sample_width, "encoding": self.encoding}


@dataclass(frozen=True, slots=True)
class AudioFrame:
    pcm: bytes
    format: AudioFormat

    def __post_init__(self) -> None:
        if not isinstance(self.pcm, bytes):
            raise TypeError("pcm must be immutable bytes")
        if not isinstance(self.format, AudioFormat):
            raise TypeError("format must be an AudioFormat")
        if not self.pcm:
            raise ValueError("an audio frame must contain at least one sample")
        if len(self.pcm) % self.format.bytes_per_sample:
            raise ValueError("PCM data ends inside a sample")

    @property
    def samples(self) -> int:
        return len(self.pcm) // self.format.bytes_per_sample

    @property
    def duration_s(self) -> float:
        return self.samples / self.format.sample_rate


class PCMFramer:
    """Reframe arbitrary chunks without padding, dropping, or resampling.

    The final frame can be shorter than frame_ms. An incomplete sample at EOF
    is an error, not an excuse to silently discard a corrupt byte.
    """

    def __init__(self, format: AudioFormat, frame_ms: float = 20) -> None:
        if not math.isfinite(frame_ms) or not 1 <= frame_ms <= 1000:
            raise ValueError("frame_ms must be finite and between 1 and 1000")
        self.format = format
        self.frame_samples = max(1, round(format.sample_rate * frame_ms / 1000))
        self.frame_bytes = self.frame_samples * format.bytes_per_sample
        self._pending = bytearray()
        self._finished = False

    def feed(self, data: bytes) -> list[AudioFrame]:
        if self._finished:
            raise RuntimeError("cannot feed a finished PCM framer")
        self._pending.extend(data)
        frames = []
        end = len(self._pending) // self.frame_bytes * self.frame_bytes
        for start in range(0, end, self.frame_bytes):
            frames.append(AudioFrame(bytes(self._pending[start:start + self.frame_bytes]),
                                     self.format))
        del self._pending[:end]
        return frames

    def finish(self) -> list[AudioFrame]:
        if self._finished:
            return []
        self._finished = True
        if len(self._pending) % self.format.bytes_per_sample:
            self._pending.clear()
            raise ValueError("truncated PCM stream: incomplete final sample")
        if not self._pending:
            return []
        tail = AudioFrame(bytes(self._pending), self.format)
        self._pending.clear()
        return [tail]
