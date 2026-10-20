"""Incremental, bounded parsing of PCM16 RIFF/WAVE responses.

HTTP chunk boundaries have no relationship to WAV chunks or sample boundaries.
In particular, Fish Speech sends a normal WAV header with unknown RIFF/data
lengths (0xffffffff) followed by little-endian signed 16-bit samples.
"""

from __future__ import annotations

import struct

from audri.audio import AudioFormat


class WavFormatError(ValueError):
    """The input is unsupported, malformed, truncated, or exceeds header limits."""


class WavStreamParser:
    """Strip a WAV container without buffering the waveform.

    ``feed`` returns only PCM bytes; these bytes may end within a sample. Pass
    them to ``PCMFramer`` before playback. ``finish`` validates sample alignment
    and finite container lengths. Both ordinary files and streaming WAVs work.

    Only uncompressed PCM16 little-endian WAV is accepted. Rejecting float WAV,
    compressed WAV, and RF64 is deliberate: interpreting them as PCM16 would
    produce noise. Ancillary chunks are skipped with a bounded metadata budget.
    """

    def __init__(self, *, max_header_bytes: int = 1 << 20) -> None:
        if (
            isinstance(max_header_bytes, bool)
            or not isinstance(max_header_bytes, int)
            or max_header_bytes < 44
        ):
            raise ValueError("max_header_bytes must be at least 44")
        self.format: AudioFormat | None = None
        self._limit = max_header_bytes
        self._buffer = bytearray()
        self._state = "riff"
        self._chunk = b""
        self._remaining = 0
        self._padding = 0
        self._riff_bytes: int | None = None
        self._received = 0
        self._metadata_bytes = 12
        self._audio_bytes = 0
        self._saw_data = False
        self._unknown_data = False
        self._finished = False

    def feed(self, data: bytes) -> bytes:
        if self._finished:
            raise WavFormatError("Cannot feed a finished WAV parser")
        if not isinstance(data, bytes):
            raise TypeError("WAV input must be bytes")
        self._received += len(data)
        if self._riff_bytes is not None and self._received > self._riff_bytes:
            raise WavFormatError("WAV contains bytes beyond its declared RIFF length")
        self._buffer.extend(data)
        output: list[bytes] = []
        while True:
            if self._state == "riff":
                if len(self._buffer) < 12:
                    break
                marker, size, kind = struct.unpack("<4sI4s", self._take(12))
                if marker != b"RIFF" or kind != b"WAVE":
                    raise WavFormatError("Expected a little-endian RIFF/WAVE stream")
                if size != 0xFFFFFFFF:
                    if size < 4:
                        raise WavFormatError("Invalid RIFF length")
                    self._riff_bytes = size + 8
                    if self._received > self._riff_bytes:
                        raise WavFormatError("WAV exceeds its declared RIFF length")
                self._state = "chunk"
            elif self._state == "chunk":
                if len(self._buffer) < 8:
                    break
                chunk, size = struct.unpack("<4sI", self._take(8))
                self._metadata_bytes += 8
                if self._metadata_bytes > self._limit:
                    raise WavFormatError("WAV metadata exceeds the configured header limit")
                self._chunk = chunk
                self._padding = size & 1
                if chunk == b"data":
                    if self.format is None:
                        raise WavFormatError("WAV data precedes its format chunk")
                    if self._saw_data:
                        raise WavFormatError("Multiple WAV data chunks are unsupported")
                    self._saw_data = True
                    self._unknown_data = size == 0xFFFFFFFF
                    self._remaining = size
                    if not self._unknown_data and size % (self.format.channels * 2):
                        raise WavFormatError("WAV data length is not sample-aligned")
                    if self._unknown_data and self._riff_bytes is not None:
                        raise WavFormatError("Unknown data length requires an unknown RIFF length")
                    self._state = "data"
                else:
                    self._metadata_bytes += size + self._padding
                    if self._metadata_bytes > self._limit:
                        raise WavFormatError("WAV metadata exceeds the configured header limit")
                    if chunk == b"fmt ":
                        if self.format is not None or self._saw_data:
                            raise WavFormatError("Duplicate or late WAV format chunk")
                        if size < 16:
                            raise WavFormatError("WAV format chunk is too short")
                        self._remaining = size
                        self._state = "format"
                    else:
                        self._remaining = size
                        self._state = "skip"
            elif self._state == "format":
                if len(self._buffer) < self._remaining:
                    break
                raw = self._take(self._remaining)
                tag, channels, rate, byte_rate, alignment, bits = struct.unpack(
                    "<HHIIHH", raw[:16]
                )
                if tag != 1 or bits != 16:
                    raise WavFormatError("Only uncompressed signed PCM16 WAV is supported")
                if channels < 1 or rate < 1:
                    raise WavFormatError("WAV sample rate and channels must be positive")
                if alignment != channels * 2 or byte_rate != rate * alignment:
                    raise WavFormatError("Inconsistent WAV byte rate or block alignment")
                try:
                    self.format = AudioFormat(
                        sample_rate=rate, channels=channels, sample_width=2
                    )
                except ValueError as exc:
                    raise WavFormatError(f"Invalid WAV format: {exc}") from exc
                self._state = "padding"
            elif self._state == "skip":
                count = min(self._remaining, len(self._buffer))
                del self._buffer[:count]
                self._remaining -= count
                if self._remaining:
                    break
                self._state = "padding"
            elif self._state == "padding":
                if len(self._buffer) < self._padding:
                    break
                del self._buffer[:self._padding]
                self._state = "chunk"
            elif self._state == "data":
                count = (
                    len(self._buffer)
                    if self._unknown_data
                    else min(self._remaining, len(self._buffer))
                )
                if count:
                    output.append(self._take(count))
                    self._audio_bytes += count
                if self._unknown_data:
                    break
                self._remaining -= count
                if self._remaining:
                    break
                self._state = "padding"
            else:  # Internal invariant, not user input.
                raise AssertionError(f"Unexpected parser state: {self._state}")
        return b"".join(output)

    def finish(self) -> None:
        """Confirm the transport ended at a valid sample/container boundary."""
        if self._finished:
            return
        if not self._saw_data or self.format is None:
            raise WavFormatError("Truncated WAV: no complete format and data header")
        if self._state != "data" and not (
            self._state == "chunk" and not self._buffer
        ):
            raise WavFormatError("Truncated WAV chunk")
        if self._state == "data" and not self._unknown_data and self._remaining:
            raise WavFormatError("Truncated WAV data")
        if self._riff_bytes is not None and self._received != self._riff_bytes:
            raise WavFormatError("Truncated RIFF container")
        if not self._audio_bytes:
            raise WavFormatError("WAV response contains no audio")
        if self._audio_bytes % (self.format.channels * 2):
            raise WavFormatError("WAV stream ended within an interleaved sample frame")
        self._finished = True

    def _take(self, count: int) -> bytes:
        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result
