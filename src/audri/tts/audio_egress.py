"""Backpressure-aware sinks for typed PCM audio.

Every sink owns one output stream and accepts one unchanging audio format.
``close`` drains/finalizes a successful stream; ``abort`` discards pending
audio. Optional playback dependencies are imported only on first write.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from audri.audio import AudioFormat, AudioFrame


async def _io_call(function: Callable[..., Any], *args: Any) -> Any:
    """Finish started I/O before allowing teardown; threads cannot be killed."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A second cancellation must not detach the thread from its file or
        # device either. Teardown can begin once the operation is finished.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


class AudioEgress(ABC):
    """One-use output stream. Implement ``write`` and optionally teardown."""

    def __init__(self, *, format: AudioFormat | None = None) -> None:
        self.format = format
        self.closed = False
        self.aborted = False

    def _accept(self, frame: AudioFrame) -> None:
        if self.closed:
            raise RuntimeError("audio sink is closed")
        if not isinstance(frame, AudioFrame):
            raise TypeError("audio sinks require AudioFrame values")
        if self.format is None:
            self.format = frame.format
        elif self.format != frame.format:
            raise ValueError("audio format changed within an output stream")

    @abstractmethod
    async def write(self, frame: AudioFrame) -> None:
        """Accept a frame, waiting for output capacity when necessary."""

    async def close(self) -> None:
        self.closed = True

    async def abort(self) -> None:
        self.aborted = True
        self.closed = True

    async def run(self, pcm_queue: asyncio.Queue[AudioFrame | None]) -> None:
        """Consume a typed frame queue; ``None`` means successful EOF."""
        try:
            while (frame := await pcm_queue.get()) is not None:
                await self.write(frame)
            await self.close()
        except BaseException:
            await self.abort()
            raise


class NullAudioEgress(AudioEgress):
    """Discard audio while retaining lightweight output counters."""

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0
        self.bytes_written = 0
        self.audio_seconds = 0.0

    async def write(self, frame: AudioFrame) -> None:
        self._accept(frame)
        self.frames += 1
        self.bytes_written += len(frame.pcm)
        self.audio_seconds += frame.duration_s


class CallbackAudioEgress(AudioEgress):
    """Forward frames to a sync or async callback in order.

    Awaited completion is the backpressure boundary. Keep synchronous
    callbacks short; use an async callback for network I/O.
    """

    def __init__(
        self,
        callback: Callable[[AudioFrame], Any],
        *,
        on_close: Callable[[], Any] | None = None,
        on_abort: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.callback = callback
        self.on_close = on_close
        self.on_abort = on_abort

    @staticmethod
    async def _call(callback: Callable[..., Any], *args: Any) -> None:
        result = callback(*args)
        if inspect.isawaitable(result):
            await result

    async def write(self, frame: AudioFrame) -> None:
        self._accept(frame)
        await self._call(self.callback, frame)

    async def close(self) -> None:
        if self.closed:
            return
        if self.on_close is not None:
            await self._call(self.on_close)
        self.closed = True

    async def abort(self) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.closed = True
        if self.on_abort is not None:
            await self._call(self.on_abort)


class _NotifyingFrameQueue(asyncio.Queue[AudioFrame]):
    """Keep queue compatibility while notifying readers without consuming data."""

    def __init__(self, maxsize: int, available: asyncio.Event) -> None:
        super().__init__(maxsize=maxsize)
        self._available = available

    def put_nowait(self, item: AudioFrame) -> None:
        super().put_nowait(item)
        self._available.set()


class QueueAudioEgress(AudioEgress, AsyncIterator[AudioFrame]):
    """Bounded frame iterator with out-of-band EOF and cancellation.

    Close allows queued audio to drain. Abort clears it immediately. EOF
    never needs a spare slot, so stalled consumers cannot block shutdown.
    """

    def __init__(self, maxsize: int = 32) -> None:
        super().__init__()
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self._available = asyncio.Event()
        self.queue: asyncio.Queue[AudioFrame] = _NotifyingFrameQueue(maxsize, self._available)
        self._ended = asyncio.Event()

    async def write(self, frame: AudioFrame) -> None:
        self._accept(frame)
        put = asyncio.create_task(self.queue.put(frame))
        ended = asyncio.create_task(self._ended.wait())
        try:
            await asyncio.wait((put, ended), return_when=asyncio.FIRST_COMPLETED)
            if self.closed:
                if self.aborted:
                    self._discard()
                raise RuntimeError("audio sink closed while waiting for capacity")
            await put
        finally:
            for task in (put, ended):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, ended, return_exceptions=True)

    def __aiter__(self) -> QueueAudioEgress:
        return self

    async def __anext__(self) -> AudioFrame:
        while True:
            if self.aborted:
                raise StopAsyncIteration
            try:
                # Dequeue and return in this reader's task without an await
                # between them. A cancelled availability waiter has never
                # consumed a frame that a subsequent reader would need.
                return self.queue.get_nowait()
            except asyncio.QueueEmpty:
                if self.closed:
                    raise StopAsyncIteration
                # No task can enqueue between the empty check and clear:
                # both run synchronously on the same event loop.
                self._available.clear()
            await self._available.wait()

    def _discard(self) -> None:
        while not self.queue.empty():
            self.queue.get_nowait()

    async def close(self) -> None:
        self.closed = True
        self._ended.set()
        self._available.set()

    async def abort(self) -> None:
        self.aborted = True
        self.closed = True
        self._discard()
        self._ended.set()
        self._available.set()


class WaveAudioEgress(AudioEgress):
    """Write native-rate WAV incrementally, publishing only complete files.

    Close publishes a correct finite RIFF header; abort removes temporary
    output. Existing destinations are protected unless overwrite=True.
    Empty output uses ``format`` or 24 kHz mono PCM16.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        format: AudioFormat | None = None,
    ) -> None:
        super().__init__(format=format)
        self.path = Path(path)
        self.overwrite = overwrite
        self._temporary: Path | None = None
        self._wave: wave.Wave_write | None = None
        self._file: Any = None
        self._lock = asyncio.Lock()

    def _open(self) -> None:
        if self._wave is not None:
            return
        if not self.overwrite and self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".part", dir=self.path.parent
        )
        self._temporary = Path(temporary)
        self._file = os.fdopen(fd, "wb")
        self._wave = wave.open(self._file, "wb")
        audio_format = self.format or AudioFormat(24_000)
        self.format = audio_format
        self._wave.setnchannels(audio_format.channels)
        self._wave.setsampwidth(audio_format.sample_width)
        self._wave.setframerate(audio_format.sample_rate)

    def _write(self, pcm: bytes) -> None:
        self._open()
        assert self._wave is not None
        self._wave.writeframesraw(pcm)

    async def write(self, frame: AudioFrame) -> None:
        async with self._lock:
            self._accept(frame)
            await _io_call(self._write, frame.pcm)

    def _finish(self) -> None:
        self._open()
        assert self._wave is not None and self._temporary is not None
        self._wave.close()
        self._wave = None
        self._file.close()
        self._file = None
        if self.overwrite:
            os.replace(self._temporary, self.path)
        else:
            os.link(self._temporary, self.path)
            self._temporary.unlink()
        self._temporary = None

    async def close(self) -> None:
        async with self._lock:
            if self.closed:
                return
            try:
                await _io_call(self._finish)
            except BaseException:
                await _io_call(self._discard_file)
                self.closed = True
                self.aborted = True
                raise
            self.closed = True

    def _discard_file(self) -> None:
        try:
            if self._wave is not None:
                self._wave.close()
        finally:
            self._wave = None
            if self._file is not None:
                self._file.close()
                self._file = None
            if self._temporary is not None:
                self._temporary.unlink(missing_ok=True)
                self._temporary = None

    async def abort(self) -> None:
        async with self._lock:
            if self.closed:
                return
            self.closed = True
            self.aborted = True
            await _io_call(self._discard_file)


class LocalAudioEgress(AudioEgress):
    """Play PCM through a lazily imported sounddevice.RawOutputStream.

    Device writes run serially in a worker thread. Abort discards the device
    queue instead of draining it, defining the local barge-in boundary.
    """

    def __init__(self, *, device: Any = None, latency: str | float = "low") -> None:
        super().__init__()
        self.device = device
        self.latency = latency
        self._stream: Any = None
        self._lock = asyncio.Lock()

    def _write(self, frame: AudioFrame) -> None:
        if self._stream is None:
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError(
                    "Playback requires sounddevice and PortAudio; use "
                    "WaveAudioEgress or NullAudioEgress on a headless server."
                ) from exc
            self._stream = sounddevice.RawOutputStream(
                samplerate=frame.format.sample_rate,
                channels=frame.format.channels,
                dtype="int16",
                device=self.device,
                latency=self.latency,
                blocksize=0,
            )
            self._stream.start()
        self._stream.write(frame.pcm)

    async def write(self, frame: AudioFrame) -> None:
        async with self._lock:
            self._accept(frame)
            await _io_call(self._write, frame)

    def _finish(self, abort: bool) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            if abort:
                stream.abort()
            else:
                stream.stop()
        finally:
            stream.close()

    async def close(self) -> None:
        async with self._lock:
            if self.closed:
                return
            await _io_call(self._finish, False)
            self.closed = True

    async def abort(self) -> None:
        async with self._lock:
            if self.aborted:
                return
            self.aborted = True
            self.closed = True
            await _io_call(self._finish, True)
