"""A bounded, supervised text → synthesis → audio session.

One task owns the chunker. A TaskGroup owns the pipeline. Cancellation is
per session, and EOF travels in order with text; failure and cancellation
wake producers even when the input queue is full.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from audri.audio import AudioFormat, AudioFrame

from .audio_egress import AudioEgress
from .chunker import ChunkerConfig, TextChunker
from .driver import TTSDriver
from .types import CancellationToken, SynthesisRequest


class SessionState(str, Enum):
    IDLE = "idle"
    BUFFERING = "buffering"
    SPEAKING = "speaking"
    CANCELLING = "cancelling"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SessionClosedError(RuntimeError):
    """Input arrived after finalization or termination."""


class SessionLimitError(ValueError):
    """Input or generated output exceeded a configured session limit."""


@dataclass(frozen=True)
class SessionConfig:
    delta_queue_size: int = 32
    text_queue_size: int = 4
    audio_queue_size: int = 32
    max_delta_chars: int = 4096
    max_text_chars: int = 100_000
    max_frame_bytes: int = 262_144
    max_audio_seconds: float = 3600.0
    synthesis_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "delta_queue_size", "text_queue_size", "audio_queue_size",
            "max_delta_chars", "max_text_chars", "max_frame_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_audio_seconds", "synthesis_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class SessionMetrics:
    session_id: str
    state: str = SessionState.IDLE.value
    elapsed_seconds: float = 0.0
    first_audio_ms: float | None = None
    audio_seconds: float = 0.0
    synthesis_seconds: float = 0.0
    frames: int = 0
    chunks: int = 0
    input_chars: int = 0
    delta_queue_highwater: int = 0
    text_queue_highwater: int = 0
    audio_queue_highwater: int = 0
    error: str | None = None

    @property
    def rtf(self) -> float | None:
        """Generator wait time divided by successfully emitted audio time."""
        if self.audio_seconds <= 0:
            return None
        return self.synthesis_seconds / self.audio_seconds

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rtf"] = self.rtf
        return result


class SpeechSession:
    """One utterance with independent cancellation and bounded buffering.

    An optional ``request`` is a synthesis template: text is replaced for
    each committed chunk; references, seed, and other controls pass through.
    ``finish`` finalizes input and cancels the session if its caller is
    cancelled. ``wait`` does not cancel the session when its caller is.
    """

    def __init__(
        self,
        session_id: str,
        voice: str,
        language: str,
        driver: TTSDriver,
        egress: AudioEgress,
        chunker_config: ChunkerConfig | None = None,
        instruction: str | None = None,
        *,
        config: SessionConfig | None = None,
        request: SynthesisRequest | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        self.session_id = session_id
        self.voice = voice
        self.language = language
        self.instruction = instruction
        self.driver = driver
        self.egress = egress
        self.config = config or SessionConfig()
        self.request = request
        self.chunker = TextChunker(chunker_config)
        self.state = SessionState.IDLE
        self.metrics = SessionMetrics(session_id)
        self.cancel_token = CancellationToken()
        self._delta_queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue(
            maxsize=self.config.delta_queue_size
        )
        self._commit_queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=self.config.text_queue_size
        )
        self._audio_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(
            maxsize=self.config.audio_queue_size
        )
        self._input_lock = asyncio.Lock()
        self._finished = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._finalized = False
        self._cancel_requested = False
        self._started_at = 0.0
        self._produced_seconds = 0.0
        self._format: AudioFormat | None = None

    def _set_state(self, state: SessionState) -> None:
        self.state = state
        self.metrics.state = state.value

    def start(self) -> None:
        if self.state is not SessionState.IDLE:
            raise RuntimeError("a speech session can only be started once")
        self._started_at = time.monotonic()
        self._set_state(SessionState.BUFFERING)
        self._runner = asyncio.create_task(self._run(), name=f"speech-{self.session_id}")

    def _raise_closed(self) -> None:
        if self._error is not None:
            raise self._error
        raise SessionClosedError(f"session {self.session_id!r} is {self.state.value}")

    async def _put_delta(self, value: tuple[str, bool]) -> None:
        async def accept() -> None:
            await self._delta_queue.put(value)
            # These updates cannot be interrupted after queue acceptance.
            # In particular a caller cancelled at this point cannot enqueue
            # another final marker while the first is already in flight.
            self.metrics.input_chars += len(value[0])
            self._finalized = self._finalized or value[1]
            self.metrics.delta_queue_highwater = max(
                self.metrics.delta_queue_highwater, self._delta_queue.qsize()
            )

        put = asyncio.create_task(accept())
        finished = asyncio.create_task(self._finished.wait())
        try:
            await asyncio.wait((put, finished), return_when=asyncio.FIRST_COMPLETED)
            if self._cancel_requested or (
                self._finished.is_set() and self.state is not SessionState.DONE
            ):
                self._raise_closed()
            await put
        finally:
            for task in (put, finished):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, finished, return_exceptions=True)

    async def push_delta(self, text: str, is_final: bool = False) -> None:
        if not isinstance(text, str):
            raise TypeError("text delta must be a string")
        if not isinstance(is_final, bool):
            raise TypeError("is_final must be a bool")
        if len(text) > self.config.max_delta_chars:
            raise SessionLimitError(
                f"delta exceeds max_delta_chars={self.config.max_delta_chars}"
            )
        async with self._input_lock:
            if self._runner is None:
                raise RuntimeError("start the speech session before sending input")
            if self._finalized and is_final and not text and not self._cancel_requested:
                if self._error is not None:
                    raise self._error
                return
            if self._finished.is_set() or self._cancel_requested:
                self._raise_closed()
            if self._finalized:
                raise SessionClosedError("input is already finalized")
            if self.metrics.input_chars + len(text) > self.config.max_text_chars:
                raise SessionLimitError(
                    f"session exceeds max_text_chars={self.config.max_text_chars}"
                )
            if text or is_final:
                await self._put_delta((text, is_final))

    async def finish(self) -> SessionMetrics:
        try:
            if not self._finished.is_set():
                await self.push_delta("", is_final=True)
            return await self.wait()
        except asyncio.CancelledError:
            await self.cancel()
            raise

    async def wait(self) -> SessionMetrics:
        if self._runner is None:
            raise RuntimeError("start the speech session before waiting")
        if not self._finished.is_set():
            await asyncio.shield(self._runner)
        if self._error is not None:
            raise self._error
        return self.metrics

    async def cancel(self) -> SessionMetrics:
        """Discard this session's buffered audio before returning.

        No shared driver-global cancellation hook is called. A provider may
        continue remote GPU work after its transport has been closed.
        """
        if self._runner is None:
            self._cancel_requested = True
            self.cancel_token.cancel()
            self._set_state(SessionState.CANCELLED)
            await self.egress.abort()
            self._finished.set()
            return self.metrics
        if not self._finished.is_set() and not self._cancel_requested:
            self._cancel_requested = True
            self._set_state(SessionState.CANCELLING)
            self.cancel_token.cancel()
            self._runner.cancel()
        try:
            await asyncio.shield(self._runner)
        except asyncio.CancelledError:
            # Cancelling a task before its first instruction skips its finally.
            if self._runner.cancelled():
                await self._complete_early_cancel()
            else:
                raise
        return self.metrics

    async def _complete_early_cancel(self) -> None:
        if self._finished.is_set():
            return
        await self.egress.abort()
        self._drain_queues()
        self._set_state(SessionState.CANCELLED)
        self.metrics.elapsed_seconds = time.monotonic() - self._started_at
        self._finished.set()

    @staticmethod
    def _first_error(error: BaseException) -> BaseException:
        while isinstance(error, BaseExceptionGroup):
            error = error.exceptions[0]
        return error

    def _drain_queues(self) -> None:
        for queue in (self._delta_queue, self._commit_queue, self._audio_queue):
            while not queue.empty():
                queue.get_nowait()

    async def _run(self) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._stage(self._chunk_loop), name=f"text-{self.session_id}")
                group.create_task(self._stage(self._tts_loop), name=f"tts-{self.session_id}")
                group.create_task(self._stage(self._egress_loop), name=f"output-{self.session_id}")
            self._set_state(SessionState.DONE)
        except asyncio.CancelledError:
            self.cancel_token.cancel()
            self._set_state(SessionState.CANCELLED)
        except BaseException as exc:
            self.cancel_token.cancel()
            self._error = self._first_error(exc)
            self.metrics.error = f"{type(self._error).__name__}: {self._error}"
            self._set_state(SessionState.FAILED)
        finally:
            if self.state is not SessionState.DONE:
                try:
                    await self.egress.abort()
                except Exception as exc:
                    if self._error is None:
                        self._error = exc
                        self.metrics.error = f"{type(exc).__name__}: {exc}"
                        self._set_state(SessionState.FAILED)
            self._drain_queues()
            self.metrics.elapsed_seconds = time.monotonic() - self._started_at
            self._finished.set()

    async def _stage(self, factory: Callable[[], Awaitable[None]]) -> None:
        try:
            await factory()
        except asyncio.CancelledError:
            # TaskGroup treats a child CancelledError as a normal exit. A
            # backend that independently cancels itself must not leave its
            # siblings waiting forever for EOF that will never arrive.
            if (not self._cancel_requested and self._runner is not None
                    and not self._runner.cancelling()):
                raise RuntimeError("a speech pipeline stage cancelled unexpectedly") from None
            raise

    async def _commit(self, chunk: str) -> None:
        await self._commit_queue.put(chunk)
        self.metrics.text_queue_highwater = max(
            self.metrics.text_queue_highwater, self._commit_queue.qsize()
        )

    async def _chunk_loop(self) -> None:
        timeout = max(0.005, min(0.05, self.chunker.config.boundary_timeout_ms / 1000))
        while True:
            try:
                text, final = await asyncio.wait_for(self._delta_queue.get(), timeout)
            except TimeoutError:
                chunk = self.chunker.check_timeout()
                if chunk is not None:
                    await self._commit(chunk)
                continue
            if text:
                chunk = self.chunker.push(text)
                if chunk is not None:
                    await self._commit(chunk)
                while (chunk := self.chunker.pop_ready()) is not None:
                    await self._commit(chunk)
            if final:
                while (chunk := self.chunker.flush()) is not None:
                    await self._commit(chunk)
                await self._commit_queue.put(None)
                return

    def _request_for(self, text: str) -> SynthesisRequest:
        if self.request is not None:
            return replace(self.request, text=text)
        return SynthesisRequest(
            text=text, voice=self.voice, language=self.language, instruction=self.instruction
        )

    async def _tts_loop(self) -> None:
        while (chunk := await self._commit_queue.get()) is not None:
            self.cancel_token.raise_if_cancelled()
            request = self._request_for(chunk)
            async with asyncio.timeout(self.config.synthesis_timeout_s):
                async with aclosing(self.driver.stream(request, cancel=self.cancel_token)) as stream:
                    while True:
                        started = time.monotonic()
                        try:
                            frame = await anext(stream)
                        except StopAsyncIteration:
                            break
                        finally:
                            self.metrics.synthesis_seconds += time.monotonic() - started
                        self.cancel_token.raise_if_cancelled()
                        if not isinstance(frame, AudioFrame):
                            raise TypeError("TTS drivers must yield AudioFrame values")
                        if len(frame.pcm) > self.config.max_frame_bytes:
                            raise SessionLimitError("driver frame exceeds max_frame_bytes")
                        if self._format is None:
                            self._format = frame.format
                        elif self._format != frame.format:
                            raise ValueError("driver changed audio format within a session")
                        self._produced_seconds += frame.duration_s
                        if self._produced_seconds > self.config.max_audio_seconds:
                            raise SessionLimitError("generated audio exceeds max_audio_seconds")
                        if self.metrics.first_audio_ms is None:
                            self.metrics.first_audio_ms = (
                                time.monotonic() - self._started_at
                            ) * 1000.0
                        self._set_state(SessionState.SPEAKING)
                        await self._audio_queue.put(frame)
                        self.metrics.audio_queue_highwater = max(
                            self.metrics.audio_queue_highwater, self._audio_queue.qsize()
                        )
            self.metrics.chunks += 1
        await self._audio_queue.put(None)

    async def _egress_loop(self) -> None:
        while (frame := await self._audio_queue.get()) is not None:
            self.cancel_token.raise_if_cancelled()
            await self.egress.write(frame)
            self.metrics.frames += 1
            self.metrics.audio_seconds += frame.duration_s
        await self.egress.close()
