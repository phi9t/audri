"""Speech Session — asyncio state machine.

Wires together TextChunker → TTSDriver → LocalAudioEgress into a live
streaming pipeline.  Each SpeechSession is created per-utterance and torn
down when the LLM stream ends or a barge-in cancel arrives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .audio_egress import LocalAudioEgress
from .chunker import ChunkerConfig, TextChunker
from .driver import TTSDriver

logger = logging.getLogger(__name__)

# How often (seconds) the commit loop checks for chunker timeout
_TIMEOUT_POLL_INTERVAL = 0.05


class SessionState(Enum):
    IDLE = auto()
    BUFFERING = auto()
    SPEAKING = auto()
    CANCELLING = auto()
    DONE = auto()


@dataclass
class SpeechSession:
    session_id: str
    voice: str
    language: str
    driver: TTSDriver
    egress: LocalAudioEgress
    chunker_config: Optional[ChunkerConfig] = None
    instruction: Optional[str] = None

    # Set after start()
    state: SessionState = field(default=SessionState.IDLE, init=False)
    chunker: TextChunker = field(init=False)
    _commit_queue: asyncio.Queue = field(init=False)
    _audio_queue: asyncio.Queue = field(init=False)
    _commit_task: Optional[asyncio.Task] = field(default=None, init=False)
    _tts_task: Optional[asyncio.Task] = field(default=None, init=False)
    _egress_task: Optional[asyncio.Task] = field(default=None, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.chunker = TextChunker(self.chunker_config)
        self._commit_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch background tasks. Call once before pushing deltas."""
        self.state = SessionState.BUFFERING
        self._commit_task = asyncio.create_task(
            self._commit_loop(), name=f"commit-{self.session_id}"
        )
        self._tts_task = asyncio.create_task(
            self._tts_loop(), name=f"tts-{self.session_id}"
        )
        self._egress_task = asyncio.create_task(
            self.egress.run(self._audio_queue), name=f"egress-{self.session_id}"
        )

    async def wait(self) -> None:
        """Await full pipeline teardown."""
        tasks = [t for t in [self._commit_task, self._tts_task, self._egress_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.state = SessionState.DONE

    # ------------------------------------------------------------------
    # Public delta / flush API
    # ------------------------------------------------------------------

    async def push_delta(self, text: str, is_final: bool = False) -> None:
        """Feed a token delta into the chunker.

        If a chunk boundary is found, the chunk is enqueued for TTS.
        On *is_final*, flushes any remaining text and sends the EOS sentinel.
        """
        if self.state in (SessionState.CANCELLING, SessionState.DONE):
            return

        if text:
            chunk = self.chunker.push(text)
            if chunk:
                self.state = SessionState.SPEAKING
                await self._commit_queue.put(chunk)

        if is_final:
            remaining = self.chunker.flush()
            if remaining:
                await self._commit_queue.put(remaining)
            await self._commit_queue.put(None)  # EOS sentinel
            self._finalized = True
            if self._commit_task and not self._commit_task.done():
                self._commit_task.cancel()

    async def cancel(self) -> None:
        """Barge-in cancel: stop TTS and egress immediately."""
        self.state = SessionState.CANCELLING

        # Cancel TTS task (currently synthesising)
        self.driver.cancel()
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()

        # Cancel commit loop
        if self._commit_task and not self._commit_task.done():
            self._commit_task.cancel()

        # Signal egress to shut down
        await self._audio_queue.put(None)

        # Await all tasks
        await self.wait()

    # ------------------------------------------------------------------
    # Internal tasks
    # ------------------------------------------------------------------

    async def _commit_loop(self) -> None:
        """Poll chunker timeout and enqueue committed chunks."""
        try:
            while not self._finalized:
                await asyncio.sleep(_TIMEOUT_POLL_INTERVAL)
                chunk = self.chunker.check_timeout()
                if chunk:
                    logger.debug(
                        "[%s] timeout-commit: %r", self.session_id, chunk[:40]
                    )
                    await self._commit_queue.put(chunk)
        except asyncio.CancelledError:
            pass

    async def _tts_loop(self) -> None:
        """Dequeue committed chunks, synthesise, and forward PCM to egress."""
        try:
            while True:
                chunk = await self._commit_queue.get()
                if chunk is None:
                    break  # EOS
                t_synth_start = time.monotonic()
                logger.debug(
                    "[%s] synth start: %r", self.session_id, chunk[:60]
                )
                frame_count = 0
                async for frame in self.driver.synth(chunk, self.voice, self.language, self.instruction):
                    await self._audio_queue.put(frame)
                    frame_count += 1
                t_synth_end = time.monotonic()
                logger.debug(
                    "[%s] synth done: %d frames in %.0f ms",
                    self.session_id,
                    frame_count,
                    (t_synth_end - t_synth_start) * 1000,
                )
        except asyncio.CancelledError:
            pass
        finally:
            # Signal egress EOS
            await self._audio_queue.put(None)
