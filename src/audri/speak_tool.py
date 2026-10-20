"""Agent-facing session manager with explicit ownership and shutdown."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from .tts.audio_egress import AudioEgress, LocalAudioEgress, NullAudioEgress
from .tts.chunker import ChunkerConfig
from .tts.driver import TTSDriver
from .tts.session import SessionConfig, SessionMetrics, SpeechSession
from .tts.types import SynthesisRequest


class SpeakTool:
    """Own live sessions sharing a model backend.

    ``start → delta* → end`` preserves the original Audri API. ``stop`` is
    idempotent and isolates cancellation. Use an async context manager or
    call ``aclose`` to clean up if a producer fails. Set close_driver=False
    when another owner manages the supplied backend's lifetime.
    """

    def __init__(
        self,
        driver: TTSDriver,
        chunker_config: ChunkerConfig | None = None,
        egress_factory: Callable[[], AudioEgress] | None = None,
        *,
        session_config: SessionConfig | None = None,
        max_sessions: int = 16,
        close_driver: bool = True,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.driver = driver
        self.chunker_config = chunker_config
        self.session_config = session_config
        if egress_factory is None:
            headless = os.getenv("AUDRI_HEADLESS", "").lower() in {"1", "true", "yes"}
            egress_factory = NullAudioEgress if headless else LocalAudioEgress
        self.egress_factory = egress_factory
        self.max_sessions = max_sessions
        self.close_driver = close_driver
        self.sessions: dict[str, SpeechSession] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> SpeakTool:
        if self._closed:
            raise RuntimeError("SpeakTool is closed")
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def start(
        self,
        session_id: str,
        voice: str = "default",
        language: str = "English",
        instruction: str | None = None,
        *,
        request: SynthesisRequest | None = None,
        egress: AudioEgress | None = None,
    ) -> SpeechSession:
        """Create a session, replacing an earlier generation of the same ID."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("SpeakTool is closed")
            previous = self.sessions.pop(session_id, None)
            if previous is not None:
                await previous.cancel()
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"maximum active sessions reached ({self.max_sessions})")
            session = SpeechSession(
                session_id=session_id,
                voice=voice,
                language=language,
                driver=self.driver,
                egress=egress if egress is not None else self.egress_factory(),
                chunker_config=self.chunker_config,
                instruction=instruction,
                config=self.session_config,
                request=request,
            )
            session.start()
            self.sessions[session_id] = session
            return session

    def _get_session(self, session_id: str) -> SpeechSession:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown speech session {session_id!r}") from None

    async def delta(self, session_id: str, text_delta: str, is_final: bool = False) -> None:
        await self._get_session(session_id).push_delta(text_delta, is_final=is_final)

    async def end(self, session_id: str) -> SessionMetrics:
        session = self._get_session(session_id)
        try:
            return await session.finish()
        finally:
            if self.sessions.get(session_id) is session:
                self.sessions.pop(session_id)

    async def stop(self, session_id: str, reason: str = "barge-in") -> SessionMetrics | None:
        del reason  # Reserved for application tracing; never logs speech text.
        async with self._lifecycle_lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                return None
            return await session.cancel()

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._closed = True
                sessions = list(self.sessions.values())
                self.sessions.clear()
                self._close_task = asyncio.create_task(self._shutdown(sessions))
                # Consume background exceptions if the caller is cancelled;
                # a later aclose still awaits and raises the stored exception.
                self._close_task.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
        await asyncio.shield(self._close_task)

    async def _shutdown(self, sessions: list[SpeechSession]) -> None:
        results = await asyncio.gather(
            *(session.cancel() for session in sessions), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        try:
            if self.close_driver:
                await self.driver.aclose()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise ExceptionGroup("speech manager shutdown failed", errors)
