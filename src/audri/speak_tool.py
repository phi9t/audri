"""SpeakTool — top-level async API for agent voice output.

Example usage::

    tool = SpeakTool(driver=HFQwen3TTSDriver())
    await tool.start("session-1", voice="default", language="English")
    async for token in llm_stream:
        await tool.delta("session-1", token)
    await tool.end("session-1")
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .tts.audio_egress import LocalAudioEgress
from .tts.chunker import ChunkerConfig
from .tts.driver import TTSDriver
from .tts.session import SpeechSession

logger = logging.getLogger(__name__)


class SpeakTool:
    """Manages speech sessions for streaming LLM → TTS → audio pipelines."""

    def __init__(
        self,
        driver: TTSDriver,
        chunker_config: Optional[ChunkerConfig] = None,
        egress_factory: Optional[Callable[[], LocalAudioEgress]] = None,
    ) -> None:
        self.driver = driver
        self.chunker_config = chunker_config
        self.egress_factory = egress_factory or LocalAudioEgress
        self.sessions: dict[str, SpeechSession] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        session_id: str,
        voice: str = "default",
        language: str = "English",
        instruction: Optional[str] = None,
    ) -> None:
        """Create a new speech session and launch its background tasks."""
        if session_id in self.sessions:
            logger.warning("Session %r already exists — stopping old one first", session_id)
            await self.stop(session_id, reason="replaced")

        egress = self.egress_factory()
        session = SpeechSession(
            session_id=session_id,
            voice=voice,
            language=language,
            driver=self.driver,
            egress=egress,
            chunker_config=self.chunker_config,
            instruction=instruction,
        )
        session.start()
        self.sessions[session_id] = session
        logger.info("Session %r started (voice=%s, lang=%s)", session_id, voice, language)

    async def delta(
        self,
        session_id: str,
        text_delta: str,
        is_final: bool = False,
    ) -> None:
        """Forward a token delta to the named session."""
        session = self._get_session(session_id)
        if session is None:
            return
        await session.push_delta(text_delta, is_final=is_final)

    async def stop(self, session_id: str, reason: str = "barge-in") -> None:
        """Cancel the session (barge-in or external stop)."""
        session = self.sessions.pop(session_id, None)
        if session is None:
            logger.debug("stop: session %r not found", session_id)
            return
        logger.info("Session %r stopping: %s", session_id, reason)
        await session.cancel()

    async def end(self, session_id: str) -> None:
        """Flush remaining text, await pipeline completion, and clean up."""
        session = self._get_session(session_id)
        if session is None:
            return
        await session.push_delta("", is_final=True)
        await session.wait()
        self.sessions.pop(session_id, None)
        logger.info("Session %r ended", session_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> Optional[SpeechSession]:
        session = self.sessions.get(session_id)
        if session is None:
            logger.warning("Session %r not found", session_id)
        return session
