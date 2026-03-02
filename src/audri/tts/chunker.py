"""Text Committer / Chunker — pure Python, no deps.

Splits a streaming token delta sequence into committed chunks suitable for
TTS synthesis. Boundaries are chosen at sentence/clause punctuation; a
timeout fallback fires when no boundary arrives within `boundary_timeout_ms`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# Punctuation characters treated as strong boundaries (prefer over spaces)
_STRONG_BOUNDARY = frozenset(".!?;:\n")


@dataclass
class ChunkerConfig:
    min_chars_to_start: int = 48
    """Don't commit until at least this many chars have accumulated."""
    max_chars_per_chunk: int = 180
    """Force a word-boundary commit when pending text exceeds this."""
    lookahead_chars: int = 40
    """Scan this many chars ahead of min_chars_to_start to find a boundary."""
    boundary_timeout_ms: float = 220
    """Fire a word-boundary commit if no delta arrives within this window."""


class TextChunker:
    """Accumulates streaming text deltas and commits chunks at natural boundaries.

    Usage::

        chunker = TextChunker()
        for delta in stream:
            chunk = chunker.push(delta)
            if chunk:
                tts_queue.put(chunk)
        final = chunker.flush()
        if final:
            tts_queue.put(final)
    """

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self.config = config or ChunkerConfig()
        self.pending_text: str = ""
        self.last_delta_ts: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, delta: str) -> Optional[str]:
        """Append *delta* and return a committed chunk if a boundary is found."""
        self.pending_text += delta
        self.last_delta_ts = time.monotonic()
        return self._try_commit()

    def flush(self) -> Optional[str]:
        """Force-commit all remaining text (call on speak_end / EOS)."""
        text = self.pending_text.strip()
        self.pending_text = ""
        return text if text else None

    def check_timeout(self) -> Optional[str]:
        """Call periodically; returns a committed chunk if the timeout fired."""
        if self.timeout_fired and self.pending_text.strip():
            return self._commit_word_boundary(len(self.pending_text))
        return None

    @property
    def timeout_fired(self) -> bool:
        elapsed = time.monotonic() - self.last_delta_ts
        return elapsed > self.config.boundary_timeout_ms / 1000.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_commit(self) -> Optional[str]:
        n = len(self.pending_text)
        cfg = self.config

        # Not enough text yet — but check hard max first
        if n >= cfg.max_chars_per_chunk:
            return self._commit_word_boundary(n)

        if n < cfg.min_chars_to_start:
            return None

        # Enough text — search for a boundary in the lookahead window
        search_end = min(n, cfg.min_chars_to_start + cfg.lookahead_chars)
        pos = self._best_boundary_before(search_end)
        if pos is not None:
            return self._commit_at(pos + 1)  # include the boundary char

        return None

    def _choose_commit(self) -> Optional[int]:
        """Return commit position (exclusive) or None. Pure logic, no side-effects."""
        n = len(self.pending_text)
        cfg = self.config

        if n >= cfg.max_chars_per_chunk:
            pos = self._best_word_boundary_before(n)
            return pos if pos is not None else n

        if n < cfg.min_chars_to_start:
            return None

        search_end = min(n, cfg.min_chars_to_start + cfg.lookahead_chars)
        pos = self._best_boundary_before(search_end)
        if pos is not None:
            return pos + 1

        return None

    def _best_boundary_before(self, end: int) -> Optional[int]:
        """Scan left from *end* for a strong punctuation boundary."""
        text = self.pending_text
        for i in range(end - 1, -1, -1):
            if text[i] in _STRONG_BOUNDARY:
                return i
        # Fall back to a space if no punctuation found
        return self._best_word_boundary_before(end)

    def _best_word_boundary_before(self, end: int) -> Optional[int]:
        """Scan left from *end* for a space (word boundary)."""
        text = self.pending_text
        for i in range(end - 1, -1, -1):
            if text[i] == " ":
                return i
        return None

    def _commit_word_boundary(self, search_end: int) -> Optional[str]:
        pos = self._best_word_boundary_before(search_end)
        if pos is None:
            # No space found — hard split at search_end
            pos = search_end
        return self._commit_at(pos + 1)

    def _commit_at(self, end: int) -> Optional[str]:
        chunk = self.pending_text[:end].strip()
        self.pending_text = self.pending_text[end:].lstrip()
        return chunk if chunk else None
