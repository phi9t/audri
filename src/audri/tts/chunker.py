"""Bounded, multilingual text commits for an incremental token stream."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

_BOUNDARIES = frozenset(".!?;:\n。！？；：")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e"})


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    min_chars_to_start: int = 48
    max_chars_per_chunk: int = 180
    lookahead_chars: int = 40
    boundary_timeout_ms: float = 220

    def __post_init__(self) -> None:
        for name in ("min_chars_to_start", "max_chars_per_chunk", "lookahead_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "lookahead_chars" else 1):
                raise ValueError(f"{name} must be a valid nonnegative integer")
        if self.min_chars_to_start > self.max_chars_per_chunk:
            raise ValueError("min_chars_to_start cannot exceed max_chars_per_chunk")
        if not math.isfinite(self.boundary_timeout_ms) or self.boundary_timeout_ms <= 0:
            raise ValueError("boundary_timeout_ms must be finite and positive")


class TextChunker:
    """Keep every emitted chunk within the hard size bound, including final flush.

    Sentence punctuation in Latin and CJK text is preferred. Decimal points and
    common abbreviations are not sentence boundaries. Bracketed Fish controls
    and <|...|> tokens are kept whole; an overlong control is rejected instead
    of being split into corrupted model input. This is a text boundary policy,
    not a Markdown renderer or a full linguistic sentence tokenizer.

    A single large delta can create multiple ready chunks. Call pop_ready until
    it returns None. On EOF, call flush until it returns None.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self.pending_text = ""
        self.last_delta_ts = time.monotonic()

    def push(self, delta: str) -> str | None:
        if not isinstance(delta, str):
            raise TypeError("text delta must be a string")
        self.pending_text += delta
        if delta:
            self.last_delta_ts = time.monotonic()
        return self.pop_ready()

    def pop_ready(self) -> str | None:
        self.pending_text = self.pending_text.lstrip()
        text = self.pending_text
        n = len(text)
        cfg = self.config
        if not n:
            return None
        limit = min(n, cfg.max_chars_per_chunk)
        safe_limit = self._safe_end(limit)
        if n >= cfg.min_chars_to_start:
            search_end = min(safe_limit, cfg.min_chars_to_start + cfg.lookahead_chars)
            for end in range(search_end, cfg.min_chars_to_start - 1, -1):
                if self._boundary(end) and self._safe_end(end) == end:
                    return self._commit(end)
        if n >= cfg.max_chars_per_chunk:
            if safe_limit == 0:
                # A control of exactly max length may still be awaiting its
                # closing delimiter. A larger one can never meet the bound.
                if n > cfg.max_chars_per_chunk:
                    raise ValueError("a control token exceeds max_chars_per_chunk")
                return None
            return self._commit(self._word_end(safe_limit))
        return None

    def flush(self) -> str | None:
        self.pending_text = self.pending_text.lstrip()
        if not self.pending_text:
            return None
        limit = min(len(self.pending_text), self.config.max_chars_per_chunk)
        safe = self._safe_end(limit, final=True)
        if safe == 0:
            raise ValueError("a control token exceeds max_chars_per_chunk")
        if len(self.pending_text) <= self.config.max_chars_per_chunk:
            return self._commit(safe)
        return self._commit(self._word_end(safe))

    def check_timeout(self) -> str | None:
        if not self.timeout_fired:
            return None
        self.pending_text = self.pending_text.lstrip()
        if not self.pending_text:
            return None
        limit = self._safe_end(min(len(self.pending_text), self.config.max_chars_per_chunk))
        if not limit:
            return None
        return self._commit(self._word_end(limit))

    @property
    def timeout_fired(self) -> bool:
        return time.monotonic() - self.last_delta_ts >= self.config.boundary_timeout_ms / 1000

    def _boundary(self, end: int) -> bool:
        text = self.pending_text
        char = text[end - 1]
        if char not in _BOUNDARIES:
            return False
        if char in ".:":
            before = text[end - 2] if end > 1 else ""
            after = text[end] if end < len(text) else ""
            if before.isdigit() and (after.isdigit() or not after):
                return False
        if char == ".":
            word = re.search(r"([A-Za-z.]+)\.$", text[:end])
            if word and (word[1].lower() in _ABBREVIATIONS or len(word[1]) == 1):
                return False
        return True

    def _safe_end(self, limit: int, final: bool = False) -> int:
        """Return an endpoint that does not bisect a model control span."""
        text = self.pending_text
        i = 0
        while i < limit:
            if text[i] == "<" and i + 1 == len(text) and not final:
                # The next token delta may turn this into an <|...|> control.
                return i
            if text[i] == "[":
                closing = text.find("]", i + 1)
                stop = closing + 1 if closing >= 0 else len(text) + 1
            elif text.startswith("<|", i):
                closing = text.find("|>", i + 2)
                stop = closing + 2 if closing >= 0 else len(text) + 1
            else:
                i += 1
                continue
            if stop > limit:
                if final and closing < 0 and len(text) <= limit:
                    # Unclosed ordinary prose brackets at EOF remain literal.
                    return limit
                return i
            i = stop
        return limit

    def _word_end(self, limit: int) -> int:
        # At the actual end there is nothing to split or save for later.
        if limit == len(self.pending_text):
            return limit
        for end in range(limit, 0, -1):
            if self.pending_text[end - 1].isspace() and self._safe_end(end) == end:
                return end
        return limit

    def _commit(self, end: int) -> str | None:
        chunk = self.pending_text[:end].strip()
        self.pending_text = self.pending_text[end:].lstrip()
        return chunk or None
