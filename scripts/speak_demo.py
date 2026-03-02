#!/usr/bin/env python3
"""Milestone 1: Simulated LLM token stream → real-time local audio playback.

Usage:
    python3 /workspace/audri/scripts/speak_demo.py [--model Qwen/Qwen3-TTS-0.6B]

This demo simulates a streaming LLM response (token-by-token with small delays)
and pipes it through the full audri pipeline:
    tokens → TextChunker → HFQwen3TTSDriver → LocalAudioEgress → speakers

Per-chunk metrics are logged: commit time, TTS start, first audio frame, RTF.

If running on a headless GPU server (no audio output device), you can redirect
the output to a WAV file by setting AUDRI_HEADLESS=1, which writes each chunk
to outputs/speak_demo_<n>.wav instead of playing through sounddevice.

NOTE: Run inside the qwen3-tts venv:
    source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate
    python3 /workspace/audri/scripts/speak_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import AsyncIterator

import numpy as np
import wave

# Ensure /repo/src is on the path
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("speak_demo")


# ---------------------------------------------------------------------- #
# Simulated LLM stream
# ---------------------------------------------------------------------- #

DEMO_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Streaming text-to-speech allows voice output to begin before the full "
    "response is ready, dramatically reducing perceived latency. "
    "This is especially useful for conversational AI assistants."
)

# Approximate token sizes for simulation
_TOKEN_SIZE = 4  # chars per "token"
_TOKEN_DELAY_S = 0.03  # 30ms between tokens ≈ ~30 tok/s


async def fake_llm_stream(text: str) -> AsyncIterator[str]:
    """Yield token-sized chunks of *text* with realistic inter-token delays."""
    for i in range(0, len(text), _TOKEN_SIZE):
        chunk = text[i : i + _TOKEN_SIZE]
        yield chunk
        await asyncio.sleep(_TOKEN_DELAY_S)


# ---------------------------------------------------------------------- #
# Headless mode: write WAV instead of playing audio
# ---------------------------------------------------------------------- #

HEADLESS = os.environ.get("AUDRI_HEADLESS", "").lower() in ("1", "true", "yes")
OUTPUT_DIR = os.path.join(_REPO, "outputs")


def _make_headless_driver(model_name: str):
    """Return a driver subclass that saves WAV files instead of streaming to speakers."""
    from audri.tts.driver import Qwen3TTSDriver

    class HeadlessDriver(Qwen3TTSDriver):
        """Collects PCM frames and saves them to WAV on each synth() call."""

        _chunk_idx: int = 0

        async def synth(self, text, voice="default", language="English", instruction=None):
            frames = []
            async for frame in super().synth(text, voice, language, instruction):
                frames.append(frame)
                yield frame
            if frames:
                pcm = np.concatenate([np.frombuffer(f, dtype=np.int16) for f in frames])
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                path = os.path.join(OUTPUT_DIR, f"speak_demo_{self._chunk_idx:02d}.wav")
                with wave.open(path, "wb") as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(self.sample_rate)
                    wav_out.writeframes(pcm.tobytes())
                logger.info("Headless: saved %s (%d samples)", path, len(pcm))
                HeadlessDriver._chunk_idx += 1

    return HeadlessDriver(model_name)


class NullAudioEgress:
    """Headless egress that drains audio frames without playback."""

    async def run(self, pcm_queue: asyncio.Queue):
        while True:
            frame = await pcm_queue.get()
            if frame is None:
                break


# ---------------------------------------------------------------------- #
# Main demo
# ---------------------------------------------------------------------- #

async def run_demo(model_name: str, text: str) -> None:
    from audri.speak_tool import SpeakTool
    from audri.tts.chunker import ChunkerConfig
    from audri.tts.driver import Qwen3TTSDriver

    print("=" * 60)
    print("audri speak_demo — Milestone 1")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Headless: {HEADLESS}")
    print(f"Text:     {text!r}")
    print()

    if HEADLESS:
        driver = _make_headless_driver(model_name)
    else:
        driver = Qwen3TTSDriver(model_name=model_name)

    chunker_config = ChunkerConfig(
        min_chars_to_start=48,
        max_chars_per_chunk=180,
        lookahead_chars=40,
        boundary_timeout_ms=220,
    )

    egress_factory = NullAudioEgress if HEADLESS else None
    tool = SpeakTool(
        driver=driver,
        chunker_config=chunker_config,
        egress_factory=egress_factory,
    )

    session_id = "demo-1"
    t_start = time.monotonic()

    print("Starting session …")
    await tool.start(session_id, voice="default", language="English")

    t_stream_start = time.monotonic()
    print("Streaming tokens …")
    token_count = 0
    async for token in fake_llm_stream(text):
        await tool.delta(session_id, token)
        token_count += 1

    t_last_token = time.monotonic()
    print(f"\nStream complete — {token_count} tokens in {(t_last_token - t_stream_start)*1000:.0f} ms")
    print("Awaiting TTS + audio completion …")

    await tool.end(session_id)
    t_done = time.monotonic()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"  total_elapsed_ms = {(t_done - t_start)*1000:.0f}")
    print(f"  stream_ms        = {(t_last_token - t_stream_start)*1000:.0f}")
    print(f"  tts_tail_ms      = {(t_done - t_last_token)*1000:.0f}")
    if HEADLESS:
        print(f"\n  WAV files saved to: {OUTPUT_DIR}/speak_demo_*.wav")
    print()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="audri speak_demo")
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    p.add_argument("--text", default=DEMO_TEXT)
    args = p.parse_args()

    asyncio.run(run_demo(args.model, args.text))


if __name__ == "__main__":
    main()
