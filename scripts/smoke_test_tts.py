#!/usr/bin/env python3
"""Milestone 0: Qwen3-TTS model load + synthesis smoke test.

Usage:
    python3 /workspace/audri/scripts/smoke_test_tts.py [--model Qwen/Qwen3-TTS-0.6B]

Outputs:
    /workspace/audri/outputs/smoke_test.wav

Prints:
    model_load_ms, synth_ms, audio_duration_s, RTF (real-time factor)

NOTE: Run this inside the qwen3-tts venv:
    source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate
    python3 /workspace/audri/scripts/smoke_test_tts.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave
from typing import Optional

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from audri.tts.driver import Qwen3TTSDriver

# Ensure workspace src is on the path when running inside the container
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

TEST_TEXT = "Hello! This is a smoke test of the Qwen3 TTS system."
OUTPUT_DIR = os.path.join(_REPO, "outputs")
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "smoke_test.wav")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS smoke test")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="HuggingFace model ID (default: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)",
    )
    p.add_argument(
        "--voice",
        default="default",
        help="Voice to use for synthesis",
    )
    p.add_argument(
        "--text",
        default=TEST_TEXT,
        help="Text to synthesise",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help="Optional instruction prompt for custom voice models",
    )
    return p.parse_args()


async def _run_smoke(
    model: str, voice: str, text: str, instruction: Optional[str]
) -> tuple[int, int, float, np.ndarray, int]:
    driver = Qwen3TTSDriver(model_name=model, speaker=voice, instruction=instruction)
    t0 = time.monotonic()
    await driver._ensure_loaded()
    model_load_ms = (time.monotonic() - t0) * 1000

    print(f"  model_load_ms = {model_load_ms:.0f}")
    print(f"  sample_rate   = {driver.sample_rate} Hz")

    frames: list[np.ndarray] = []
    t1 = time.monotonic()
    async for frame in driver.synth(text, voice=voice, language="English", instruct=instruction):
        frames.append(np.frombuffer(frame, dtype=np.int16))
    synth_ms = (time.monotonic() - t1) * 1000

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
    return model_load_ms, synth_ms, audio, driver.sample_rate


def _write_wav(pcm: np.ndarray, sample_rate: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with wave.open(OUTPUT_WAV, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()

    print(f"Model:  {args.model}")
    print(f"Text:   {args.text!r}")
    print()

    model_load_ms, synth_ms, waveform, sample_rate = asyncio.run(
        _run_smoke(args.model, args.voice, args.text, getattr(args, "instruction", None))
    )

    audio_duration_s = len(waveform) / sample_rate if sample_rate else 0.0
    rtf = (synth_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else float("inf")

    print(f"  synth_ms      = {synth_ms:.0f}")
    print(f"  audio_duration_s = {audio_duration_s:.3f}")
    print(f"  RTF (synth/audio) = {rtf:.3f}")
    print()

    _write_wav(waveform, sample_rate)
    print(f"WAV saved → {OUTPUT_WAV}")
    print()

    print("=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    print(f"  model_load_ms    = {model_load_ms:.0f}")
    print(f"  synth_ms         = {synth_ms:.0f}")
    print(f"  audio_duration_s = {audio_duration_s:.3f}")
    rtf_label = "PASS (real-time)" if rtf < 1.0 else f"SLOW ({rtf:.2f}x — hardware limited)"
    print(f"  RTF              = {rtf:.3f}  ({rtf_label})")
    print()

    if audio_duration_s == 0.0:
        print("SMOKE TEST FAILED — no audio generated.")
        sys.exit(1)

    if rtf >= 1.0:
        print(f"NOTE: RTF {rtf:.2f}x — synthesis works but slower than real-time on this hardware.")
    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
