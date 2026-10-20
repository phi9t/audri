#!/usr/bin/env python3
"""Qwen token-stream demo. Run through `just demo` in the Zephyr container."""

import argparse
import asyncio
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audri import SpeakTool  # noqa: E402
from audri.tts.audio_egress import LocalAudioEgress, WaveAudioEgress  # noqa: E402
from audri.tts.driver import Qwen3TTSDriver  # noqa: E402

TEXT = ("Streaming text-to-speech begins when a useful text chunk is ready. "
        "Each output frame carries its actual sample rate. "
        "A new turn can interrupt playback without cancelling another conversation.")


async def run_demo(model: str, text: str) -> None:
    headless = os.environ.get("AUDRI_HEADLESS", "").lower() in {"1", "true", "yes"}
    sink = (WaveAudioEgress(REPO / "outputs/speak_demo.wav", overwrite=True)
            if headless else LocalAudioEgress())
    async with SpeakTool(Qwen3TTSDriver(model_name=model)) as tool:
        await tool.start("demo", egress=sink)
        for offset in range(0, len(text), 4):
            await tool.delta("demo", text[offset:offset + 4])
            await asyncio.sleep(.03)
        metrics = await tool.end("demo")
        print(metrics.as_dict())
        if headless:
            print(f"WAV: {REPO / 'outputs/speak_demo.wav'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--text", default=TEXT)
    arguments = parser.parse_args()
    asyncio.run(run_demo(arguments.model, arguments.text))
