#!/usr/bin/env python3
"""Real Qwen model load and synthesis. Run `just smoke` in the GPU container."""

import argparse
import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audri import SynthesisRequest  # noqa: E402
from audri.tts.audio_egress import WaveAudioEgress  # noqa: E402
from audri.tts.driver import Qwen3TTSDriver  # noqa: E402


async def smoke(args):
    output = REPO / "outputs/smoke_test.wav"
    sink = WaveAudioEgress(output, overwrite=True)
    async with Qwen3TTSDriver(model_name=args.model) as driver:
        started = time.monotonic()
        await driver._ensure_loaded()
        model_load_ms = (time.monotonic() - started) * 1000
        started = time.monotonic()
        duration = 0.0
        first_audio_ms = None
        try:
            request = SynthesisRequest(args.text, voice=args.voice, instruction=args.instruction)
            async with aclosing(driver.stream(request)) as stream:
                async for frame in stream:
                    if first_audio_ms is None:
                        first_audio_ms = (time.monotonic() - started) * 1000
                    await sink.write(frame)
                    duration += frame.duration_s
            if duration <= 0:
                raise RuntimeError("model generated no audio")
            await sink.close()
        except BaseException:
            await sink.abort()
            raise
        elapsed = time.monotonic() - started
        return {"output": str(output), "model_load_ms": model_load_ms,
                "first_audio_ms": first_audio_ms, "audio_seconds": duration,
                "elapsed_seconds": elapsed, "rtf": elapsed / duration,
                "sample_rate": sink.format.sample_rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--instruction")
    parser.add_argument("--text", default="Hello! This is a smoke test of the Qwen3 TTS system.")
    print(json.dumps(asyncio.run(smoke(parser.parse_args())), indent=2))
