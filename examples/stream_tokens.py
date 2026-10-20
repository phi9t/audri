"""Run in Audri's container environment against a self-hosted Fish server.

This example deliberately has no dependency on an LLM vendor. Replace
``token_source`` with your model's async iterator of text deltas.
"""

import asyncio
import os

from audri import SpeakTool
from audri.tts.audio_egress import WaveAudioEgress
from audri.tts.fish import FishSpeechDriver
from audri.tts.types import SynthesisRequest


async def token_source():
    text = "Hello. This is speech generated as text arrives. 你好，这是一段流式语音。"
    for start in range(0, len(text), 4):
        yield text[start:start + 4]
        await asyncio.sleep(0.03)


async def main():
    driver = FishSpeechDriver(base_url=os.environ.get("FISH_URL", "http://127.0.0.1:8080"),
                              api_key=os.environ.get("FISH_API_KEY"))
    async with SpeakTool(driver) as tool:
        await tool.start("example", request=SynthesisRequest("", language="auto"),
                         egress=WaveAudioEgress("outputs/token-stream.wav"))
        async for delta in token_source():
            await tool.delta("example", delta)
        metrics = await tool.end("example")
        print(metrics.as_dict())
        # During generation, barge-in is simply:
        # await tool.stop("example")


if __name__ == "__main__":
    asyncio.run(main())
