"""A real loopback HTTP socket → Fish adapter → session → published WAV.

The upstream double intentionally splits the WAV header and PCM samples. It
tests protocol integration, not the acoustic quality of downloaded weights.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from audri.speak_tool import SpeakTool
from audri.tts.audio_egress import WaveAudioEgress
from audri.tts.chunker import ChunkerConfig
from audri.tts.fish import FishSpeechDriver


@unittest.skipUnless(importlib.util.find_spec("httpx"), "httpx extra required")
class SocketIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_stream_to_native_rate_wav_over_real_http(self):
        pcm = struct.pack("<1765h", *[index % 1000 for index in range(1765)])
        header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0xFFFFFFFF, b"WAVE",
                             b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", 0xFFFFFFFF)
        body = header + pcm
        requests = []
        workers = set()

        async def handle(reader, writer):
            task = asyncio.current_task()
            workers.add(task)
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                fields = head.decode("ascii").split("\r\n")
                self.assertTrue(fields[0].startswith("POST /v1/tts "))
                headers = dict(line.split(":", 1) for line in fields[1:] if ":" in line)
                length = int(next(value for key, value in headers.items() if key.lower() == "content-length"))
                payload = json.loads(await reader.readexactly(length))
                requests.append(payload)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                # Odd chunk lengths guarantee PCM samples cross HTTP chunks.
                for index in range(0, len(body), 37):
                    piece = body[index:index + 37]
                    writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    await writer.drain()
                    await asyncio.sleep(0)
                writer.write(b"0\r\n\r\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                workers.discard(task)

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "speech.wav"
                driver = FishSpeechDriver(f"http://127.0.0.1:{port}")
                async with SpeakTool(driver, chunker_config=ChunkerConfig(min_chars_to_start=5,
                                      max_chars_per_chunk=40, boundary_timeout_ms=1000)) as tool:
                    await tool.start("socket", egress=WaveAudioEgress(output))
                    for delta in ["Hello ", "world.", " 你好", "，世界。"]:
                        await tool.delta("socket", delta)
                    metrics = await asyncio.wait_for(tool.end("socket"), 3)
                self.assertGreaterEqual(len(requests), 2)
                self.assertTrue(all(p["format"] == "wav" and p["streaming"] for p in requests))
                with wave.open(str(output), "rb") as source:
                    self.assertEqual(source.getframerate(), 44100)
                    self.assertEqual(source.readframes(source.getnframes()), pcm * len(requests))
                self.assertAlmostEqual(metrics.audio_seconds, 1765 * len(requests) / 44100)
                self.assertEqual(metrics.state, "done")
        finally:
            server.close()
            await server.wait_closed()
            if workers:
                await asyncio.gather(*workers)
