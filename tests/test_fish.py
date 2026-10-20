import asyncio
import base64
import importlib.util
import struct
import unittest

from audri.tts.fish import FishSpeechDriver, FishSpeechError
from audri.tts.types import CancellationToken, SynthesisRequest, VoiceReference


def streaming_wav(pcm, rate=44100, channels=1):
    return (
        b"RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00"
        + struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
        + b"data\xff\xff\xff\xff" + pcm
    )


class FakeResponse:
    def __init__(self, chunks=(), *, status=200, gate=None, result=None):
        self.chunks = chunks
        self.status_code = status
        self.gate = gate
        self.result = result
        self.read_chunks = 0

    def json(self):
        return self.result

    async def aiter_bytes(self):
        if self.gate is not None:
            await self.gate.wait()
        for chunk in self.chunks:
            await asyncio.sleep(0)
            self.read_chunks += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class FakeStream:
    def __init__(self, client, response):
        self.client = client
        self.response = response

    async def __aenter__(self):
        self.client.active += 1
        self.client.maximum = max(self.client.maximum, self.client.active)
        self.client.entered.set()
        return self.response

    async def __aexit__(self, *args):
        self.client.active -= 1
        self.client.exited.set()


class FakeClient:
    def __init__(self, factory, *, references=("narrator",)):
        self.factory = factory
        self.references = references
        self.calls = []
        self.metadata_calls = []
        self.active = 0
        self.maximum = 0
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()
        self.closed = False

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeStream(self, self.factory())

    async def get(self, url, **kwargs):
        self.metadata_calls.append((url, kwargs))
        if url.endswith("/references/list"):
            result = {"reference_ids": list(self.references)}
        elif url.endswith("/tts/defaults"):
            result = {"model_family": "drama3", "temperature": 0.7}
        else:
            result = {"status": "ok"}
        return FakeResponse(result=result)

    async def aclose(self):
        self.closed = True


async def collect(driver, request=None, cancel=None):
    return [frame async for frame in driver.stream(request or SynthesisRequest("hello"), cancel=cancel)]


class FishDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_wav_uses_actual_format(self):
        pcm = struct.pack("<h", -123) * 3000
        wire = streaming_wav(pcm, rate=48000, channels=2)
        client = FakeClient(lambda: FakeResponse([wire[i:i + 7] for i in range(0, len(wire), 7)]))
        driver = FishSpeechDriver(client=client, api_key="test-only")
        frames = await collect(driver)
        self.assertEqual(b"".join(f.pcm for f in frames), pcm)
        self.assertTrue(all(f.format.sample_rate == 48000 and f.format.channels == 2 for f in frames))
        self.assertEqual(client.active, 0)
        method, url, kwargs = client.calls[0]
        self.assertEqual((method, url), ("POST", "http://127.0.0.1:8080/v1/tts"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-only")
        self.assertTrue(kwargs["json"]["streaming"])
        self.assertEqual(kwargs["json"]["format"], "wav")
        self.assertNotIn("model", kwargs["json"])
        await driver.aclose()
        self.assertFalse(client.closed)

    async def test_inline_reference_and_speaker_mapping(self):
        audio = b"reference-audio" * 32
        reference = VoiceReference(audio=audio, text="A real transcript.", speaker=3)
        client = FakeClient(lambda: FakeResponse([streaming_wav(b"\0\0" * 100)]))
        driver = FishSpeechDriver(client=client)
        await collect(driver, SynthesisRequest("你好", instruction="whisper", references=(reference,), seed=42))
        payload = client.calls[0][2]["json"]
        self.assertEqual(payload["text"], "[whisper] 你好")
        self.assertEqual(base64.b64decode(payload["references"][0]["audio"]), audio)
        self.assertEqual(payload["references"][0]["speaker"], 3)
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("language", payload)
        await driver.aclose()

    async def test_saved_voice_is_verified_before_generation(self):
        client = FakeClient(lambda: FakeResponse([streaming_wav(b"\0\0" * 100)]))
        driver = FishSpeechDriver(client=client)
        await collect(driver, SynthesisRequest("hello", voice="narrator"))
        self.assertEqual(client.calls[0][2]["json"]["reference_id"], "narrator")
        self.assertTrue(client.metadata_calls[0][0].endswith("/v1/references/list"))
        with self.assertRaisesRegex(FishSpeechError, "not registered"):
            await collect(driver, SynthesisRequest("hello", reference_id="missing"))
        self.assertEqual(len(client.calls), 1)
        await driver.aclose()

    async def test_http_failure_is_not_retried(self):
        client = FakeClient(lambda: FakeResponse(status=503))
        driver = FishSpeechDriver(client=client)
        with self.assertRaises(FishSpeechError) as raised:
            await collect(driver)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.active, 0)
        await driver.aclose()

    async def test_truncated_stream_fails_instead_of_silent_success(self):
        client = FakeClient(lambda: FakeResponse([streaming_wav(b"\0\0" * 1000 + b"x")]))
        driver = FishSpeechDriver(client=client)
        with self.assertRaisesRegex(FishSpeechError, "sample frame"):
            await collect(driver)
        self.assertEqual(client.active, 0)
        await driver.aclose()

    async def test_transport_failure_is_not_retried(self):
        client = FakeClient(lambda: FakeResponse([TimeoutError("transport idle")]))
        driver = FishSpeechDriver(client=client)
        with self.assertRaises(TimeoutError):
            await collect(driver)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.active, 0)
        await driver.aclose()

    async def test_pre_cancelled_request_opens_no_connection(self):
        token = CancellationToken()
        token.cancel()
        client = FakeClient(lambda: FakeResponse([]))
        driver = FishSpeechDriver(client=client)
        with self.assertRaises(asyncio.CancelledError):
            await collect(driver, cancel=token)
        self.assertEqual(client.calls, [])
        await driver.aclose()

    async def test_empty_text_emits_nothing_and_makes_no_request(self):
        client = FakeClient(lambda: FakeResponse([]))
        driver = FishSpeechDriver(client=client)
        self.assertEqual(await collect(driver, SynthesisRequest("  ")), [])
        self.assertEqual(client.calls, [])
        await driver.aclose()

    async def test_cancel_during_network_read_closes_response(self):
        client = FakeClient(lambda: FakeResponse([], gate=asyncio.Event()))
        driver = FishSpeechDriver(client=client)
        token = CancellationToken()
        task = asyncio.create_task(collect(driver, cancel=token))
        await asyncio.wait_for(client.entered.wait(), 1)
        token.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(client.active, 0)
        await driver.aclose()

    async def test_cancel_while_consumer_paused_still_releases_transport(self):
        wire = streaming_wav(b"\0\0" * 44100)
        client = FakeClient(lambda: FakeResponse([wire]))
        driver = FishSpeechDriver(client=client)
        token = CancellationToken()
        stream = driver.stream(SynthesisRequest("hello"), cancel=token)
        await anext(stream)
        token.cancel()
        await asyncio.wait_for(client.exited.wait(), 1)
        self.assertEqual(client.active, 0)
        with self.assertRaises(asyncio.CancelledError):
            await anext(stream)
        await stream.aclose()
        await driver.aclose()

    async def test_concurrency_limit_and_cancelled_waiter(self):
        gate = asyncio.Event()
        client = FakeClient(lambda: FakeResponse([streaming_wav(b"\0\0" * 100)], gate=gate))
        driver = FishSpeechDriver(client=client, max_concurrency=1)
        first = asyncio.create_task(collect(driver))
        await asyncio.wait_for(client.entered.wait(), 1)
        token = CancellationToken()
        second = asyncio.create_task(collect(driver, cancel=token))
        await asyncio.sleep(0)
        token.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(second, 1)
        gate.set()
        await asyncio.wait_for(first, 1)
        await collect(driver)
        self.assertEqual(client.maximum, 1)
        self.assertEqual(len(client.calls), 2)
        await driver.aclose()

    async def test_external_task_cancel_and_driver_close_clean_up(self):
        client = FakeClient(lambda: FakeResponse([], gate=asyncio.Event()))
        driver = FishSpeechDriver(client=client)
        task = asyncio.create_task(collect(driver))
        await asyncio.wait_for(client.entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(client.active, 0)
        client.entered.clear()
        task = asyncio.create_task(collect(driver))
        await asyncio.wait_for(client.entered.wait(), 1)
        await driver.aclose()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(client.active, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await driver.health()

    async def test_health_and_model_defaults(self):
        client = FakeClient(lambda: FakeResponse())
        driver = FishSpeechDriver(client=client)
        self.assertEqual(await driver.health(), {"status": "ok"})
        self.assertEqual((await driver.defaults())["model_family"], "drama3")
        await driver.aclose()

    async def test_early_generator_close_stops_production(self):
        client = FakeClient(lambda: FakeResponse([streaming_wav(b"\0\0" * 44100)]))
        driver = FishSpeechDriver(client=client)
        stream = driver.stream(SynthesisRequest("hello"))
        await anext(stream)
        await stream.aclose()
        self.assertEqual(client.active, 0)
        self.assertFalse(driver._streams)
        await driver.aclose()

    async def test_configuration_and_ambiguous_voice_rejected(self):
        for kwargs in ({"base_url": "file:///tmp/x"}, {"base_url": "http://user:pass@host"},
                       {"timeout": 0}, {"timeout": float("nan")}, {"max_concurrency": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FishSpeechDriver(**kwargs)
        request = SynthesisRequest("hello", voice="first", reference_id="second")
        with self.assertRaises(ValueError):
            FishSpeechDriver._payload(request)

    @unittest.skipUnless(importlib.util.find_spec("httpx"), "optional httpx extra not installed")
    async def test_real_httpx_mock_transport(self):
        import httpx

        wire = streaming_wav(b"\0\0" * 1500)
        seen = []

        class WireStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0, len(wire), 11):
                    yield wire[i:i + 11]

        async def handler(request):
            seen.append(request)
            return httpx.Response(200, headers={"Content-Type": "audio/wav"}, stream=WireStream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            driver = FishSpeechDriver(client=client)
            frames = await collect(driver)
            self.assertEqual(b"".join(f.pcm for f in frames), b"\0\0" * 1500)
            self.assertEqual(seen[0].headers["content-type"], "application/json")
            await driver.aclose()


if __name__ == "__main__":
    unittest.main()
