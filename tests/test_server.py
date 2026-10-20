import asyncio
import io
import json
import unittest
import wave

from audri.audio import AudioFormat, AudioFrame
from audri.tts.driver import TTSDriver

try:
    from fastapi.testclient import TestClient
    from audri.server import ServerLimits, create_app
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


class RecordingDriver(TTSDriver):
    def __init__(self):
        self.requests = []
        self.cleaned = 0
        self.closed = False
        self.pcm = b"\x01\x00" * 882

    async def stream(self, request, *, cancel=None):
        self.requests.append(request)
        try:
            if cancel:
                cancel.raise_if_cancelled()
            yield AudioFrame(self.pcm, AudioFormat(44100))
            await asyncio.sleep(0)
            yield AudioFrame(self.pcm, AudioFormat(44100))
        finally:
            self.cleaned += 1

    async def aclose(self):
        self.closed = True


class FailingDriver(RecordingDriver):
    async def stream(self, request, *, cancel=None):
        try:
            raise RuntimeError("upstream secret-bearing response must not be returned")
            yield
        finally:
            self.cleaned += 1


class SlowDriver(RecordingDriver):
    async def stream(self, request, *, cancel=None):
        try:
            yield AudioFrame(self.pcm, AudioFormat(44100))
            await asyncio.Event().wait()
        finally:
            self.cleaned += 1


class NeverReadyDriver(RecordingDriver):
    async def stream(self, request, *, cancel=None):
        try:
            await asyncio.Event().wait()
            yield
        finally:
            self.cleaned += 1


@unittest.skipUnless(HAS_SERVER, "optional server dependencies are unavailable")
class ServerTests(unittest.TestCase):
    def test_wav_is_finalized_at_native_rate(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver)) as client:
            response = client.post("/v1/audio/speech", json={"input": "Hello."})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertEqual(int(response.headers["content-length"]), len(response.content))
            with wave.open(io.BytesIO(response.content), "rb") as source:
                self.assertEqual(source.getframerate(), 44100)
                self.assertEqual(source.getnchannels(), 1)
                self.assertEqual(source.readframes(source.getnframes()), driver.pcm * 2)
            self.assertEqual(client.get("/health").json()["active_sessions"], 0)
        self.assertTrue(driver.closed)
        self.assertEqual(driver.cleaned, 1)

    def test_pcm_stream_has_format_headers_and_no_wav_header(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver)) as client:
            response = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "pcm"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, driver.pcm * 2)
            self.assertEqual(response.headers["x-audio-sample-rate"], "44100")
            self.assertEqual(response.headers["x-audio-encoding"], "pcm_s16le")
            self.assertEqual(client.get("/health").json()["active_sessions"], 0)

    def test_request_validation_and_authentication_precede_inference(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver, api_key="test-key")) as client:
            self.assertEqual(client.post("/v1/audio/speech", json={"input": "hello"}).status_code, 401)
            headers = {"Authorization": "Bearer test-key"}
            for payload in ({"input": ""}, {"input": "hello", "model": "guess"}, {"input": "hello", "response_format": "mp3"},
                            {"input": "hello", "temperature": 0}, {"input": "hello", "references": [{"audio": "***", "text": "voice"}]}):
                response = client.post("/v1/audio/speech", json=payload, headers=headers)
                self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(driver.requests, [])
            self.assertEqual(client.post("/v1/audio/speech", json={"input": "hello"}, headers=headers).status_code, 200)

    def test_http_body_size_is_bounded(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver, limits=ServerLimits(max_request_bytes=32))) as client:
            response = client.post("/v1/audio/speech", json={"input": "x" * 100})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(driver.requests, [])

    def test_backend_failure_is_visible_and_redacted(self):
        driver = FailingDriver()
        with TestClient(create_app(driver)) as client:
            response = client.post("/v1/audio/speech", json={"input": "hello"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("secret-bearing", response.text)
            self.assertEqual(client.get("/health").json()["active_sessions"], 0)
        self.assertEqual(driver.cleaned, 1)

    def test_request_timeout_closes_backend(self):
        driver = NeverReadyDriver()
        with TestClient(create_app(driver, limits=ServerLimits(request_timeout_s=0.02))) as client:
            response = client.post("/v1/audio/speech", json={"input": "hello"})
            self.assertEqual(response.status_code, 504)
            self.assertEqual(driver.cleaned, 1)
            self.assertEqual(client.get("/health").json()["active_sessions"], 0)

    def test_wav_duration_is_bounded(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver, limits=ServerLimits(max_audio_seconds=0.01))) as client:
            response = client.post("/v1/audio/speech", json={"input": "hello"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(driver.cleaned, 1)

    def test_websocket_emits_format_before_pcm_and_terminal_metrics(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver)) as client:
            with client.websocket_connect("/v1/speech/stream") as websocket:
                websocket.send_json({"type": "start", "session_id": "first", "language": "Chinese"})
                self.assertEqual(websocket.receive_json()["type"], "started")
                websocket.send_json({"type": "delta", "text": "你好，世界。"})
                websocket.send_json({"type": "end"})
                format = websocket.receive_json()
                self.assertEqual(format["type"], "format")
                self.assertEqual(format["sample_rate"], 44100)
                pcm = bytearray()
                while True:
                    message = websocket.receive()
                    if message.get("bytes") is not None:
                        pcm.extend(message["bytes"])
                    else:
                        result = json.loads(message["text"])
                        self.assertEqual(result["type"], "ended")
                        self.assertGreater(result["metrics"]["audio_seconds"], 0)
                        break
                self.assertEqual(bytes(pcm), driver.pcm * 2)
                self.assertEqual(driver.requests[0].language, "Chinese")
                websocket.send_json({"type": "start", "session_id": "second"})
                self.assertEqual(websocket.receive_json()["type"], "started")
                websocket.send_json({"type": "stop"})
                self.assertEqual(websocket.receive_json()["type"], "stopped")

    def test_websocket_stop_interrupts_drain_and_has_no_stale_pcm(self):
        driver = SlowDriver()
        with TestClient(create_app(driver)) as client:
            with client.websocket_connect("/v1/speech/stream") as websocket:
                websocket.send_json({"type": "start", "session_id": "barge"})
                self.assertEqual(websocket.receive_json()["type"], "started")
                websocket.send_json({"type": "delta", "text": "Hello."})
                websocket.send_json({"type": "end"})
                self.assertEqual(websocket.receive_json()["type"], "format")
                self.assertEqual(websocket.receive_bytes(), driver.pcm)
                websocket.send_json({"type": "stop"})
                self.assertEqual(websocket.receive_json(), {"type": "stopped", "session_id": "barge"})
                websocket.send_json({"type": "ping"})
                self.assertEqual(websocket.receive_json(), {"type": "pong"})
                self.assertEqual(driver.cleaned, 1)

    def test_background_failure_is_reported_without_an_end_message(self):
        driver = FailingDriver()
        with TestClient(create_app(driver)) as client:
            with client.websocket_connect("/v1/speech/stream") as websocket:
                websocket.send_json({"type": "start"})
                self.assertEqual(websocket.receive_json()["type"], "started")
                websocket.send_json({"type": "delta", "text": "A sentence long enough to commit immediately. " * 4})
                result = websocket.receive_json()
                self.assertEqual(result["type"], "error")
                self.assertNotIn("secret-bearing", result["message"])

    def test_websocket_limits_and_admission_are_explicit(self):
        driver = RecordingDriver()
        with TestClient(create_app(driver, max_sessions=1, limits=ServerLimits(max_delta_chars=8))) as client:
            with client.websocket_connect("/v1/speech/stream") as websocket:
                websocket.send_json({"type": "start"})
                websocket.receive_json()
                self.assertEqual(client.post("/v1/audio/speech", json={"input": "hello"}).status_code, 429)
                websocket.send_json({"type": "delta", "text": "x" * 9})
                self.assertEqual(websocket.receive_json()["type"], "error")
                websocket.send_json({"type": "stop"})
                self.assertEqual(websocket.receive_json()["type"], "stopped")

    def test_invalid_limits_fail_at_configuration(self):
        for fields in ({"max_audio_seconds": float("nan")}, {"max_delta_chars": 1.5}, {"request_timeout_s": 0}):
            with self.assertRaises(ValueError):
                ServerLimits(**fields)


@unittest.skipUnless(HAS_SERVER, "optional server dependencies are unavailable")
class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    def scope(self):
        return {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
            "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }

    async def test_pcm_disconnect_closes_pending_backend_iterator(self):
        driver = SlowDriver()
        app = create_app(driver)
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b'{"input":"hello","response_format":"pcm"}', "more_body": False})
        received = []

        async def send(message):
            received.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                await incoming.put({"type": "http.disconnect"})

        await asyncio.wait_for(app(self.scope(), incoming.get, send), 2)
        self.assertEqual(driver.cleaned, 1)
        self.assertTrue(any(item["type"] == "http.response.body" and item.get("body") == driver.pcm for item in received))

    async def test_disconnect_before_first_frame_stops_generation(self):
        started = asyncio.Event()

        class PendingDriver(NeverReadyDriver):
            async def stream(self, request, *, cancel=None):
                started.set()
                async for frame in super().stream(request, cancel=cancel):
                    yield frame

        driver = PendingDriver()
        app = create_app(driver)
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b'{"input":"hello"}', "more_body": False})
        received = []

        async def send(message):
            received.append(message)

        task = asyncio.create_task(app(self.scope(), incoming.get, send))
        try:
            await asyncio.wait_for(started.wait(), 1)
            await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(driver.cleaned, 1)
        self.assertTrue(any(item["type"] == "http.response.start" and item["status"] == 499 for item in received))


if __name__ == "__main__":
    unittest.main()
