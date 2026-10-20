"""Optional HTTP and WebSocket transport for Audri's streaming core.

PCM HTTP responses stream immediately. WAV responses are finalized on temporary
disk before being sent, so their RIFF lengths are valid for ordinary players.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import json
import logging
import math
import tempfile
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse

from .speak_tool import SpeakTool
from .tts.audio_egress import CallbackAudioEgress
from .tts.session import SessionConfig
from .tts.types import CancellationToken, SynthesisRequest, VoiceReference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerLimits:
    max_request_bytes: int = 4 * 1024 * 1024
    max_text_chars: int = 100000
    max_delta_chars: int = 4096
    max_audio_seconds: float = 600.0
    request_timeout_s: float = 180.0
    websocket_idle_timeout_s: float = 60.0
    send_timeout_s: float = 15.0
    websocket_pending_deltas: int = 16

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("max_request_bytes", "max_text_chars", "max_delta_chars", "websocket_pending_deltas"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")


class _Disconnected(Exception):
    pass


class _Admission:
    def __init__(self, maximum: int):
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("max_sessions must be a positive integer")
        self.maximum = maximum
        self.active = 0
        self.lock = asyncio.Lock()

    async def enter(self) -> bool:
        async with self.lock:
            if self.active >= self.maximum:
                return False
            self.active += 1
            return True

    async def leave(self):
        async with self.lock:
            self.active -= 1


class _OwnedStreamingResponse(StreamingResponse):
    """Release request resources even if the peer disconnects during a send."""

    def __init__(self, *args, cleanup, send_timeout_s, **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup
        self._send_timeout_s = send_timeout_s

    async def __call__(self, scope, receive, send):
        async def bounded_send(message):
            async with asyncio.timeout(self._send_timeout_s):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            # Starlette cancels an AnyIO scope after disconnect. Shield this
            # teardown from its repeated cancellation while still bounding a
            # misbehaving iterator's close operation.
            with anyio.CancelScope(shield=True):
                try:
                    if hasattr(self.body_iterator, "aclose"):
                        async with asyncio.timeout(5):
                            await self.body_iterator.aclose()
                finally:
                    await self._cleanup()


def _authorized(header: str | None, key: str | None) -> bool:
    return key is None or (header is not None and hmac.compare_digest(header.encode("utf-8"), ("Bearer " + key).encode("utf-8")))


async def _json_body(request: Request, maximum: int) -> dict:
    body = bytearray()
    async for piece in request.stream():
        if len(body) + len(piece) > maximum:
            raise HTTPException(413, "request body is too large")
        body.extend(piece)
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, "request body must be JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, "request body must be an object")
    return parsed


_REQUEST_FIELDS = {
    "voice", "language", "instruction", "reference_id", "references", "seed",
    "max_new_tokens", "temperature", "top_p", "repetition_penalty", "chunk_length",
    "normalize", "single_chunk",
}


def _synthesis_request(payload: dict, text: str, limits: ServerLimits) -> SynthesisRequest:
    if not isinstance(text, str) or len(text) > limits.max_text_chars:
        raise ValueError("input must be text within max_text_chars")
    extra = set(payload) - _REQUEST_FIELDS
    if extra:
        raise ValueError("unknown request fields: " + ", ".join(sorted(extra)))
    fields = dict(payload)
    references = fields.pop("references", [])
    if not isinstance(references, list) or len(references) > 16:
        raise ValueError("references must be a list containing at most 16 entries")
    decoded = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) - {"audio", "text", "speaker"}:
            raise ValueError("reference entries accept audio, text, and optional speaker")
        if not isinstance(reference.get("audio"), str) or not isinstance(reference.get("text"), str):
            raise ValueError("references require base64 audio and transcript text")
        try:
            raw = base64.b64decode(reference["audio"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("reference audio must be valid base64") from exc
        decoded.append(VoiceReference(audio=raw, text=reference["text"], speaker=reference.get("speaker")))
    return SynthesisRequest(text=text, references=tuple(decoded), **fields)


def _format_dict(format) -> dict:
    return {
        "sample_rate": format.sample_rate,
        "channels": format.channels,
        "sample_width": format.sample_width,
        "encoding": "pcm_s16le",
    }


async def _audio_frames(driver, synthesis, token, limits):
    total = 0.0
    format = None
    frames = 0
    iterator = driver.stream(synthesis, cancel=token)
    deadline = asyncio.get_running_loop().time() + limits.request_timeout_s
    try:
        while True:
            # Never keep asyncio.timeout open across a yield: the first frame
            # and response body can be pulled by different ASGI tasks.
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("speech request deadline exceeded")
            try:
                async with asyncio.timeout(remaining):
                    frame = await anext(iterator)
            except StopAsyncIteration:
                break
            token.raise_if_cancelled()
            if format is None:
                format = frame.format
            elif frame.format != format:
                raise ValueError("backend changed audio format during synthesis")
            if len(frame.pcm) > 262144:
                raise ValueError("backend returned an oversized audio frame")
            total += frame.duration_s
            if total > limits.max_audio_seconds:
                raise ValueError("generated audio exceeds max_audio_seconds")
            frames += 1
            yield frame
        if frames == 0:
            raise RuntimeError("backend returned no audio")
    finally:
        token.cancel()
        await iterator.aclose()


async def _until_disconnect(request, operation, token):
    async def invoke():
        return await operation

    async def watch():
        while True:
            if await request.is_disconnected():
                token.cancel()
                raise _Disconnected()
            await asyncio.sleep(0.05)

    task = asyncio.create_task(invoke())
    watcher = asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        await watcher
        raise AssertionError("disconnect watcher unexpectedly completed")
    finally:
        for pending in (task, watcher):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


def _http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, _Disconnected):
        return HTTPException(499, "client disconnected")
    if isinstance(exc, TimeoutError):
        return HTTPException(504, "speech synthesis timed out")
    # Request validation occurs before inference; runtime errors belong to the
    # backend and must not expose upstream URLs, credentials, or response bodies.
    logger.error("Speech backend failed", exc_info=exc)
    return HTTPException(502, "speech backend failed")


def create_app(driver, *, api_key: str | None = None, max_sessions: int = 8, limits: ServerLimits | None = None, close_driver: bool = True) -> FastAPI:
    """Create an application around one shared, concurrency-safe TTS driver.

    Authentication is optional for local use. Set ``api_key`` before exposing
    this application beyond a trusted interface. WebSockets use the same Bearer
    Authorization header as HTTP requests.
    """
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise ValueError("api_key must be a nonempty string or None")
    limits = limits or ServerLimits()
    admission = _Admission(max_sessions)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if close_driver:
                await driver.aclose()

    app = FastAPI(title="Audri", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "active_sessions": admission.active, "max_sessions": admission.maximum,
                "backend": type(driver).__name__, "backend_readiness_checked": False}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        if not _authorized(request.headers.get("authorization"), api_key):
            raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
        try:
            async with asyncio.timeout(limits.request_timeout_s):
                payload = await _json_body(request, limits.max_request_bytes)
        except TimeoutError as exc:
            raise HTTPException(408, "request body timed out") from exc
        text = payload.pop("input", None)
        response_format = payload.pop("response_format", "wav")
        if response_format not in ("wav", "pcm"):
            raise HTTPException(422, "response_format must be wav or pcm")
        try:
            synthesis = _synthesis_request(payload, text, limits)
            if not synthesis.text.strip():
                raise ValueError("input must not be blank")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if not await admission.enter():
            raise HTTPException(429, "all speech sessions are occupied", headers={"Retry-After": "1"})

        token = CancellationToken()
        iterator = _audio_frames(driver, synthesis, token, limits)
        temporary = None
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            token.cancel()
            try:
                async with asyncio.timeout(5):
                    await iterator.aclose()
            finally:
                if temporary is not None:
                    temporary.close()
                await admission.leave()

        try:
            first = await _until_disconnect(request, anext(iterator), token)
            format_headers = {
                "X-Audio-Sample-Rate": str(first.format.sample_rate),
                "X-Audio-Channels": str(first.format.channels),
                "X-Audio-Sample-Width": str(first.format.sample_width),
                "X-Audio-Encoding": "pcm_s16le",
                "Cache-Control": "no-store",
            }
            if response_format == "pcm":
                async def pcm_body():
                    yield first.pcm
                    async for frame in iterator:
                        yield frame.pcm

                return _OwnedStreamingResponse(pcm_body(), cleanup=cleanup, send_timeout_s=limits.send_timeout_s, media_type="application/octet-stream", headers=format_headers)

            temporary = tempfile.TemporaryFile(mode="w+b")

            async def render_wav():
                with wave.open(temporary, "wb") as output:
                    output.setnchannels(first.format.channels)
                    output.setsampwidth(first.format.sample_width)
                    output.setframerate(first.format.sample_rate)
                    output.writeframesraw(first.pcm)
                    async for frame in iterator:
                        output.writeframesraw(frame.pcm)

            await _until_disconnect(request, render_wav(), token)
            length = temporary.tell()
            temporary.seek(0)
            format_headers["Content-Length"] = str(length)

            async def wav_body():
                while chunk := await asyncio.to_thread(temporary.read, 65536):
                    yield chunk

            return _OwnedStreamingResponse(wav_body(), cleanup=cleanup, send_timeout_s=limits.send_timeout_s, media_type="audio/wav", headers=format_headers)
        except BaseException as exc:
            await cleanup()
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, Exception):
                raise _http_exception(exc) from exc
            raise

    @app.websocket("/v1/speech/stream")
    async def stream(websocket: WebSocket):
        if not _authorized(websocket.headers.get("authorization"), api_key):
            await websocket.close(code=1008, reason="authentication required")
            return
        if not await admission.enter():
            await websocket.close(code=1013, reason="all speech sessions are occupied")
            return
        tool = SpeakTool(driver, max_sessions=1, close_driver=False, session_config=SessionConfig(
            max_delta_chars=limits.max_delta_chars,
            max_text_chars=limits.max_text_chars,
            max_audio_seconds=limits.max_audio_seconds,
            synthesis_timeout_s=limits.request_timeout_s,
        ))
        send_lock = asyncio.Lock()
        current: dict[str, Any] | None = None

        async def send_json(message):
            async with send_lock:
                await asyncio.wait_for(websocket.send_json(message), limits.send_timeout_s)

        async def stop_current():
            nonlocal current
            previous, current = current, None
            if previous is None:
                return None
            previous["task"].cancel()
            await asyncio.gather(previous["task"], return_exceptions=True)
            await tool.stop(previous["id"])
            return previous["id"]

        async def launch(message):
            nonlocal current
            if current is not None:
                raise ValueError("a session is already active; send stop before starting another")
            identifier = message.pop("session_id", str(uuid.uuid4()))
            if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
                raise ValueError("session_id must be a nonempty string of at most 128 characters")
            template = _synthesis_request(message, "", limits)
            queue: asyncio.Queue[str | None] = asyncio.Queue(limits.websocket_pending_deltas)
            record: dict[str, Any] = {"id": identifier, "queue": queue, "ending": False, "chars": 0}
            audio_format = None

            async def emit_frame(frame):
                nonlocal audio_format
                if current is not record:
                    return
                async with send_lock:
                    if audio_format is None:
                        audio_format = frame.format
                        await asyncio.wait_for(websocket.send_json({"type": "format", "session_id": identifier, **_format_dict(frame.format)}), limits.send_timeout_s)
                    await asyncio.wait_for(websocket.send_bytes(frame.pcm), limits.send_timeout_s)

            session = await tool.start(identifier, request=template, egress=CallbackAudioEgress(emit_frame))

            async def feed():
                while True:
                    delta = await queue.get()
                    if delta is None:
                        return await tool.end(identifier)
                    await tool.delta(identifier, delta)

            async def supervise():
                nonlocal current
                feeder = asyncio.create_task(feed())
                waiter = asyncio.create_task(session.wait())
                outcome = None
                deliver = False
                try:
                    done, _ = await asyncio.wait((feeder, waiter), return_when=asyncio.FIRST_COMPLETED)
                    if waiter in done:
                        # Surface a background failure even if no further
                        # deltas arrive. Normal completion also waits for
                        # SpeakTool.end to remove the session from its map.
                        waiter.result()
                    result = await feeder
                    outcome = {"type": "ended", "session_id": identifier, "metrics": result.as_dict()}
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("WebSocket speech session failed")
                    outcome = {"type": "error", "session_id": identifier, "message": "speech session failed"}
                finally:
                    for task in (feeder, waiter):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(feeder, waiter, return_exceptions=True)
                    await tool.stop(identifier)
                    if current is record:
                        deliver = True
                        current = None
                if deliver and outcome is not None:
                    with contextlib.suppress(Exception):
                        await send_json(outcome)

            current = record
            record["task"] = asyncio.create_task(supervise())
            await send_json({"type": "started", "session_id": identifier, "max_delta_chars": limits.max_delta_chars,
                             "pending_delta_capacity": limits.websocket_pending_deltas})

        try:
            await websocket.accept()
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), limits.websocket_idle_timeout_s)
                if len(raw.encode("utf-8")) > limits.max_request_bytes:
                    await send_json({"type": "error", "message": "WebSocket message is too large"})
                    await websocket.close(code=1009)
                    break
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError("message must be a JSON object")
                    kind = message.pop("type", None)
                    if kind == "start":
                        await launch(message)
                    elif kind == "delta":
                        if set(message) != {"text"} or not isinstance(message["text"], str):
                            raise ValueError("delta requires exactly one text string")
                        if current is None or current["ending"]:
                            raise ValueError("delta requires an active session that has not ended")
                        text = message["text"]
                        if len(text) > limits.max_delta_chars:
                            raise ValueError("delta exceeds max_delta_chars")
                        if current["chars"] + len(text) > limits.max_text_chars:
                            raise ValueError("session text exceeds max_text_chars")
                        current["queue"].put_nowait(text)
                        current["chars"] += len(text)
                    elif kind == "end":
                        if message:
                            raise ValueError("end accepts no additional fields")
                        if current is None or current["ending"]:
                            raise ValueError("end requires an active session that has not ended")
                        current["queue"].put_nowait(None)
                        current["ending"] = True
                    elif kind == "stop":
                        if message:
                            raise ValueError("stop accepts no additional fields")
                        identifier = await stop_current()
                        await send_json({"type": "stopped", "session_id": identifier})
                    elif kind == "ping":
                        if message:
                            raise ValueError("ping accepts no additional fields")
                        await send_json({"type": "pong"})
                    else:
                        raise ValueError("type must be start, delta, end, stop, or ping")
                except asyncio.QueueFull:
                    identifier = await stop_current()
                    await send_json({"type": "error", "session_id": identifier,
                                     "message": "input queue is full; session stopped; send deltas more slowly"})
                except (ValueError, TypeError) as exc:
                    await send_json({"type": "error", "message": str(exc)})
        except TimeoutError:
            with contextlib.suppress(Exception):
                await websocket.close(code=1001, reason="idle or send timeout")
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            try:
                await stop_current()
                await tool.aclose()
            finally:
                await admission.leave()

    return app
