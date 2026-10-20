"""A streaming client for the self-hosted Fish Speech native HTTP server.

The heavy model runtime lives in a separate service. Importing this module does
not import torch, download weights, or require httpx until a connection is used.
"""

from __future__ import annotations

import asyncio
import base64
import math
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from audri.audio import AudioFrame, PCMFramer
from audri.tts.driver import DriverCapabilities, TTSDriver
from audri.tts.types import CancellationToken, SynthesisRequest
from audri.wav import WavFormatError, WavStreamParser

FISH_SPEECH_COMMIT = "541566932abbaebcb0a40e562a1f846738383f38"
FISH_MODELS = {
    "s2-pro": {
        "repository": "fishaudio/s2-pro",
        "license": "Fish Audio Research License",
        "recommended_gpu_memory_gb": 24,
    },
    "drama-3": {
        "repository": "fishaudio/drama-3",
        "license": "Fish Audio Community License 1.0",
        "minimum_fp8_gpu_memory_gb": 48,
    },
}
FISH_CODE_LICENSE = "Fish Audio Community License 1.0"


class FishSpeechError(RuntimeError):
    """The Fish service rejected a request or returned an invalid response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FishSpeechDriver(TTSDriver):
    """Stream validated PCM frames from a native Fish Speech HTTP service.

    A service instance loads one checkpoint at startup. ``voice`` selects a
    saved reference ID, not a checkpoint; inline references and ``reference_id``
    are also supported. A default voice with no references lets Fish choose a
    voice. Language is inferred by Fish from the text.

    ``timeout`` is the HTTP transport's idle timeout. Backpressure is bounded to
    two queued frames plus the current network chunk. Cancelling a token or
    closing the async iterator cancels the local request and closes its HTTP
    response. The native Fish server does not expose a GPU abort operation.

    An injected client remains owned by its caller. An internally created
    client is closed by ``aclose``. There are no automatic synthesis retries.
    """

    capabilities = DriverCapabilities(
        "fish", native_streaming=True, voice_references=True,
        instructions=True, cancellation="transport",
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str | None = None,
        timeout: float = 60,
        max_concurrency: int = 1,
        client: Any = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) service URL without credentials or query")
        # Accessing port also validates malformed/non-numeric port strings.
        _ = parsed.port
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
        self.timeout = timeout
        self._headers = {"Accept": "audio/wav"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = client
        self._owns_client = client is None
        self._slots = asyncio.Semaphore(max_concurrency)
        self._streams: set[asyncio.Task[None]] = set()
        self._closed = False

    def _get_client(self) -> Any:
        if self._closed:
            raise RuntimeError("FishSpeechDriver is closed")
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError(
                    "Fish Speech requires the optional httpx dependency; use the audri fish extra"
                ) from exc
            self._client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False, trust_env=False
            )
        return self._client

    async def health(self) -> dict[str, Any]:
        """Return the native server's health result; HTTP errors are raised."""
        return await self._get_json("/v1/health")

    async def defaults(self) -> dict[str, Any]:
        """Return the actual loaded model family and its sampling defaults."""
        return await self._get_json("/v1/tts/defaults")

    async def _get_json(self, path: str) -> dict[str, Any]:
        response = await self._get_client().get(
            self.base_url + path,
            headers={**self._headers, "Accept": "application/json"},
            params={"format": "json"},
            timeout=self.timeout,
        )
        self._check_status(response)
        try:
            result = response.json()
        except (ValueError, TypeError) as exc:
            raise FishSpeechError("Fish service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise FishSpeechError("Fish service metadata must be a JSON object")
        return result

    @staticmethod
    def _check_status(response: Any) -> None:
        if not 200 <= response.status_code < 300:
            raise FishSpeechError(
                f"Fish Speech returned HTTP {response.status_code}",
                status_code=response.status_code,
            )

    @staticmethod
    def _payload(request: SynthesisRequest) -> dict[str, Any]:
        text = request.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech text must not be empty")
        for name in ("temperature", "top_p"):
            value = getattr(request, name)
            if value is not None and not 0.1 <= value <= 1.0:
                raise ValueError(f"Fish {name} must be between 0.1 and 1.0")
        if not 0.9 <= request.repetition_penalty <= 2.0:
            raise ValueError("Fish repetition_penalty must be between 0.9 and 2.0")
        if request.instruction:
            text = f"[{request.instruction}] {text}"
        reference_id = request.reference_id
        if request.voice not in (None, "", "default"):
            if reference_id is not None and reference_id != request.voice:
                raise ValueError("voice and reference_id select different Fish references")
            reference_id = request.voice
        if reference_id is not None and request.references:
            raise ValueError("Choose a saved reference ID or inline references, not both")
        if reference_id is not None and (
            not isinstance(reference_id, str)
            or len(reference_id) > 255
            or not re.fullmatch(r"[a-zA-Z0-9_ -]+", reference_id)
            or not reference_id.strip()
        ):
            raise ValueError("Invalid Fish reference ID")
        references = []
        for reference in request.references:
            audio = base64.b64encode(reference.audio).decode("ascii")
            # The native server only decodes JSON base64 strings longer than
            # 255 characters. Real voice clips exceed this comfortably.
            if len(audio) <= 255:
                raise ValueError("Fish JSON references must contain at least 192 audio bytes")
            item: dict[str, Any] = {"audio": audio, "text": reference.text}
            if reference.speaker is not None:
                item["speaker"] = reference.speaker
            references.append(item)
        payload: dict[str, Any] = {
            "text": text,
            "format": "wav",
            "streaming": True,
            "references": references,
            "use_memory_cache": "on",
            "normalize": request.normalize,
            "single_chunk": request.single_chunk,
            "chunk_length": request.chunk_length,
        }
        if reference_id is not None:
            payload["reference_id"] = reference_id
        for name in (
            "seed", "max_new_tokens", "temperature", "top_p", "repetition_penalty"
        ):
            value = getattr(request, name)
            if value is not None:
                payload[name] = value
        return payload

    async def _pump(
        self, payload: dict[str, Any], output: asyncio.Queue[AudioFrame | Exception]
    ) -> None:
        try:
            async with self._slots:
                # Upstream creates an empty directory and falls back to a random
                # voice for unknown IDs. Refuse that silent timbre change.
                if "reference_id" in payload:
                    saved = await self._get_json("/v1/references/list")
                    ids = saved.get("reference_ids")
                    if not isinstance(ids, list) or payload["reference_id"] not in ids:
                        raise FishSpeechError("The requested Fish reference ID is not registered")
                parser = WavStreamParser()
                framer: PCMFramer | None = None
                async with self._get_client().stream(
                    "POST",
                    self.base_url + "/v1/tts",
                    json=payload,
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    self._check_status(response)
                    # A fixed httpx chunk_size would buffer early audio until
                    # that byte count is reached, adding avoidable latency.
                    async for chunk in response.aiter_bytes():
                        pcm = parser.feed(chunk)
                        if parser.format is not None and framer is None:
                            framer = PCMFramer(parser.format, frame_ms=20)
                        if pcm:
                            assert framer is not None
                            for frame in framer.feed(pcm):
                                await output.put(frame)
                    parser.finish()
                    assert framer is not None
                    for frame in framer.finish():
                        await output.put(frame)
        except asyncio.CancelledError:
            raise
        except WavFormatError as exc:
            await output.put(FishSpeechError(f"Invalid Fish audio response: {exc}"))
        except Exception as exc:
            await output.put(exc)

    async def stream(
        self, request: SynthesisRequest, *, cancel: CancellationToken | None = None
    ) -> AsyncIterator[AudioFrame]:
        if self._closed:
            raise RuntimeError("FishSpeechDriver is closed")
        token = cancel if cancel is not None else CancellationToken()
        token.raise_if_cancelled()
        if not request.text.strip():
            return
        payload = self._payload(request)
        output: asyncio.Queue[AudioFrame | Exception] = asyncio.Queue(maxsize=2)
        pump = asyncio.create_task(self._pump(payload, output))
        self._streams.add(pump)

        async def stop_on_cancel() -> None:
            await token.wait()
            pump.cancel()

        cancellation = asyncio.create_task(stop_on_cancel())
        receive: asyncio.Task[AudioFrame | Exception] | None = None
        try:
            while True:
                if receive is None:
                    receive = asyncio.create_task(output.get())
                done, _ = await asyncio.wait(
                    {pump, cancellation, receive}, return_when=asyncio.FIRST_COMPLETED
                )
                token.raise_if_cancelled()
                if receive in done:
                    item = receive.result()
                    receive = None
                    if isinstance(item, Exception):
                        raise item
                    yield item
                elif pump in done:
                    if pump.cancelled():
                        raise asyncio.CancelledError
                    error = pump.exception()
                    if error is not None:
                        raise error
                    if output.empty():
                        break
        finally:
            tasks = [pump, cancellation]
            if receive is not None:
                tasks.append(receive)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._streams.discard(pump)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = list(self._streams)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._owns_client and self._client is not None:
            await self._client.aclose()
