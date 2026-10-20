"""Optional local Qwen3-TTS backend with serialized model ownership.

The upstream generate_* Python methods return a complete waveform. This
adapter streams frames after each committed text chunk finishes inference;
it does not describe waveform slicing as native incremental model decoding.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, nullcontext
import io
from typing import Any, TypeVar

from audri.audio import AudioFormat, AudioFrame, PCMFramer
from .driver import DriverCapabilities, TTSDriver
from .types import CancellationToken, SynthesisRequest

_T = TypeVar("_T")
_DEFAULT_INSTRUCTION = "Speak in a natural, clear, and expressive voice."


class Qwen3TTSDriver(TTSDriver):
    capabilities = DriverCapabilities("qwen3-tts", native_streaming=False,
                                       voice_references=True, instructions=True,
                                       cancellation="discard-output; model call completes")

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                 speaker: str = "Default", instruction: str | None = None,
                 dtype: Any = "bfloat16", attn_implementation: str | None = "sdpa",
                 device: str = "cuda:0", frame_ms: float = 20) -> None:
        self.model_name = model_name
        self.speaker = speaker
        self.instruction = instruction
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device = device
        self.frame_ms = frame_ms
        self.sample_rate = 24_000
        PCMFramer(AudioFormat(self.sample_rate), frame_ms)
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._pending: set[asyncio.Future[Any]] = set()
        self._closed = False

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("Qwen requires the qwen3-tts Zephyr environment; run just build-env") from exc
        dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name, device_map=self.device, dtype=dtype,
            attn_implementation=self.attn_implementation,
        )

    async def _ensure_loaded(self) -> None:
        """Retained for the original smoke-test entry point."""
        await self._run_serial(self._load_model, None)

    @staticmethod
    async def _wait(future: asyncio.Future[_T], cancel: CancellationToken | None) -> _T:
        if cancel is None:
            return await asyncio.shield(future)
        cancel.raise_if_cancelled()
        watcher = asyncio.create_task(cancel.wait())
        try:
            await asyncio.wait((future, watcher), return_when=asyncio.FIRST_COMPLETED)
            cancel.raise_if_cancelled()
            return future.result()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    async def _run_serial(self, function: Callable[[], _T],
                          cancel: CancellationToken | None) -> _T:
        if self._closed:
            raise RuntimeError("Qwen driver is closed")
        acquisition = asyncio.create_task(self._lock.acquire())
        try:
            await self._wait(acquisition, cancel)
        except BaseException:
            # Cancellation can race a successful lock acquisition.
            acquisition.cancel()
            await asyncio.gather(acquisition, return_exceptions=True)
            if not acquisition.cancelled() and acquisition.exception() is None and acquisition.result():
                self._lock.release()
            raise
        submitted = False
        try:
            if self._closed:
                raise RuntimeError("Qwen driver is closed")
            if cancel:
                cancel.raise_if_cancelled()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audri-qwen")
            work = asyncio.get_running_loop().run_in_executor(self._executor, function)
            self._pending.add(work)
            submitted = True

            def completed(future: asyncio.Future[Any]) -> None:
                self._pending.discard(future)
                self._lock.release()
                # Observe errors even when the requesting session has departed.
                if not future.cancelled():
                    future.exception()

            work.add_done_callback(completed)
            # Cancelling the coroutine never releases the model lock early.
            # The running native call still owns the model until it returns.
            return await self._wait(work, cancel)
        finally:
            if not submitted:
                self._lock.release()

    async def stream(self, request: SynthesisRequest, *, cancel: CancellationToken | None = None
                     ) -> AsyncIterator[AudioFrame]:
        if cancel:
            cancel.raise_if_cancelled()
        if not request.text.strip():
            return
        pcm, sample_rate = await self._run_serial(lambda: self._synthesize(request), cancel)
        if cancel:
            cancel.raise_if_cancelled()
        format = AudioFormat(sample_rate)
        self.sample_rate = sample_rate
        framer = PCMFramer(format, self.frame_ms)
        # Frame a bounded slice at a time; do not enqueue the entire waveform.
        for offset in range(0, len(pcm), framer.frame_bytes):
            if cancel:
                cancel.raise_if_cancelled()
            yield AudioFrame(pcm[offset:offset + framer.frame_bytes], format)
            await asyncio.sleep(0)

    async def synth(self, text: str, voice: str = "default", language: str = "English",
                    instruction: str | None = None, *, instruct: str | None = None
                    ) -> AsyncIterator[bytes]:
        # The old driver and smoke script disagreed on this keyword.
        if instruction is not None and instruct is not None:
            raise ValueError("provide instruction or instruct, not both")
        async with aclosing(super().synth(text, voice, language,
                                           instruction if instruction is not None else instruct)) as stream:
            async for pcm in stream:
                yield pcm

    def _synthesize(self, request: SynthesisRequest) -> tuple[bytes, int]:
        if request.reference_id is not None:
            raise ValueError("registered reference_id is a Fish service feature; Qwen requires inline audio")
        if request.single_chunk:
            raise ValueError("single_chunk is a Fish Drama 3 control")
        self._load_model()
        import torch

        model = self._model
        model_type = getattr(model.model, "tts_model_type", None)
        instruction = request.instruction if request.instruction is not None else self.instruction
        if request.references and model_type != "base":
            raise ValueError("Qwen reference audio requires a Base checkpoint")
        kwargs: dict[str, Any] = {
            "text": request.text, "language": request.language,
            "max_new_tokens": request.max_new_tokens, "temperature": request.temperature,
            "top_p": request.top_p, "repetition_penalty": request.repetition_penalty,
        }
        # fork_rng restores the RNG afterwards; use a dedicated model worker
        # process when reproducibility across unrelated models is important.
        devices = [int(self.device.partition(":")[2] or 0)] if self.device.startswith("cuda") else []
        rng = torch.random.fork_rng(devices=devices) if request.seed is not None else nullcontext()
        with rng, torch.inference_mode():
            if request.seed is not None:
                # torch.manual_seed also changes every other CUDA device.
                # Touch only generators whose states fork_rng saved here.
                torch.default_generator.manual_seed(request.seed)
                for device_index in devices:
                    torch.cuda.default_generators[device_index].manual_seed(request.seed)
            if model_type == "voice_design":
                if request.voice != "default":
                    raise ValueError("VoiceDesign uses instruction, not a preset voice ID")
                result = model.generate_voice_design(instruct=instruction or _DEFAULT_INSTRUCTION, **kwargs)
            elif model_type == "custom_voice":
                speaker = self.speaker if request.voice == "default" else request.voice
                supported = model.get_supported_speakers()
                if supported:
                    names = {s.lower(): s for s in supported}
                    if speaker.lower() == "default":
                        speaker = supported[0]
                    elif speaker.lower() in names:
                        speaker = names[speaker.lower()]
                    else:
                        raise ValueError(f"unknown Qwen speaker {speaker!r}; choices: {', '.join(supported)}")
                result = model.generate_custom_voice(speaker=speaker, instruct=instruction, **kwargs)
            elif model_type == "base":
                if len(request.references) != 1:
                    raise ValueError("Qwen Base requires exactly one inline VoiceReference")
                if request.voice != "default" or request.references[0].speaker is not None:
                    raise ValueError("Qwen Base takes one reference without a voice ID or speaker index")
                if instruction:
                    raise ValueError("Qwen Base does not support style instructions")
                import soundfile as sf
                reference = request.references[0]
                audio, rate = sf.read(io.BytesIO(reference.audio), dtype="float32", always_2d=True)
                result = model.generate_voice_clone(ref_audio=(audio.mean(axis=1), rate),
                                                     ref_text=reference.text, **kwargs)
            else:
                raise ValueError(f"unsupported Qwen model type: {model_type!r}")
        waveforms, rate = result
        if len(waveforms) != 1:
            raise ValueError("Qwen returned an unexpected batch size")
        return self._to_int16(waveforms[0]).tobytes(), int(rate)

    @staticmethod
    def _to_int16(waveform: Any) -> Any:
        import numpy as np
        values = np.asarray(waveform, dtype=np.float32).squeeze()
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Qwen must return nonempty, finite, mono audio")
        # Saturate instead of per-chunk peak normalization, which caused
        # loudness pumping and amplified background noise in the old driver.
        return np.rint(np.clip(values, -1.0, 1.0) * 32767).astype("<i2")

    async def aclose(self) -> None:
        self._closed = True
        if self._pending:
            await asyncio.shield(asyncio.gather(*tuple(self._pending), return_exceptions=True))
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._model = None
