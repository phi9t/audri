"""TTS Driver abstraction + implementations.

Path B (HF local):  HFQwen3TTSDriver  — loads Qwen3-TTS via transformers
Path A (vLLM-Omni): VLLMOmniTTSDriver — stub, raises NotImplementedError

Both yield raw PCM bytes: int16, 24 kHz, mono, in ~20ms frames (480 samples).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from threading import Event
from typing import AsyncIterator, Optional

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# PCM constants
SAMPLE_RATE = 24_000
FRAME_SAMPLES = 480          # 20ms @ 24kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 → 2 bytes per sample

class TTSDriver(ABC):
    """Abstract TTS driver interface."""

    @abstractmethod
    async def synth(
        self,
        text: str,
        voice: str = "default",
        language: str = "English",
        instruction: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """Synthesise *text* and yield raw PCM frames (int16, 24kHz, mono, 480 samples each)."""
        ...  # pragma: no cover

    def cancel(self) -> None:
        """Best-effort cancellation hook for in-flight synthesis."""
        return


_DEFAULT_VOICE_DESIGN_INSTRUCT = "Speak in a natural, clear, and expressive voice."


class Qwen3TTSDriver(TTSDriver):
    """Qwen3-TTS driver using the upstream qwen-tts package.

    Supports all three Qwen3-TTS model types:
      - custom_voice: generate_custom_voice()  (preset speakers, e.g. 12Hz-1.7B-CustomVoice)
      - voice_design: generate_voice_design()  (natural-language style, e.g. 12Hz-1.7B-VoiceDesign)
      - base:         generate_voice_clone()   (reference audio cloning)

    The model type is detected at load time from model.tts_model_type.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        speaker: str = "Default",
        instruction: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = "sdpa",
    ) -> None:
        self.model_name = model_name
        self.speaker = speaker
        self.instruction = instruction
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.sample_rate = SAMPLE_RATE
        self._model: Optional[Qwen3TTSModel] = None
        self._lock = asyncio.Lock()
        self._cancel_event = Event()

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._model is not None:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        logger.info("Loading %s …", self.model_name)
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map="cuda:0",
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )

    async def synth(
        self,
        text: str,
        voice: str = "default",
        language: str = "English",
        instruct: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        await self._ensure_loaded()
        self._cancel_event.clear()

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def _infer() -> None:
            try:
                waveform, sr = self._run_inference(text, voice, language, instruct)
                if self._cancel_event.is_set():
                    return
                self.sample_rate = sr or self.sample_rate
                pcm_int16 = self._to_int16(waveform)
                for start in range(0, len(pcm_int16), FRAME_SAMPLES):
                    if self._cancel_event.is_set():
                        break
                    frame = pcm_int16[start : start + FRAME_SAMPLES]
                    if len(frame) < FRAME_SAMPLES:
                        frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
                    loop.call_soon_threadsafe(queue.put_nowait, frame.tobytes())
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        infer_future = loop.run_in_executor(None, _infer)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                logger.error("TTS inference failed: %s", item, exc_info=True)
                raise RuntimeError("TTS inference failed") from item
            yield item

        await infer_future

    def cancel(self) -> None:
        self._cancel_event.set()

    def _run_inference(
        self, text: str, voice: str, language: str, instruct: Optional[str]
    ) -> tuple[np.ndarray, int]:
        instruction = instruct if instruct is not None else self.instruction
        model_type = getattr(self._model.model, "tts_model_type", "custom_voice")

        if model_type == "voice_design":
            # VoiceDesign: instruct is a required natural-language voice description
            design = instruction or _DEFAULT_VOICE_DESIGN_INSTRUCT
            result = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=design,
            )
        else:
            # custom_voice (default): preset speaker with optional instruction
            speaker = self.speaker if voice == "default" else voice
            supported = self._model.get_supported_speakers()
            if supported and speaker.lower() not in {s.lower() for s in supported}:
                speaker = supported[0]
                logger.warning("Speaker %r not supported; falling back to %r", self.speaker, speaker)
            result = self._model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruction,
            )

        waveform_list, sample_rate = result
        waveform = np.asarray(waveform_list[0], dtype=np.float32).squeeze()
        return waveform, sample_rate

    @staticmethod
    def _to_int16(waveform: np.ndarray) -> np.ndarray:
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        return (waveform * 32767).astype(np.int16)


class VLLMOmniTTSDriver(TTSDriver):
    """Path A stub — vLLM-Omni HTTP streaming client.

    Calls ``/v1/audio/speech`` on a running vLLM-Omni server and streams
    back PCM bytes.  Not implemented yet.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url

    async def synth(
        self,
        text: str,
        voice: str = "default",
        language: str = "English",
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "VLLMOmniTTSDriver is not implemented yet. "
            "Use HFQwen3TTSDriver (Path B) for local inference."
        )
        # Unreachable — satisfies type checker
        yield b""  # type: ignore[misc]
