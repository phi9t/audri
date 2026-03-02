# audri

Streaming text-to-speech pipeline for conversational AI.

Connects an LLM token stream to local audio synthesis chunk-by-chunk, so the
first audio frame plays within seconds of the model starting to respond —
rather than waiting for the full response.

```
LLM tokens  →  TextChunker  →  Qwen3TTSDriver  →  PCM frames
                (boundaries)    (inference)         (to speaker / WAV)
```

See [foundation.org](foundation.org) for the full system design.

## Requirements

- NVIDIA GPU (CUDA 12.x)
- Docker
- [`just`](https://github.com/casey/just)

The runtime uses [Spack](https://spack.io)-built PyTorch/torchaudio baked into
the container image. All Python code runs inside the container — never on the
host directly.

## Quick start

```bash
# 1. Verify the container infra is intact
just zephyr-check

# 2. Build the qwen3-tts inference environment (first time only, ~5 min)
just build-env

# 3. Smoke test — model load + WAV synthesis → outputs/smoke_test.wav
just smoke

# 4. Streaming demo — simulated LLM token stream → per-chunk WAVs
just demo
```

## Models

Inference uses [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models from
HuggingFace. Models are downloaded automatically on first run and cached
locally. No HF token required for public model variants.

| Model | Type | Size |
|-------|------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice design (NL style prompt) | 1.7B |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Preset speakers | 0.6B |

Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.

To run with a different model:

```bash
just smoke model=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
just demo  model=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

## Python API

```python
from audri.speak_tool import SpeakTool
from audri.tts.driver import Qwen3TTSDriver

driver = Qwen3TTSDriver(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
tool = SpeakTool(driver=driver)

await tool.start("session-1", voice="default", language="English")
async for token in llm_stream:
    await tool.delta("session-1", token)
await tool.end("session-1")

# Barge-in (cancel mid-utterance):
await tool.stop("session-1")
```

Raw synthesis without the session API:

```python
async for pcm_frame in driver.synth("Hello, world.", language="English"):
    # pcm_frame: bytes — int16, 24 kHz, mono, 480 samples (20 ms)
    ...
```

## Just recipes

```
just zephyr-check        Verify the container infra is intact
just zephyr-config       Show effective image/project config
just zephyr-policy-check Enforce no-direct-pip-install policy
just gpu-check           GPU info (name, CC, VRAM)
just build-env           Build the qwen3-tts venv (with validation)
just build-env-fast      Build the qwen3-tts venv (skip validation)
just image-build         Build the audri-tts container image
just layer "pkg"         Add packages to the qwen3-tts venv
just smoke               Milestone 0 — model load + WAV synthesis
just demo                Milestone 1 — streaming demo (headless WAVs)
just run "cmd"           Run an arbitrary command in the container
```

## Architecture

Three async stages, connected by queues, running concurrently per session:

1. **TextChunker** — accumulates LLM token deltas into natural-boundary chunks
   (min 48 chars, punctuation boundaries, 220 ms timeout fallback)
2. **Qwen3TTSDriver** — loads the model once, synthesises each chunk in a
   thread executor, streams 20 ms PCM frames as they are ready
3. **LocalAudioEgress** — writes frames to a sounddevice stream (or WAV in
   headless mode)

Session lifecycle: `start → delta* → end` with `stop` for barge-in cancel at
any point.

## Hardware notes

Tested on 2× NVIDIA GeForce GTX 1070 Ti (Pascal, 8 GB each). Pascal has no
tensor cores and no native bfloat16, so inference runs at fp32 throughput with
bfloat16 storage — RTF ≈ 2.2× on this hardware. RTF < 1.0 (real-time) is
achievable on Turing/Ampere or newer.

## Infrastructure

The repo vendors two hermetic container kits:

| Kit | Path | Role |
|-----|------|------|
| Zephyr container infra | `.sygaldry/zephyr/` | Container runner (`repoctl`) |
| MLSys overlay | `.codex-zephyr-mlsys/` | UV env builder on Spack base |

See [AGENTS.md](AGENTS.md) for detailed infrastructure documentation and rules
for AI agents working in this repo.

## License

MIT — see [LICENSE](LICENSE).
