# Fish Speech integration

Audri connects to the **native self-hosted Fish Speech HTTP server**. The model,
codec, and CUDA environment stay in that service; the Audri client needs only
`httpx`. This avoids importing or reinstalling Fish's Torch dependencies into
Audri's existing Zephyr/Spack environment.

Built with Fish Audio. Audri's adapter is independently implemented against the
HTTP protocol. No Fish model weights or inference source are bundled here.

## Compatibility baseline

The adapter was implemented against Fish Speech commit
[`541566932abbaebcb0a40e562a1f846738383f38`](https://github.com/fishaudio/fish-speech/tree/541566932abbaebcb0a40e562a1f846738383f38).
Keep the source revision and exact model revision in your deployment record.

| Server checkpoint | Model family | Upstream hardware guidance | Model license |
| --- | --- | --- | --- |
| `fishaudio/s2-pro` | `s2` | 24 GB GPU memory; CPU inference also supported | Fish Audio Research License |
| `fishaudio/drama-3` | `drama3` | At least 48 GB with FP8; NVIDIA GPU only; no `--compile` | Fish Audio Community License 1.0 |

The **server selects the checkpoint at startup**. A request's `voice` or
`reference_id` selects a reference voice, not a model. `audri doctor` reads the
loaded family from `/v1/tts/defaults`. Each additional native server worker loads
another copy of the models. Start with one worker per model deployment.

These requirements come from the pinned upstream
[installation guide](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/docs/en/install.md)
and [model manager](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/tools/server/model_manager.py).
They are deployment guidance, not a performance measurement made by Audri.

## Container-only quick start

Run these shell commands on a Linux machine with Docker and the NVIDIA container
runtime. Python and dependency installation remain inside containers, following
`AGENTS.md`. The GPU model service and the CPU Audri client are separate images.

Build a pinned native Fish server in a sibling checkout:

```bash
git clone https://github.com/fishaudio/fish-speech.git ../fish-speech
git -C ../fish-speech checkout 541566932abbaebcb0a40e562a1f846738383f38
docker build -f ../fish-speech/docker/Dockerfile --target server \
  -t audri-fish-native:5415669 ../fish-speech
```

Download the S2-Pro checkpoint using the `hf` executable already installed in
that image. This is an explicit multi-gigabyte model download; the Audri client
never downloads a checkpoint on import or connection.

```bash
mkdir -p checkpoints
docker run --rm --user "$(id -u):$(id -g)" \
  -e HF_HOME=/app/checkpoints/.hf-cache \
  -v "$PWD/checkpoints:/app/checkpoints" \
  --entrypoint /app/.venv/bin/hf audri-fish-native:5415669 \
  download fishaudio/s2-pro --local-dir /app/checkpoints/s2-pro
```

For repeatable deployments, supply `hf download --revision <model-commit>` and
retain the accompanying license. Then start the native service:

```bash
docker run --rm --name audri-fish --gpus device=0 \
  -p 127.0.0.1:8080:8080 \
  -v "$PWD/checkpoints:/app/checkpoints:ro" \
  --mount type=volume,source=audri-fish-references,target=/app/references \
  -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
  -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
  audri-fish-native:5415669
```

The named reference volume preserves uploaded voices across server restarts.
For Drama 3, download `fishaudio/drama-3`, set
`LLAMA_CHECKPOINT_PATH=checkpoints/drama-3`, and leave compile disabled. If the
Drama checkpoint omits `codec.pth`, keep the sibling S2-Pro codec as described
upstream. The local service must finish model loading and warmup before requests
can succeed.

In another terminal, from the Audri repository:

```bash
docker build -f docker/Dockerfile.client --target runtime -t audri-client .
docker run --rm --network host audri-client doctor --url http://127.0.0.1:8080
mkdir -p outputs
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs:/outputs" audri-client \
  speak 'Hello. This is a complete speech request through Audri.' \
  --backend fish --url http://127.0.0.1:8080 --output /outputs/hello.wav
```

`--network host` in these examples is for Linux. When Fish runs on another
machine, set `--url` to its reachable address. The URL is the **server root**,
such as `http://host:8080`, without `/v1/tts` appended. If the native server uses
`--api-key`, pass the matching `FISH_API_KEY` environment variable to the Audri
container. Keep credentials out of command-line URLs.

For the existing Zephyr workflow, add the lightweight HTTP client through the
approved layering tool and run the CLI inside the container:

```bash
just layer 'httpx>=0.28.1,<0.29'
just run 'PYTHONPATH=/workspace/audri/src python -m audri doctor --url http://127.0.0.1:8080'
```

## Voice references and text control

Use one of these reference forms for each request:

| Input | Meaning |
| --- | --- |
| No reference and `voice="default"` | Let the model select a voice |
| `reference_id="narrator"` or `voice="narrator"` | Use a voice already registered on that Fish server |
| `references=(VoiceReference(audio=..., text=..., speaker=...),)` | Send encoded reference audio bytes and the matching transcript |

Audri verifies saved IDs through `/v1/references/list` before synthesis. This
avoids the native server's behavior of creating an empty reference directory and
silently generating an unconditioned voice for an unknown ID. Prefer inline
references when you need the request to carry its complete voice inputs.

For example, register your own reference voice using the native server's
multipart API:

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/references/add \
  -H 'Accept: application/json' \
  -F id=narrator \
  -F 'audio=@reference.wav;type=audio/wav' \
  -F 'text=The exact words spoken in the reference recording.'
```

Or mount the reference into the Audri client container and provide
`--reference-audio /references/reference.wav --reference-text 'The transcript'`
to `audri speak`. Only use a person's reference voice with their permission or
another applicable right, as required by the model's license.

Inline references are encoded as base64 in JSON; no remote filesystem paths are
sent. Audri rejects clips smaller than 192 bytes because the pinned native
server's JSON decoder only recognizes base64 strings longer than 255 characters.
Real speech references exceed this size comfortably.

`instruction="whisper"` becomes an inline `[whisper]` tag before the text. Native
speaker tags such as `<|speaker:3|>` remain intact. A reference's `speaker` value
must be 0–15. Drama 3 uses explicit speaker tags; S2-Pro also depends on reference
ordering. `language` is framework metadata: this native API does not expose a
language parameter and infers language from the text. The adapter therefore does
not pretend that `language=` forces a locale.

Sampling controls are sent explicitly from `SynthesisRequest`. Fish accepts
`temperature` and `top_p` from 0.1 to 1.0. Current S2-Pro and Drama 3 ignore
`repetition_penalty` in favor of their internal repetition-aware sampling.
`single_chunk` affects Drama 3; `chunk_length` affects S2-Pro. Audri's common
request currently restricts chunk length to 100–300 and token limits to positive
values; it does not expose Fish's special `max_new_tokens=0` behavior.

## Streaming and cancellation contract

The pinned native route accepts JSON and MessagePack. Audri sends JSON to
`POST /v1/tts` with `streaming=true` and `format="wav"`. Native streaming rejects
other output formats. This adapter does not target Fish's hosted cloud API,
SGLang-Omni, or vLLM-Omni; those deployments require adapters for their actual
wire protocols.

The response starts with a WAV header whose RIFF and data sizes are
`0xffffffff`, followed by signed 16-bit PCM. The current codec is normally
44,100 Hz mono. Audri reads the header's actual rate and channels, validates the
format, and reframes samples without resampling or discarding bytes. Transport
chunks can split the header or an individual sample. Ancillary WAV chunks are
handled within a bounded metadata budget; compressed/float WAV, inconsistent
metadata, truncated finite files, empty audio, and partial final samples fail
explicitly.

Never save the upstream stream unchanged as a completed WAV file: its length
fields deliberately do not describe a finite file. Audri's WAV sink writes a
proper finalized container. Frames carry their format, so downstream code does
not accidentally play 44.1 kHz Fish audio as 24 kHz Qwen audio.

Each driver limits concurrent HTTP requests with a semaphore. Each stream has a
queue of two frames, plus the current network chunk, so a slow consumer applies
backpressure. The HTTP parser does not request a fixed `httpx` byte chunk size,
which would delay the first small response until that buffer filled. Timeout is
an HTTP transport idle timeout, not a promise that a long generation completes
within that many wall-clock seconds.

Use `contextlib.aclosing(driver.stream(...))` when consuming a stream directly,
or use Audri's session orchestration. Cancelling its `CancellationToken` closes
the local HTTP request even when the consumer is paused. Closing a stream early
also releases its concurrency slot. Closing a driver cancels its active streams
and closes only the HTTP client it owns. An injected client remains owned by its
caller. Failed synthesis is never automatically replayed after emitting audio.

**Transport cancellation does not establish GPU cancellation.** The pinned
native Fish service has no abort endpoint, iterates a synchronous model
generator in its async route, and retains/concatenates generated segments. Audri
does not claim to fix those server-side scheduling and memory behaviors. Use a
separately verified serving adapter and benchmark for higher concurrency; do not
transfer upstream H200/SGLang latency numbers to the native service.

Drama 3 advertises editing and multi-turn revision, but this native
`ServeTTSRequest` has no history, infill, or previous-take fields. The adapter
exposes the synthesis and reference features actually present in the API, and
does not claim an editing feature merely because the underlying model has it.

## Verification

`tests/test_wav.py` checks arbitrary network fragmentation, sample integrity,
stereo metadata, ancillary chunks, unsupported formats, truncation, and metadata
bounds. `tests/test_fish.py` checks request mapping, actual format propagation,
HTTP/transport errors, no implicit retries, missing voices, cancellation during
network waits and paused consumption, early close, concurrency-slot cleanup,
metadata endpoints, and the real `httpx.MockTransport` interface. They use
protocol fixtures; no test pretends to generate speech without model weights.

Run the client tests inside the repository's container test target:

```bash
docker build -f docker/Dockerfile.client --target test -t audri-tests .
```

Actual model audio quality, GPU latency, peak VRAM, and sustained serving
concurrency require a live model server. The client benchmark reports measured
request latency and real-time factor from that deployment; it is not an audio
quality benchmark.

## Licensing and source references

The pinned Fish Speech **code** and Drama 3 use the
[Fish Audio Community License 1.0](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/LICENSE),
dated October 13, 2026. Its commercial grant is conditional, including a revenue
threshold and special restrictions on hosted speech-generation services.
S2-Pro's weights retain the separate
[Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md),
which requires a separate agreement for commercial use. These are available
weights with usage conditions, not an unrestricted commercial model license.
The license accompanying the exact downloaded revision governs that artifact.

Primary protocol sources at the compatibility commit:

- [Request schema](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/fish_speech/utils/schema.py)
- [Native routes](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/tools/server/views.py)
- [HTTP serialization and encoding](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/tools/server/api_utils.py)
- [Unknown-length WAV header](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/fish_speech/inference_engine/utils.py)
- [Reference loading](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/fish_speech/inference_engine/reference_loader.py)
- [Inference and model differences](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/docs/en/inference.md)
