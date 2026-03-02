# AGENTS.md

This file tells AI agents (Claude Code, Codex, etc.) how to work in this repo.

## Critical constraint: all code runs in the container

**Never run Python, pip, or uv directly on the host.** This is a GPU-only repo backed by a hermetic container. The Spack stack (PyTorch, JAX, torchaudio, etc.) only exists inside the image. Bare host invocations will fail with missing imports or wrong versions.

---

## Infrastructure layout

Two vendored kits live in the repo root:

| Kit                    | Path                   | Role                                   |
|------------------------|------------------------|----------------------------------------|
| Zephyr container infra | `.sygaldry/zephyr/`    | Container runner (`repoctl`, `jobctl`) |
| MLSys overlay runtime  | `.codex-zephyr-mlsys/` | UV env builder on top of Spack base    |

Both are managed by upstream scripts in sygaldry. **Do not edit them directly.**

Config files (read-only to agents):
- `.sygaldry/zephyr/infra.yaml` — digest-pinned image references, project ID, cache root
- `.codex-zephyr-mlsys/runtime.yaml` — digest-pinned snapshot ref for MLSys builds

Current image config (`image_mode: derived`):
```
image_ref:     ghcr.io/phi9t/sygaldry/zephyr:spack@sha256:8c9507...668c  (pinned Spack base)
runtime_image: ghcr.io/phi9t/sygaldry/zephyr:hf                           (local derived, used for repoctl)
```

---

## Running code

### Pattern 1 — one-off commands (`repoctl run`)

Use this for scripts, quick experiments, and anything that uses the `:hf` derived image:

```bash
# Run a Python script (repo is mounted at /workspace/audri inside the container)
.sygaldry/zephyr/bin/repoctl run -- python3 /workspace/audri/scripts/infer.py

# Inline Python
.sygaldry/zephyr/bin/repoctl run -- python3 -c "
import torch, torchaudio, jax
print(f'torch={torch.__version__} torchaudio={torchaudio.__version__} jax={jax.__version__} cuda={torch.cuda.is_available()}')
"

# Activate a previously layered venv and run
.sygaldry/zephyr/bin/repoctl run -- bash -c "
  source /workspace/audri/.venv-mlsys/hf-transformers/bin/activate
  python3 /workspace/audri/run.py
"

# Interactive shell
.sygaldry/zephyr/bin/repoctl shell
```

The container is `--rm` (ephemeral). The repo root is mounted at `/workspace/audri` (read-write).
The Spack view is at `/opt/spack_store/view/` and Python is `/opt/spack_store/view/bin/python3`.

### Pattern 2 — MLSys environment builds (`launch_mlsys_hf.sh`)

Use this to build and validate a structured UV env on top of the Spack base using `ghcr.io/phi9t/sygaldry/zephyr:hf-20260226` (contains the required runtime libs for uv layering). The env is defined in `.codex-zephyr-mlsys/envs/<name>.yaml`. This is primarily a **build + validate** operation; the resulting venv lives in the container at `/tmp/mlsys-envs/<env-name>/` (ephemeral) unless `MLSYS_VENV_ROOT` is overridden.

```bash
# Build + validate (recommended first-time setup)
./scripts/launch_mlsys_hf.sh qwen3-tts

# Build without validation (faster)
./scripts/launch_mlsys_hf.sh qwen3-tts --no-validate

# Persist the venv under the repo (survives container exit, gitignored via .venv*)
MLSYS_VENV_ROOT=/workspace/audri/.venv-mlsys ./scripts/launch_mlsys_hf.sh qwen3-tts

# Then use the persisted venv in a subsequent repoctl run
.sygaldry/zephyr/bin/repoctl run -- bash -c "
  source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate
  python3 /workspace/audri/scripts/infer.py
"
```

### just targets (project shortcuts)

```bash
just zephyr-check                    # structural health check (fast)
just zephyr-config                   # show effective image/project config
just zephyr-policy-check             # enforce no direct pip install patterns
just gpu-check                       # GPU info (name, CC, VRAM)
just build-env                       # build + validate qwen3-tts env
just build-env-fast                  # build only, skip validation
just layer "pkg1 pkg2"               # add extra uv-layered packages to qwen3-tts
just smoke                           # Milestone 0: model load + WAV synthesis
just demo                            # Milestone 1: streaming demo (headless WAVs)
just run "cmd"                       # arbitrary command in container with venv active
```

---

## Spack baked-in packages — do not reinstall these

The following are compiled into the container image at `/opt/spack_store/view/`. They are **not** installable via UV and must not appear in any env yaml `packages:` list or be `pip install`-ed:

```
torch==2.9.0        torchaudio==2.9.0      torchvision==0.24.0
jax==0.7.0          triton==3.4.0          flax (jax ecosystem)
numpy==2.3.4        scipy==1.16.3
numba==0.62.0rc2    llvmlite==0.45.0rc2
Python 3.13.8       CUDA 12.9.1
```

The `no_nvidia_pip: true` field in env yamls makes the validator enforce this.

---

## Available MLSys environments

| Env name | Added packages | Key use case |
|----------|---------------|-------------|
| `qwen3-tts` | transformers, accelerate, soundfile, librosa | Qwen3-TTS inference |
| `hf-transformers` | transformers, tokenizers, accelerate | General HF inference |
| `hf-datasets` | datasets, huggingface-hub | Dataset loading |
| `vllm` | vllm | Fast LLM serving |
| `sglang` | sglang | Structured generation |
| `llm-serving-all` | vllm + sglang + extras | Full serving stack |
| `torchtitan` | torchtitan | Large-scale training |
| `megatronlm` | megatron-lm | Megatron training |

---

## qwen3-tts workflow

torchaudio is baked into the Spack base. The `qwen3-tts` env adds only the inference stack on top.

```bash
# Validate the stack (run once after vendoring or image changes)
./scripts/launch_mlsys_hf.sh qwen3-tts

# Quick torchaudio check without building a full env
.sygaldry/zephyr/bin/repoctl run -- python3 -c "
import torch, torchaudio
assert torch.cuda.is_available()
print(f'torch={torch.__version__} torchaudio={torchaudio.__version__} OK')
"

# Persist the venv, then run separately
MLSYS_VENV_ROOT=/workspace/audri/.venv-mlsys ./scripts/launch_mlsys_hf.sh qwen3-tts
.sygaldry/zephyr/bin/repoctl run -- bash -c "
  source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate
  python3 /workspace/audri/scripts/tts_infer.py
"

# Add extra packages using uv layering (never pip/uv --system directly)
just layer "some-package"
```

If a HuggingFace gated model requires auth, pass `HF_TOKEN` via the environment:
```bash
HF_TOKEN=<token> .sygaldry/zephyr/bin/repoctl run -- python3 /workspace/audri/scripts/tts_infer.py
```

---

## Background jobs (`jobctl`)

For long-running or fire-and-forget jobs, use `jobctl` (produces JSONL logs under the cache root):

```bash
JOBCTL=".sygaldry/zephyr/bin/jobctl"

"${JOBCTL}" run --project-id audri --job tts-train -- "python3 /workspace/audri/scripts/train.py"
"${JOBCTL}" status --project-id audri --job tts-train
"${JOBCTL}" tail   --project-id audri --job tts-train --lines 50
"${JOBCTL}" health --project-id audri --job tts-train
"${JOBCTL}" stop   --project-id audri --job tts-train
```

Job outputs land at `/mnt/data_infra/zephyr_container_infra/projects/audri/jobs/tts-train/`.

---

## Rules for agents

1. **Never `python3 ...`, `pip install ...`, or `uv pip install ...` on the host.** Always go through `repoctl run` or `scripts/launch_mlsys_hf.sh`.
2. **Never install additional packages with direct `pip install` or `uv pip install --system` in container commands.** Use `scripts/launch_mlsys_hf.sh` or `just zephyr-layer ...` so uv layering constraints are always enforced.
3. **Never edit `.sygaldry/zephyr/infra.yaml` or `.codex-zephyr-mlsys/runtime.yaml`.** These are digest-pinned and managed by vendoring scripts.
4. **Never add spack-owned packages** (torch, torchaudio, numpy, scipy, jax, triton, numba, llvmlite, Python itself) to env yaml `packages:` lists.
5. **New Python scripts belong in the repo root** (e.g. `scripts/`, `src/`). They are accessible inside the container at `/workspace/audri/<path>`.
6. **`just zephyr-check` before any container operation** if you're unsure the infra is intact.
7. **Do not commit `.venv-mlsys/` or `.venv*/` directories** — they are gitignored and rebuilt inside the container.

---

## Re-vendoring (when upstream sygaldry changes)

Only run this when explicitly asked:

```bash
SYGALDRY=/mnt/data_infra/workspace/sygaldry
SNAP="ghcr.io/phi9t/sygaldry/zephyr:spack@sha256:8c9507aea53995f29a5712c0cbdb99deb3d571fb9631b3d42352b3d6d6fb668c"

"${SYGALDRY}/skills/zephyr/scripts/zephyr_vendor_infra.sh" update --target-repo .
"${SYGALDRY}/skills/zephyr/scripts/zephyr_mlsys_vendor.sh" update --target-repo . --snapshot-ref "${SNAP}"
```

After re-vendoring, run `just zephyr-check` to verify, then commit the updated kit files.
