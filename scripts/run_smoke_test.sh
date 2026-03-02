#!/usr/bin/env bash
# scripts/run_smoke_test.sh — Milestone 0 smoke test launcher
#
# Usage:
#   ./scripts/run_smoke_test.sh [--model Qwen/Qwen3-TTS-0.6B] [--text "..."]
#   HF_TOKEN=hf_xxxx... ./scripts/run_smoke_test.sh
#
# Pre-requisites:
#   Accept the model license at https://huggingface.co/Qwen/Qwen3-TTS-0.6B
#   Provide your HF token via one of:
#     a) HF_TOKEN=hf_xxxx... ./scripts/run_smoke_test.sh   (written to shared cache)
#     b) echo -n "$HF_TOKEN" > /mnt/data_infra/zephyr_container_infra/shared/hf_cache/token
#
# System libs (libsndfile1, libportaudio2, sox) are baked into sygaldry/zephyr:audri-tts.
# Outputs: outputs/smoke_test.wav

set -euo pipefail
cd "$(dirname "$0")/.."

ARGS="${@}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo -n "${HF_TOKEN}" > /mnt/data_infra/zephyr_container_infra/shared/hf_cache/token
    echo "[smoke_test] Wrote HF_TOKEN to shared HF cache."
fi

exec .sygaldry/zephyr/bin/repoctl run -- bash -c "
  unset PYTHONPATH &&
  source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate &&
  python3 /workspace/audri/scripts/smoke_test_tts.py --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign ${ARGS}
"
