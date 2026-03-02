#!/usr/bin/env bash
# scripts/run_speak_demo.sh — Milestone 1 streaming demo launcher
#
# Usage:
#   ./scripts/run_speak_demo.sh [--model Qwen/Qwen3-TTS-0.6B]
#   AUDRI_HEADLESS=1 ./scripts/run_speak_demo.sh        (saves WAVs, no audio device needed)
#   HF_TOKEN=hf_xxxx... ./scripts/run_speak_demo.sh
#
# Headless mode (default): writes per-chunk WAVs to outputs/speak_demo_*.wav
# Live mode: plays audio through sounddevice (requires an audio device)
#
# Pre-requisites: same as run_smoke_test.sh (HF token + model license)

set -euo pipefail
cd "$(dirname "$0")/.."

ARGS="${@}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo -n "${HF_TOKEN}" > /mnt/data_infra/zephyr_container_infra/shared/hf_cache/token
    echo "[speak_demo] Wrote HF_TOKEN to shared HF cache."
fi

exec .sygaldry/zephyr/bin/repoctl run -- bash -c "
  unset PYTHONPATH &&
  source /workspace/audri/.venv-mlsys/qwen3-tts/bin/activate &&
  AUDRI_HEADLESS=${AUDRI_HEADLESS:-1} python3 /workspace/audri/scripts/speak_demo.py ${ARGS}
"
