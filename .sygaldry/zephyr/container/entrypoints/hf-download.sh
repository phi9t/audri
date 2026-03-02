#!/bin/bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/opt/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/uv_cache}"
VENV_DIR="${HF_VENV_DIR:-/opt/uv_cache/venvs/hf-download}"
UV_BIN="${UV_BIN:-uv}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "$(dirname "${VENV_DIR}")"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "ERROR: uv is required but not found in PATH (set UV_BIN to override)." >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    "${UV_BIN}" venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -c "import datasets, huggingface_hub" >/dev/null 2>&1; then
    "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" datasets huggingface-hub
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

MODE="${1:-help}"
shift || true

case "${MODE}" in
    dataset)
        DATASET="${1:?Dataset ID required}"
        CONFIG="${2:-default}"
        SPLIT="${3:-train[:100]}"
        "${PYTHON_BIN}" -c "
from datasets import load_dataset
import os
ds = load_dataset('${DATASET}', '${CONFIG}', split='${SPLIT}', cache_dir=os.environ['HF_HOME'])
print(f'Downloaded {len(ds)} rows from ${DATASET}')
"
        ;;
    model)
        MODEL_ID="${1:?Model ID required}"
        "${PYTHON_BIN}" -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${MODEL_ID}', cache_dir=os.environ['HF_HOME'])
print(f'Downloaded ${MODEL_ID} to {path}')
"
        ;;
    *)
        cat <<EOF
Usage:
  hf-download.sh dataset <dataset_id> [config] [split]
  hf-download.sh model <model_id>

Examples:
  hf-download.sh dataset HuggingFaceFW/fineweb default "train[:1000]"
  hf-download.sh model Qwen/Qwen3-0.6B-Base

Environment:
  HF_HOME    - Cache directory (default: /opt/hf_cache)
  HF_TOKEN   - HuggingFace token for gated models
EOF
        exit 1
        ;;
esac
