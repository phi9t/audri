#!/bin/bash
set -euo pipefail

# HF LoRA setup that reuses Spack-built PyTorch (no torch override).
# Installs only HuggingFace/PEFT layers in a venv and provides a wrapper
# that places venv site-packages ahead of Spack site-packages at runtime.

if [[ $# -gt 0 ]]; then
    echo "Usage: hf-lora-setup.sh" >&2
    echo "Environment variables:" >&2
    echo "  VENV_DIR (default: .venv-hf-lora)" >&2
    echo "  HF_TRANSFORMERS_VERSION (default: 5.2.0)" >&2
    echo "  HF_DATASETS_VERSION (default: 4.5.0)" >&2
    echo "  HF_PEFT_VERSION (default: 0.18.1)" >&2
    echo "  HF_ACCELERATE_VERSION (default: 1.12.0)" >&2
    exit 2
fi

VENV_DIR="${VENV_DIR:-.venv-hf-lora}"
SPACK_PY="${PYTHON_BIN:-/opt/spack_store/view/bin/python3}"
HF_TRANSFORMERS_VERSION="${HF_TRANSFORMERS_VERSION:-5.2.0}"
HF_DATASETS_VERSION="${HF_DATASETS_VERSION:-4.5.0}"
HF_PEFT_VERSION="${HF_PEFT_VERSION:-0.18.1}"
HF_ACCELERATE_VERSION="${HF_ACCELERATE_VERSION:-1.12.0}"

if [[ ! -x "${SPACK_PY}" ]]; then
    echo "ERROR: Spack Python not found at ${SPACK_PY}" >&2
    exit 1
fi

uv venv --python "${SPACK_PY}" --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Install HF stack without touching torch/triton from Spack.
uv pip install \
    "transformers==${HF_TRANSFORMERS_VERSION}" \
    "datasets==${HF_DATASETS_VERSION}" \
    soundfile
uv pip install --no-deps \
    "peft==${HF_PEFT_VERSION}" \
    "accelerate==${HF_ACCELERATE_VERSION}"

cat > "${VENV_DIR}/bin/hf-lora-python" <<'EOF'
#!/bin/bash
set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${BIN_DIR}/python"

PY_VER="$("${PY_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
VENV_SITE="$("${PY_BIN}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
SPACK_SITE="/opt/spack_store/view/lib/python${PY_VER}/site-packages"

if [[ -d "${SPACK_SITE}" ]]; then
    export PYTHONPATH="${VENV_SITE}:${SPACK_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "${PY_BIN}" "$@"
EOF
chmod +x "${VENV_DIR}/bin/hf-lora-python"

echo "HF LoRA environment ready in ${VENV_DIR}"
echo "Torch source: Spack view (no torch override)"
echo "Run jobs with: ${VENV_DIR}/bin/hf-lora-python hf_lora_train.py"
