#!/bin/bash
#
# UV-Spack Layering Verification
# ================================
#
# Validates that UV-installed packages correctly layer on top of the
# Spack environment without overriding core packages or pulling in
# redundant NVIDIA/CUDA pip packages.
#
# Usage:
#   container/verify_uv_layering.sh                          # default image
#   container/verify_uv_layering.sh sygaldry/zephyr:spack    # specific image
#   container/verify_uv_layering.sh --no-gpu                 # skip GPU tests
#   container/verify_uv_layering.sh --with-vllm              # include vllm/sglang install
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#
# Prerequisites:
#   - Docker daemon running
#   - Snapshot image built (sygaldry/zephyr:spack or specified image)
#   - For GPU tests: NVIDIA Docker runtime + GPU

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
source "${SCRIPT_DIR}/lib/verify_common.sh"

# ============================================================================
# Configuration
# ============================================================================

IMAGE="${SYGALDRY_SNAPSHOT_IMAGE:-sygaldry/zephyr:spack}"
GPU=true
WITH_VLLM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gpu)
            GPU=false
            shift
            ;;
        --with-vllm)
            WITH_VLLM=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [IMAGE] [--no-gpu] [--with-vllm]"
            echo ""
            echo "Verify UV-Spack package layering in a snapshot image."
            echo ""
            echo "Options:"
            echo "  IMAGE        Docker image to test (default: sygaldry/zephyr:spack)"
            echo "  --no-gpu     Skip GPU tests (T8.5)"
            echo "  --with-vllm  Include vllm/sglang install test (T8.6, slow)"
            exit 0
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            exit 2
            ;;
        *)
            IMAGE="$1"
            shift
            ;;
    esac
done

readonly IMAGE
readonly GPU
readonly WITH_VLLM

verify_reset_counters

# Run a command inside the image (no GPU)
run_no_gpu() {
    local cmd="$1"
    verify_run_no_gpu "${IMAGE}" "${cmd}" 2>&1
}

# Run a command inside the image with GPU
run_with_gpu() {
    local cmd="$1"
    verify_run_with_gpu "${IMAGE}" "${cmd}" 2>&1
}

# ============================================================================
# Preflight
# ============================================================================

echo "=== UV-Spack Layering Verification ==="
echo ""
echo "  Image:     ${IMAGE}"
echo "  GPU tests: ${GPU}"
echo "  vllm:      ${WITH_VLLM}"
echo ""

verify_require_docker || exit 1
verify_require_image "${IMAGE}" || exit 1
echo "  Docker: OK"
echo ""

# ============================================================================
# Shared setup script that runs inside the container
# ============================================================================

# This script is injected into the container to:
# 1. Set up the Spack environment
# 2. Create a UV venv and install test packages
# The venv is created in /tmp/test-venv to avoid permission issues.
SETUP_SCRIPT='
set -euo pipefail
export PATH="/usr/local:/usr/local/bin:${PATH}"

# Source Spack
source /opt/spack_src/share/spack/setup-env.sh
spack env activate /opt/spack_env/default 2>/dev/null || true

PYTHON_BIN="/opt/spack_store/view/bin/python3"
VENV_DIR="/tmp/test-venv"

# Fresh UV cache per verification run (avoid stale sdist artifacts)
export UV_CACHE_DIR="/tmp/uv-cache-$$"
mkdir -p "${UV_CACHE_DIR}"

# Use uv-install.sh entrypoint (datasets split out — see T8.7)
export VENV_DIR="${VENV_DIR}"
/opt/container_entrypoints/uv-install.sh transformers tokenizers accelerate
'

# ============================================================================
# T8.1: UV venv creation
# ============================================================================

echo "=== T8.1: UV venv creation ==="

result="$(run_no_gpu "${SETUP_SCRIPT}"$'\n'"echo UV_VENV_OK")" || true
if echo "${result}" | grep -q "UV_VENV_OK"; then
    pass "UV venv created and packages installed"
else
    fail "UV venv creation" "output: $(echo "${result}" | tail -5)"
    echo ""
    echo "FATAL: Cannot proceed without UV venv. Stopping."
    exit 1
fi

# ============================================================================
# T8.2: Spack provenance
# ============================================================================

echo ""
echo "=== T8.2: Spack provenance ==="

# Check that core Spack packages resolve from the Spack view
SPACK_PKGS="torch torchvision torchaudio jax numpy scipy triton"
for pkg in ${SPACK_PKGS}; do
    mod_name="${pkg//-/_}"
    result="$(run_no_gpu "
        ${SETUP_SCRIPT} >/dev/null 2>&1
        source /tmp/test-venv/bin/activate
        python3 -c \"
import ${mod_name}
f = getattr(${mod_name}, '__file__', '') or ''
if '/opt/spack_store/view' in f:
    print('SPACK_OK')
else:
    print('NOT_SPACK: ' + f)
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "SPACK_OK" ]]; then
        pass "${pkg} from Spack view"
    else
        fail "${pkg} from Spack view" "got '${last_line}'"
    fi
done

# ============================================================================
# T8.3: UV provenance
# ============================================================================

echo ""
echo "=== T8.3: UV provenance ==="

UV_PKGS="transformers tokenizers accelerate"
for pkg in ${UV_PKGS}; do
    mod_name="${pkg//-/_}"
    result="$(run_no_gpu "
        ${SETUP_SCRIPT} >/dev/null 2>&1
        source /tmp/test-venv/bin/activate
        python3 -c \"
import ${mod_name}
f = getattr(${mod_name}, '__file__', '') or ''
if '/tmp/test-venv' in f:
    print('UV_OK')
elif '/opt/spack_store/view' in f:
    print('SPACK_VIEW: ' + f)
else:
    print('UNKNOWN: ' + f)
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "UV_OK" ]]; then
        pass "${pkg} from UV venv"
    else
        fail "${pkg} from UV venv" "got '${last_line}'"
    fi
done

# ============================================================================
# T8.4: No NVIDIA pip packages
# ============================================================================

echo ""
echo "=== T8.4: No NVIDIA pip packages ==="

result="$(run_no_gpu "
    ${SETUP_SCRIPT} >/dev/null 2>&1
    source /tmp/test-venv/bin/activate
    python3 -c \"
import importlib.metadata as md
nvidia_pkgs = []
for dist in md.distributions():
    name = (dist.metadata.get('Name') or '').lower()
    if name.startswith('nvidia-'):
        loc = str(getattr(dist, '_path', 'unknown'))
        # Ignore packages inside Spack view
        if '/opt/spack_store/view' not in loc:
            nvidia_pkgs.append(name)
if nvidia_pkgs:
    print('NVIDIA_FOUND: ' + ','.join(nvidia_pkgs))
else:
    print('NO_NVIDIA_OK')
\"
")" || true
last_line="$(echo "${result}" | tail -1)"
if [[ "${last_line}" == "NO_NVIDIA_OK" ]]; then
    pass "No nvidia-* pip packages installed by UV"
else
    fail "No nvidia-* pip packages installed by UV" "got '${last_line}'"
fi

# ============================================================================
# T8.5: GPU functional after UV layering
# ============================================================================

if [[ "${GPU}" != "true" ]]; then
    echo ""
    echo "=== T8.5: GPU tests skipped (--no-gpu) ==="
else
    if ! verify_has_nvidia_runtime; then
        echo ""
        echo "=== T8.5: GPU tests skipped (NVIDIA runtime not detected) ==="
    else
        echo ""
        echo "=== T8.5: GPU functional after UV layering ==="

        # torch.cuda.is_available() in the UV venv
        result="$(run_with_gpu "
            ${SETUP_SCRIPT} >/dev/null 2>&1
            source /tmp/test-venv/bin/activate
            python3 -c \"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
a = torch.randn(64, 64, device='cuda')
b = torch.randn(64, 64, device='cuda')
c = torch.matmul(a, b)
assert c.shape == (64, 64)
print('GPU_OK')
\"
        ")" || true
        last_line="$(echo "${result}" | tail -1)"
        if [[ "${last_line}" == "GPU_OK" ]]; then
            pass "torch CUDA matmul works after UV layering"
        else
            fail "torch CUDA matmul after UV layering" "got '${last_line}'"
        fi

        # transformers model load (quick, verifies HF + torch integration)
        result="$(run_with_gpu "
            ${SETUP_SCRIPT} >/dev/null 2>&1
            source /tmp/test-venv/bin/activate
            python3 -c \"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
print('HF_OK')
\" 2>/dev/null
        ")" || true
        last_line="$(echo "${result}" | tail -1)"
        if [[ "${last_line}" == "HF_OK" ]]; then
            pass "HuggingFace transformers tokenizer load"
        else
            fail "HuggingFace transformers tokenizer load" "got '${last_line}'"
        fi
    fi
fi

# ============================================================================
# T8.7: datasets install (non-blocking)
# ============================================================================

echo ""
echo "=== T8.7: datasets install (non-blocking) ==="

DATASETS_SETUP='
set -euo pipefail
export PATH="/usr/local:/usr/local/bin:${PATH}"
source /opt/spack_src/share/spack/setup-env.sh
spack env activate /opt/spack_env/default 2>/dev/null || true
export VENV_DIR="/tmp/test-venv-datasets"
export UV_CACHE_DIR="/tmp/uv-cache-$$"
mkdir -p "${UV_CACHE_DIR}"
/opt/container_entrypoints/uv-install.sh datasets
'

result="$(run_no_gpu "${DATASETS_SETUP}"$'\n'"echo DATASETS_INSTALL_OK")" || true
if echo "${result}" | grep -q "DATASETS_INSTALL_OK"; then
    pass "datasets installed in separate venv"
else
    # Non-blocking: report but don't fail the run
    echo "  WARN: datasets install failed (non-blocking)"
    echo "        output: $(echo "${result}" | tail -3)"
fi

if [[ "${WITH_VLLM}" == "true" ]]; then
    run_fn="run_no_gpu"
    if [[ "${GPU}" == "true" ]] && verify_has_nvidia_runtime; then
        run_fn="run_with_gpu"
    fi

    # ========================================================================
    # T8.6: vLLM install
    # ========================================================================

    echo ""
    echo "=== T8.6: vllm install ==="

    VLLM_SETUP='
set -euo pipefail
export PATH="/usr/local:/usr/local/bin:${PATH}"
source /opt/spack_src/share/spack/setup-env.sh
spack env activate /opt/spack_env/default 2>/dev/null || true
export VENV_DIR="/tmp/test-venv-vllm"
export UV_CACHE_DIR="/tmp/uv-cache-$$"
mkdir -p "${UV_CACHE_DIR}"
export UV_EXTRA_OVERRIDES="/opt/container_entrypoints/llm_serving_overrides.txt"
/opt/container_entrypoints/uv-install.sh "vllm>=0.15.0"
'

    result="$($run_fn "${VLLM_SETUP}"$'\n'"echo VLLM_INSTALL_OK")" || true
    if echo "${result}" | grep -q "VLLM_INSTALL_OK"; then
        pass "vllm installed"
    else
        fail "vllm install" "output: $(echo "${result}" | tail -5)"
    fi

    # Check provenance in vllm venv — no nvidia-* pip leaks
    result="$($run_fn "
        ${VLLM_SETUP} >/dev/null 2>&1
        source /tmp/test-venv-vllm/bin/activate
        python3 -c \"
import importlib.metadata as md
nvidia_pkgs = []
for dist in md.distributions():
    name = (dist.metadata.get('Name') or '').lower()
    if name.startswith('nvidia-'):
        loc = str(getattr(dist, '_path', 'unknown'))
        if '/opt/spack_store/view' not in loc:
            nvidia_pkgs.append(name)
if nvidia_pkgs:
    print('NVIDIA_FOUND: ' + ','.join(nvidia_pkgs))
else:
    print('NO_NVIDIA_OK')
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "NO_NVIDIA_OK" ]]; then
        pass "No nvidia-* pip packages in vllm venv"
    else
        fail "No nvidia-* pip packages in vllm venv" "got '${last_line}'"
    fi

    # Hard-fail check: verify vllm can import and use Spack torch
    result="$($run_fn "
        ${VLLM_SETUP} >/dev/null 2>&1
        source /tmp/test-venv-vllm/bin/activate
        python3 -c \"
import torch
assert torch.cuda.is_available(), 'CUDA not available'
try:
    import vllm
    print(f'vllm {vllm.__version__} imported OK')
except ImportError as e:
    err = str(e)
    if any(k in err for k in ['undefined symbol', 'CUDA', 'libcusparse', 'libnvJitLink']):
        print('HARD_FAIL: vllm')
        print(f'  core_conflict: torch=={torch.__version__}')
        print(f'  failure_type: abi_mismatch')
        print(f'  evidence: {e}')
        import sys; sys.exit(2)
    raise
print('VLLM_FUNCTIONAL_OK')
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "VLLM_FUNCTIONAL_OK" ]]; then
        pass "vllm import + torch CUDA functional"
    elif echo "${result}" | grep -q "HARD_FAIL"; then
        fail "vllm HARD_FAIL: core package ABI mismatch" "$(echo "${result}" | grep -A3 'HARD_FAIL')"
    else
        fail "vllm functional check" "got '${last_line}'"
    fi

    # ========================================================================
    # T8.8: sglang install
    # ========================================================================

    echo ""
    echo "=== T8.8: sglang install ==="

    SGLANG_SETUP='
set -euo pipefail
export PATH="/usr/local:/usr/local/bin:${PATH}"
source /opt/spack_src/share/spack/setup-env.sh
spack env activate /opt/spack_env/default 2>/dev/null || true
export VENV_DIR="/tmp/test-venv-sglang"
export UV_CACHE_DIR="/tmp/uv-cache-$$"
mkdir -p "${UV_CACHE_DIR}"
export UV_EXTRA_OVERRIDES="/opt/container_entrypoints/llm_serving_overrides.txt"
/opt/container_entrypoints/uv-install.sh "sglang>=0.5.8" pybase64 pydantic fastapi uvicorn zmq
'

    result="$($run_fn "${SGLANG_SETUP}"$'\n'"echo SGLANG_INSTALL_OK")" || true
    if echo "${result}" | grep -q "SGLANG_INSTALL_OK"; then
        pass "sglang installed"
    else
        fail "sglang install" "output: $(echo "${result}" | tail -5)"
    fi

    # Check provenance in sglang venv — no nvidia-* pip leaks
    result="$($run_fn "
        ${SGLANG_SETUP} >/dev/null 2>&1
        source /tmp/test-venv-sglang/bin/activate
        python3 -c \"
import importlib.metadata as md
nvidia_pkgs = []
for dist in md.distributions():
    name = (dist.metadata.get('Name') or '').lower()
    if name.startswith('nvidia-'):
        loc = str(getattr(dist, '_path', 'unknown'))
        if '/opt/spack_store/view' not in loc:
            nvidia_pkgs.append(name)
if nvidia_pkgs:
    print('NVIDIA_FOUND: ' + ','.join(nvidia_pkgs))
else:
    print('NO_NVIDIA_OK')
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "NO_NVIDIA_OK" ]]; then
        pass "No nvidia-* pip packages in sglang venv"
    else
        fail "No nvidia-* pip packages in sglang venv" "got '${last_line}'"
    fi

    # Hard-fail check: verify sglang can import and use Spack torch/triton
    result="$($run_fn "
        ${SGLANG_SETUP} >/dev/null 2>&1
        source /tmp/test-venv-sglang/bin/activate
        python3 -c \"
import torch, triton
assert torch.cuda.is_available(), 'CUDA not available'
try:
    import sglang
    print(f'sglang {sglang.__version__} imported OK')
except ImportError as e:
    err = str(e)
    if any(k in err for k in ['undefined symbol', 'CUDA', 'triton', 'libnvJitLink']):
        print('HARD_FAIL: sglang')
        print(f'  core_conflict: triton=={triton.__version__}, torch=={torch.__version__}')
        print(f'  failure_type: abi_mismatch')
        print(f'  evidence: {e}')
        import sys; sys.exit(2)
    raise
print('SGLANG_FUNCTIONAL_OK')
\"
    ")" || true
    last_line="$(echo "${result}" | tail -1)"
    if [[ "${last_line}" == "SGLANG_FUNCTIONAL_OK" ]]; then
        pass "sglang import + torch/triton CUDA functional"
    elif echo "${result}" | grep -q "HARD_FAIL"; then
        fail "sglang HARD_FAIL: core package ABI mismatch" "$(echo "${result}" | grep -A5 'HARD_FAIL')"
    else
        fail "sglang functional check" "got '${last_line}'"
    fi
fi

# ============================================================================
# Summary
# ============================================================================

verify_print_summary "UV-Spack Layering Verification Summary" "${IMAGE}"
verify_exit_on_failures
