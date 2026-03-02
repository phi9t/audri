#!/bin/bash
#
# Sygaldry Container Entrypoint (interactive shell)
# ==================================================
#
# Initializes Spack + CUDA + MLSys venv, validates GPU, and starts an
# interactive shell. If arguments are provided, executes them directly.

# Resolve lib path (works both with baked /opt/container_entrypoints and dev mounts)
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)" \
    || _LIB_DIR="/opt/container_entrypoints/../lib"
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

# ============================================================================
# Environment Setup
# ============================================================================

sygaldry_init_spack || true

if ! sygaldry_activate_env; then
    echo "WARNING: Could not find an activatable Spack environment." >&2
    echo "Will attempt /opt/spack_store/view fallback." >&2
fi

sygaldry_require_torch_jax || exit 1

export EDITOR="${EDITOR:-nano}"
export TERM="${TERM:-xterm-256color}"

sygaldry_setup_cuda

run_gpu_validation=1
if [[ $# -gt 0 && "${SYGALDRY_VALIDATE_GPU_ON_EXEC:-0}" != "1" ]]; then
    run_gpu_validation=0
fi

# Fail fast if GPU support is not functional for Torch/JAX.
if [[ ${run_gpu_validation} -eq 1 ]]; then
python3 - <<'PY'
import sys

errors = []
try:
    import torch
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False")
    elif torch.cuda.device_count() < 1:
        errors.append("torch reports zero CUDA devices")
except Exception as exc:
    errors.append(f"torch error: {exc}")

try:
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        errors.append(f"jax has no GPU devices (devices={devices})")
except Exception as exc:
    errors.append(f"jax error: {exc}")

if errors:
    print("ERROR: GPU validation failed in Zephyr container:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    print("HINT:  Check nvidia-smi on host; run container/diagnose_nvidia.sh", file=sys.stderr)
    sys.exit(1)

print("GPU validation passed: Torch and JAX both see a CUDA GPU.", file=sys.stderr)
PY
fi

# ============================================================================
# Baked MLSys Venv Auto-Activation
# ============================================================================

sygaldry_activate_mlsys_venv

# ============================================================================
# Convenience Aliases and Functions
# ============================================================================

alias gpu-test='python3 -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}\")"'
alias jax-test='python3 -c "import jax; print(f\"Devices: {jax.devices()}\")"'
alias spack-build='cd ${SYGALDRY_SPACK_ENV:-/opt/spack_env/default} && ./build.sh'

hf-dataset() {
    python3 -c "from datasets import load_dataset; ds=load_dataset('$1', split='${2:-train[:100]}'); print(f'{len(ds)} rows')"
}

# ============================================================================
# Command Execution
# ============================================================================

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

# ============================================================================
# Welcome Message (interactive shell only)
# ============================================================================

if [[ -n "${SYGALDRY_MLSYS_ENV:-}" ]]; then
    echo "+-------------------------------------------------------------+"
    echo "|                    Sygaldry Build Environment               |"
    printf "|  MLSys env: %-47s |\n" "${SYGALDRY_MLSYS_ENV}"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        printf "|  Venv:      %-47s |\n" "${VIRTUAL_ENV}"
    fi
    echo "|                                                             |"
    echo "|  Quick commands:                                            |"
    echo "|    gpu-test                   - Verify PyTorch CUDA         |"
    echo "|    jax-test                   - Verify JAX GPU              |"
    if [[ "${SYGALDRY_MLSYS_ENV}" == "llm-serving-all" ]]; then
        echo "|    mlsys-activate <name>      - Switch venv (hf/vllm/sglang)|"
    fi
    echo "+-------------------------------------------------------------+"
else
    echo "+-------------------------------------------------------------+"
    echo "|                    Sygaldry Build Environment               |"
    echo "|                                                             |"
    echo "|  Quick commands:                                            |"
    echo "|    spack-env-activate         - Activate Spack environment  |"
    echo "|    gpu-test                   - Verify PyTorch CUDA         |"
    echo "|    jax-test                   - Verify JAX GPU              |"
    echo "|    spack-build                - Build Zephyr environment    |"
    echo "|                                                             |"
    echo "|  Environment:                                               |"
    echo "|    Workspace: /workspace                                    |"
    echo "|    Spack:     /opt/spack_src                                |"
    echo "|    HF Cache:  /opt/hf_cache                                 |"
    echo "+-------------------------------------------------------------+"
fi

exec bash -i
