#!/bin/bash
#
# uv-env-activate.sh — Activate MLSys venvs (multi-venv aware)
# =============================================================
#
# This script is designed to be SOURCED, not executed:
#   source uv-env-activate.sh <env-name> [--target <venv-name>]
#
# For single-venv environments:
#   source uv-env-activate.sh vllm
#
# For multi-venv environments (llm-serving-all):
#   source uv-env-activate.sh llm-serving-all --target vllm
#   source uv-env-activate.sh llm-serving-all --target sglang
#   source uv-env-activate.sh llm-serving-all --target hf
#
# Environment variables:
#   VENV_ROOT  — Root directory for venvs (default: /tmp/mlsys-envs)
#
# The script also ensures the Spack view site-packages are visible.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed." >&2
    echo "  Usage: source $0 <env-name> [--target <venv-name>]" >&2
    exit 1
fi

_MLSYS_VENV_ROOT="${VENV_ROOT:-/tmp/mlsys-envs}"
_MLSYS_ENV_NAME=""
_MLSYS_TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            _MLSYS_TARGET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: source $0 <env-name> [--target <venv-name>]"
            echo ""
            echo "Activate an MLSys venv built by uv-env-build.sh."
            echo ""
            echo "For single-venv envs:  source $0 vllm"
            echo "For multi-venv envs:   source $0 llm-serving-all --target vllm"
            return 0 2>/dev/null || true
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            return 1 2>/dev/null || true
            ;;
        *)
            _MLSYS_ENV_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "${_MLSYS_ENV_NAME}" ]]; then
    echo "Usage: source $0 <env-name> [--target <venv-name>]" >&2
    return 1 2>/dev/null || true
fi

# Resolve venv path
if [[ -n "${_MLSYS_TARGET}" ]]; then
    _MLSYS_VENV_DIR="${_MLSYS_VENV_ROOT}/${_MLSYS_ENV_NAME}/${_MLSYS_TARGET}"
else
    _MLSYS_VENV_DIR="${_MLSYS_VENV_ROOT}/${_MLSYS_ENV_NAME}"
fi

if [[ ! -d "${_MLSYS_VENV_DIR}" ]]; then
    echo "ERROR: Venv not found: ${_MLSYS_VENV_DIR}" >&2
    echo "  Available envs:" >&2
    ls -1 "${_MLSYS_VENV_ROOT}/" 2>/dev/null | sed 's/^/    /' >&2
    return 1 2>/dev/null || true
fi

if [[ ! -f "${_MLSYS_VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Not a valid venv: ${_MLSYS_VENV_DIR}" >&2
    return 1 2>/dev/null || true
fi

# Activate
source "${_MLSYS_VENV_DIR}/bin/activate"
echo "Activated: ${_MLSYS_VENV_DIR}"

# Verify Spack view is visible
if python3 -c "import torch" 2>/dev/null; then
    echo "  torch available (Spack)"
fi

# Cleanup temp vars
unset _MLSYS_VENV_ROOT _MLSYS_ENV_NAME _MLSYS_TARGET _MLSYS_VENV_DIR
