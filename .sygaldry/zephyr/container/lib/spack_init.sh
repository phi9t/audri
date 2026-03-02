#!/bin/bash
# Shared Spack initialization library for Sygaldry container entrypoints.
#
# Provides a canonical initialization cascade used by all entrypoints.
# Source this file; do not execute it directly.
#
# Functions:
#   sygaldry_init_spack           - Source Spack setup-env.sh
#   sygaldry_activate_env         - Activate a Spack environment from candidate list
#   sygaldry_ensure_view_fallback - Inject Spack view into PATH/PYTHONPATH/LD_LIBRARY_PATH
#   sygaldry_setup_cuda           - Set CUDA_HOME, PATH, LD_LIBRARY_PATH
#   sygaldry_activate_mlsys_venv  - Auto-activate baked MLSys venv (single or multi)
#   sygaldry_full_init            - All of the above in sequence

if [[ -n "${_SYGALDRY_SPACK_INIT_LOADED:-}" ]]; then
    return 0
fi
readonly _SYGALDRY_SPACK_INIT_LOADED=1

# Load shared error helpers
_SYGALDRY_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SYGALDRY_LIB_DIR}/errors.sh"

# ---------------------------------------------------------------------------
# sygaldry_init_spack
# ---------------------------------------------------------------------------
# Sources Spack's setup-env.sh. Optionally loads bash completion.
# Returns 0 on success, 1 if Spack is not installed.
sygaldry_init_spack() {
    local spack_setup="/opt/spack_src/share/spack/setup-env.sh"
    if [[ ! -f "${spack_setup}" ]]; then
        error_with_hint \
            "Spack setup script not found at /opt/spack_src." \
            "Are you inside the container? Use 'sygaldry' or 'launch_container.sh'."
        return 1
    fi
    # shellcheck disable=SC1090
    source "${spack_setup}"

    local completion="/opt/spack_src/share/spack/spack-completion.bash"
    if [[ -f "${completion}" ]]; then
        # shellcheck disable=SC1090
        source "${completion}"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# sygaldry_activate_env
# ---------------------------------------------------------------------------
# Tries to activate a Spack environment from an ordered candidate list.
# The candidate order is canonical across ALL entrypoints:
#   1. $SYGALDRY_SPACK_ENV (explicit override)
#   2. Current directory (if spack.yaml/spack.lock present)
#   3. /opt/spack_env/default  (baked image)
#   4. /opt/spack_env/zephyr   (baked image)
#   5. $SYGALDRY_ROOT/pkg/zephyr (workspace fallback)
#
# Returns 0 if an environment was activated, 1 otherwise.
sygaldry_activate_env() {
    local candidates=()

    if [[ -n "${SYGALDRY_SPACK_ENV:-}" ]]; then
        candidates+=("${SYGALDRY_SPACK_ENV}")
    fi

    # Current directory (if it looks like a Spack env)
    if [[ -f "spack.yaml" ]] || [[ -f "spack.lock" ]]; then
        candidates+=(".")
    fi

    candidates+=(
        "/opt/spack_env/default"
        "/opt/spack_env/zephyr"
        "${SYGALDRY_ROOT:-/workspace}/pkg/zephyr"
    )

    local env_path
    for env_path in "${candidates[@]}"; do
        if [[ -f "${env_path}/spack.yaml" ]] || [[ -f "${env_path}/spack.lock" ]] || [[ "${env_path}" == "." ]]; then
            if [[ -n "${SPACK_ENV:-}" ]]; then
                spack env deactivate >/dev/null 2>&1 || true
            fi
            if spack env activate "${env_path}" >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# sygaldry_ensure_view_fallback
# ---------------------------------------------------------------------------
# Adds the Spack view to PATH and LD_LIBRARY_PATH.
# Optionally exports PYTHONPATH when SYGALDRY_SPACK_EXPORT_PYTHONPATH=1.
# Used when Spack env activation fails but the view directory exists.
# Returns 0 if the view exists, 1 otherwise.
sygaldry_ensure_view_fallback() {
    local spack_view="/opt/spack_store/view"
    if [[ ! -d "${spack_view}" ]]; then
        return 1
    fi

    export PATH="${spack_view}/bin:${PATH}"

    if [[ -x "${spack_view}/bin/python3" ]] && [[ "${SYGALDRY_SPACK_EXPORT_PYTHONPATH:-0}" == "1" ]]; then
        local py_ver
        py_ver="$("${spack_view}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.13")"
        export PYTHONPATH="${spack_view}/lib/python${py_ver}/site-packages:${PYTHONPATH:-}"
    fi

    export LD_LIBRARY_PATH="${spack_view}/lib:${spack_view}/lib64:${LD_LIBRARY_PATH:-}"
    return 0
}

# ---------------------------------------------------------------------------
# sygaldry_setup_cuda
# ---------------------------------------------------------------------------
# Configures CUDA_HOME, PATH, and LD_LIBRARY_PATH for /usr/local/cuda.
sygaldry_setup_cuda() {
    if [[ -d "/usr/local/cuda" ]]; then
        export CUDA_HOME="/usr/local/cuda"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

# ---------------------------------------------------------------------------
# sygaldry_activate_mlsys_venv
# ---------------------------------------------------------------------------
# Auto-activates baked MLSys venvs. Supports single-venv and multi-venv
# images. Set SYGALDRY_MLSYS_TARGET to pick a specific sub-venv.
# No-op if SYGALDRY_MLSYS_ENV is unset or the venv root doesn't exist.
sygaldry_activate_mlsys_venv() {
    local venv_root="${SYGALDRY_MLSYS_VENV_ROOT:-/opt/mlsys-envs}"
    local env_name="${SYGALDRY_MLSYS_ENV:-}"

    if [[ -z "${env_name}" ]] || [[ ! -d "${venv_root}" ]]; then
        return 0
    fi

    local target="${SYGALDRY_MLSYS_TARGET:-}"
    local venv_dir=""

    if [[ -n "${target}" ]] && [[ -f "${venv_root}/${env_name}/${target}/bin/activate" ]]; then
        venv_dir="${venv_root}/${env_name}/${target}"
    elif [[ -f "${venv_root}/${env_name}/bin/activate" ]]; then
        venv_dir="${venv_root}/${env_name}"
    else
        local first_sub
        first_sub="$(find "${venv_root}/${env_name}" -maxdepth 3 -name activate -path '*/bin/activate' 2>/dev/null | sort | head -1)"
        if [[ -n "${first_sub}" ]]; then
            venv_dir="$(dirname "$(dirname "${first_sub}")")"
            local available
            available="$(ls -d "${venv_root}/${env_name}"/*/bin/activate 2>/dev/null | while read -r p; do basename "$(dirname "$(dirname "$p")")"; done | tr '\n' ' ')"
            echo "Multi-venv image: auto-activating '$(basename "${venv_dir}")'" >&2
            echo "  Available venvs: ${available}" >&2
            echo "  Switch with: SYGALDRY_MLSYS_TARGET=<name> or source <venv>/bin/activate" >&2
        fi
    fi

    if [[ -n "${venv_dir}" ]] && [[ -f "${venv_dir}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${venv_dir}/bin/activate"
        echo "MLSys venv activated: ${venv_dir}" >&2
    fi
    return 0
}

# ---------------------------------------------------------------------------
# sygaldry_require_torch_jax
# ---------------------------------------------------------------------------
# Verifies that torch and jax are importable. Falls back to view if not.
# Exits with error if neither works.
sygaldry_require_torch_jax() {
    if python3 -c 'import importlib.util, sys; [sys.exit(1) for name in ("torch","jax") if importlib.util.find_spec(name) is None]' 2>/dev/null; then
        return 0
    fi

    if sygaldry_ensure_view_fallback; then
        echo "Using /opt/spack_store/view fallback for Python/LD paths." >&2
        return 0
    fi

    error_with_hint \
        "Torch/JAX unavailable and /opt/spack_store/view not found." \
        "Build with spack-build.sh or use a snapshot image."
    return 1
}

# ---------------------------------------------------------------------------
# sygaldry_full_init
# ---------------------------------------------------------------------------
# Complete initialization sequence. Call this from most entrypoints.
# Steps: spack init → activate env → view fallback if needed → CUDA → MLSys venv.
sygaldry_full_init() {
    sygaldry_init_spack || true

    if ! sygaldry_activate_env; then
        echo "WARNING: Could not activate a Spack environment; trying view fallback." >&2
        sygaldry_ensure_view_fallback || true
    fi

    sygaldry_setup_cuda
    sygaldry_activate_mlsys_venv
}
