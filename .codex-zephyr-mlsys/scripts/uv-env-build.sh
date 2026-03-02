#!/bin/bash
# Build MLSys environments from YAML definitions.
set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [uv-env-build:${BASH_LINENO[0]}] $*" >&2
}

usage() {
    cat <<'USAGE'
Usage:
  uv-env-build.sh <env.yaml|env-name> [--no-validate] [--venv-root /path]

Options:
  --no-validate   Skip validation after build.
  --venv-root     Root directory for venvs (default: /tmp/mlsys-envs).
USAGE
}

resolve_runtime_file() {
    local rel_path="$1"
    local candidates=()

    if [[ -n "${ZEPHYR_MLSYS_RUNTIME_ROOT:-}" ]]; then
        candidates+=("${ZEPHYR_MLSYS_RUNTIME_ROOT}/${rel_path}")
    fi

    candidates+=(
        "${SCRIPT_DIR}/../${rel_path}"
        "${SCRIPT_DIR}/../assets/zephyr-mlsys-runtime/${rel_path}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

ENV_INPUT=""
VENV_ROOT="${VENV_ROOT:-/tmp/mlsys-envs}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-validate)
            SKIP_VALIDATE=1
            shift
            ;;
        --venv-root)
            VENV_ROOT="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            exit 2
            ;;
        *)
            ENV_INPUT="$1"
            shift
            ;;
    esac
done

if [[ -z "${ENV_INPUT}" ]]; then
    usage >&2
    exit 2
fi

ENVS_DIR="${SCRIPT_DIR}/../envs"
if [[ -f "${ENV_INPUT}" ]]; then
    ENV_YAML="${ENV_INPUT}"
elif [[ -f "${ENVS_DIR}/${ENV_INPUT}.yaml" ]]; then
    ENV_YAML="${ENVS_DIR}/${ENV_INPUT}.yaml"
elif [[ -f "${ENVS_DIR}/${ENV_INPUT}" ]]; then
    ENV_YAML="${ENVS_DIR}/${ENV_INPUT}"
else
    echo "ERROR: Cannot find env definition: ${ENV_INPUT}" >&2
    echo "Tried: ${ENV_INPUT}, ${ENVS_DIR}/${ENV_INPUT}.yaml" >&2
    exit 2
fi
readonly ENV_YAML

SPACK_PY="/opt/spack_store/view/bin/python3"
if [[ -x "${SPACK_PY}" ]]; then
    PYTHON_BIN="${SPACK_PY}"
else
    PYTHON_BIN="$(command -v python3 2>/dev/null || echo python3)"
fi

PARSER="${SCRIPT_DIR}/parse-env-yaml.py"
if [[ ! -f "${PARSER}" ]]; then
    echo "ERROR: parser not found: ${PARSER}" >&2
    exit 1
fi

# shellcheck disable=SC1090
# Parse YAML into ENV_* and VALIDATION_* shell vars.
eval "$(${PYTHON_BIN} "${PARSER}" "${ENV_YAML}")"

log "Environment: ${ENV_NAME}"
log "Description: ${ENV_DESCRIPTION}"
log "Venv count: ${VENV_COUNT}"

UV_INSTALL="$(resolve_runtime_file "container_entrypoints/uv-install.sh" || true)"
if [[ -z "${UV_INSTALL}" ]]; then
    echo "ERROR: uv-install.sh not found in runtime kit" >&2
    exit 1
fi
log "Using uv installer: ${UV_INSTALL}"

resolve_override() {
    local name="$1"
    resolve_runtime_file "container_entrypoints/${name}" || true
}

if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    if [[ -d "/opt/uv_cache" ]]; then
        export UV_CACHE_DIR="/opt/uv_cache"
    else
        export UV_CACHE_DIR="/tmp/uv-cache-$$"
        mkdir -p "${UV_CACHE_DIR}"
    fi
fi

BUILT_VENVS=()
BUILD_FAILURES=0

for i in $(seq 0 $((VENV_COUNT - 1))); do
    name_var="VENV_${i}_NAME"
    packages_var="VENV_${i}_PACKAGES"
    overrides_var="VENV_${i}_OVERRIDES"

    venv_name="${!name_var}"
    venv_packages="${!packages_var}"
    venv_overrides="${!overrides_var}"

    if [[ "${VENV_COUNT}" -eq 1 ]]; then
        venv_dir="${VENV_ROOT}/${ENV_NAME}"
    else
        venv_dir="${VENV_ROOT}/${ENV_NAME}/${venv_name}"
    fi

    override_path=""
    if [[ -n "${venv_overrides}" ]]; then
        for override_file in ${venv_overrides}; do
            resolved="$(resolve_override "${override_file}")"
            if [[ -n "${resolved}" ]]; then
                override_path="${resolved}"
                break
            fi
        done
    fi

    log "Building ${venv_name} -> ${venv_dir}"
    log "Packages: ${venv_packages}"
    if [[ -n "${override_path}" ]]; then
        log "Override: ${override_path}"
    fi

    if (
        export VENV_DIR="${venv_dir}"
        export CONSTRAINTS_FILE="/tmp/spack-constraints-${ENV_NAME}-${venv_name}.txt"
        if [[ -n "${override_path}" ]]; then
            export UV_EXTRA_OVERRIDES="${override_path}"
        else
            unset UV_EXTRA_OVERRIDES
        fi

        # shellcheck disable=SC2086
        "${UV_INSTALL}" ${venv_packages}
    ); then
        BUILT_VENVS+=("${venv_dir}")
        log "OK: ${venv_name}"
    else
        ((BUILD_FAILURES++)) || true
        log "FAIL: ${venv_name}"
    fi
done

if [[ ${BUILD_FAILURES} -gt 0 ]]; then
    log "ERROR: ${BUILD_FAILURES} build(s) failed"
    exit 1
fi

VALIDATOR="${SCRIPT_DIR}/uv-env-validate.sh"
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
    log "Validation skipped (--no-validate)"
elif [[ ! -f "${VALIDATOR}" ]]; then
    log "WARNING: validator not found: ${VALIDATOR}"
else
    log "Running validation"
    "${VALIDATOR}" "${ENV_YAML}" --venv-root "${VENV_ROOT}"
fi

echo ""
echo "================================================================"
echo " MLSys Environment Build Summary"
echo "================================================================"
echo " Environment:  ${ENV_NAME}"
echo " Description:  ${ENV_DESCRIPTION}"
echo " Venvs built:  ${#BUILT_VENVS[@]}"
for venv in "${BUILT_VENVS[@]}"; do
    echo "   - ${venv}"
done
echo " Validated:    $([ "${SKIP_VALIDATE}" == "1" ] && echo skipped || echo yes)"
echo "================================================================"
echo ""
if [[ ${#BUILT_VENVS[@]} -eq 1 ]]; then
    echo "Activate with: source ${BUILT_VENVS[0]}/bin/activate"
else
    echo "Activate with:"
    for venv in "${BUILT_VENVS[@]}"; do
        name="$(basename "${venv}")"
        echo "  source ${SCRIPT_DIR}/uv-env-activate.sh ${ENV_NAME} --target ${name}"
    done
fi
