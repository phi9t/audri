#!/bin/bash
#
# uv-env-validate.sh — Data-driven validation for MLSys environments
# ====================================================================
#
# Validates venvs built by uv-env-build.sh against the checks defined
# in the YAML env definition. Produces TAP-style output.
#
# Usage:
#   uv-env-validate.sh <env.yaml>                         # Validate built venvs
#   uv-env-validate.sh <env.yaml> --venv-root /path       # Custom venv root
#   uv-env-validate.sh <env.yaml> --no-gpu                # Skip GPU tests
#
# Checks performed:
#   1. Spack provenance — core packages resolve from /opt/spack_store/view
#   2. UV provenance — installed packages resolve from venv
#   3. nvidia-* pip leak — no NVIDIA pip packages outside Spack view
#   4. GPU functional — Python scripts from YAML definition
#   5. Hard-fail pattern matching — undefined symbol, ABI mismatch detection
#
# Output format: TAP (Test Anything Protocol)
#   ok 1 - torch from Spack view
#   not ok 2 - nvidia-cublas-cu12 found in venv
#   ok 3 - vllm import + CUDA functional
#
# Exit codes:
#   0 - All checks passed (or only soft failures)
#   1 - Hard failure detected
#   2 - Usage error

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# ============================================================================
# Logging
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [uv-env-validate:${BASH_LINENO[0]}] $*" >&2
}

# ============================================================================
# Argument parsing
# ============================================================================

ENV_INPUT=""
VENV_ROOT="${VENV_ROOT:-/tmp/mlsys-envs}"
GPU_TESTS=true
GPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-root)
            VENV_ROOT="$2"
            shift 2
            ;;
        --no-gpu)
            GPU_TESTS=false
            shift
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: uv-env-validate.sh <env.yaml|env-name> [--venv-root /path] [--no-gpu] [--gpu-only]"
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
    echo "Usage: uv-env-validate.sh <env.yaml|env-name>" >&2
    exit 2
fi

# ============================================================================
# Resolve env YAML
# ============================================================================

ENVS_DIR="${SCRIPT_DIR}/../envs"

if [[ -f "${ENV_INPUT}" ]]; then
    ENV_YAML="${ENV_INPUT}"
elif [[ -f "${ENVS_DIR}/${ENV_INPUT}.yaml" ]]; then
    ENV_YAML="${ENVS_DIR}/${ENV_INPUT}.yaml"
elif [[ -f "${ENVS_DIR}/${ENV_INPUT}" ]]; then
    ENV_YAML="${ENVS_DIR}/${ENV_INPUT}"
else
    echo "ERROR: Cannot find env definition: ${ENV_INPUT}" >&2
    exit 2
fi

readonly ENV_YAML

# ============================================================================
# Find Python
# ============================================================================

SPACK_PY="/opt/spack_store/view/bin/python3"
if [[ -x "${SPACK_PY}" ]]; then
    PYTHON_BIN="${SPACK_PY}"
else
    PYTHON_BIN=$(command -v python3 2>/dev/null || echo "python3")
fi

# ============================================================================
# Parse YAML
# ============================================================================

PARSER="${SCRIPT_DIR}/parse-env-yaml.py"
[[ -f "${PARSER}" ]] || {
    echo "ERROR: parser not found: ${PARSER}" >&2
    exit 1
}
eval "$("${PYTHON_BIN}" "${PARSER}" "${ENV_YAML}")"

log "Validating environment: ${ENV_NAME}"

# ============================================================================
# TAP harness
# ============================================================================

TEST_NUM=0
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
HARD_FAIL=false

tap_ok() {
    ((TEST_NUM++)) || true
    ((PASS_COUNT++)) || true
    echo "ok ${TEST_NUM} - $1"
}

tap_not_ok() {
    ((TEST_NUM++)) || true
    ((FAIL_COUNT++)) || true
    echo "not ok ${TEST_NUM} - $1"
    if [[ -n "${2:-}" ]]; then
        echo "  ---"
        echo "  detail: $2"
        echo "  ..."
    fi
}

tap_warn() {
    ((TEST_NUM++)) || true
    ((WARN_COUNT++)) || true
    echo "ok ${TEST_NUM} - WARN: $1 # TODO soft failure"
}

# ============================================================================
# Determine venv paths
# ============================================================================

# For single-venv envs, venv is at VENV_ROOT/ENV_NAME
# For multi-venv envs, venvs are at VENV_ROOT/ENV_NAME/VENV_NAME
get_venv_dir() {
    local venv_name="$1"
    if [[ "${VENV_COUNT}" -eq 1 ]]; then
        echo "${VENV_ROOT}/${ENV_NAME}"
    else
        echo "${VENV_ROOT}/${ENV_NAME}/${venv_name}"
    fi
}

# For provenance checks, we need to pick the right venv for each package.
# For single-venv: always use it. For multi-venv: find which venv has it.
find_venv_for_package() {
    local pkg="$1"
    for i in $(seq 0 $((VENV_COUNT - 1))); do
        local name_var="VENV_${i}_NAME"
        local venv_name="${!name_var}"
        local venv_dir
        venv_dir="$(get_venv_dir "${venv_name}")"
        if [[ -d "${venv_dir}" ]]; then
            # Check if package is importable from this venv
            local site_dir="${venv_dir}/lib/python*/site-packages"
            # shellcheck disable=SC2086
            if ls ${site_dir}/${pkg}* >/dev/null 2>&1 || \
               ls ${site_dir}/${pkg//-/_}* >/dev/null 2>&1; then
                echo "${venv_dir}"
                return 0
            fi
        fi
    done
    # Default to first venv
    local first_name_var="VENV_0_NAME"
    get_venv_dir "${!first_name_var}"
}

echo "TAP version 13"
echo ""
echo "# Environment: ${ENV_NAME}"
echo "# Description: ${ENV_DESCRIPTION}"
echo ""

# ============================================================================
# Check 1: Spack provenance (skipped in --gpu-only mode)
# ============================================================================

if [[ "${GPU_ONLY}" == "true" ]]; then
    echo "# --- Spack provenance (skipped: --gpu-only) ---"
else
echo "# --- Spack provenance ---"

for pkg in ${VALIDATION_SPACK_PROVENANCE}; do
    mod_name="${pkg//-/_}"
    # Use first available venv for activation context
    first_venv_var="VENV_0_NAME"
    first_venv="$(get_venv_dir "${!first_venv_var}")"

    if [[ ! -d "${first_venv}" ]]; then
        tap_not_ok "${pkg} from Spack view" "venv not found: ${first_venv}"
        continue
    fi

    result="$(
        source "${first_venv}/bin/activate" 2>/dev/null
        "${PYTHON_BIN}" -c "
import ${mod_name}
f = getattr(${mod_name}, '__file__', '') or ''
if '/opt/spack_store/view' in f:
    print('SPACK_OK')
else:
    print('NOT_SPACK: ' + f)
" 2>&1 | tail -1
    )" || result="ERROR"

    if [[ "${result}" == "SPACK_OK" ]]; then
        tap_ok "${pkg} from Spack view"
    else
        tap_not_ok "${pkg} from Spack view" "${result}"
    fi
done

# ============================================================================
# Check 2: UV provenance
# ============================================================================

echo ""
echo "# --- UV provenance ---"

for pkg in ${VALIDATION_UV_PROVENANCE}; do
    mod_name="${pkg//-/_}"
    venv_dir="$(find_venv_for_package "${pkg}")"

    if [[ ! -d "${venv_dir}" ]]; then
        tap_not_ok "${pkg} from UV venv" "venv not found: ${venv_dir}"
        continue
    fi

    # Use the venv's Python directly (not PYTHON_BIN/Spack Python) to ensure
    # we see the venv's site-packages, not just system site-packages.
    venv_python="${venv_dir}/bin/python3"
    result="$(
        "${venv_python}" -c "
import ${mod_name}
f = getattr(${mod_name}, '__file__', '') or ''
if '${venv_dir}' in f:
    print('UV_OK')
elif '/opt/spack_store/view' in f:
    print('SPACK_VIEW: ' + f)
else:
    print('UNKNOWN: ' + f)
" 2>&1 | tail -1
    )" || result="ERROR"

    if [[ "${result}" == "UV_OK" ]]; then
        tap_ok "${pkg} from UV venv"
    else
        tap_not_ok "${pkg} from UV venv" "${result}"
    fi
done

# ============================================================================
# Check 3: No nvidia-* pip packages
# ============================================================================

if [[ "${VALIDATION_NO_NVIDIA_PIP}" == "true" ]]; then
    echo ""
    echo "# --- NVIDIA pip leak check ---"

    for i in $(seq 0 $((VENV_COUNT - 1))); do
        name_var="VENV_${i}_NAME"
        venv_name="${!name_var}"
        venv_dir="$(get_venv_dir "${venv_name}")"

        if [[ ! -d "${venv_dir}" ]]; then
            tap_not_ok "No nvidia-* pip in ${venv_name}" "venv not found"
            continue
        fi

        result="$(
            source "${venv_dir}/bin/activate" 2>/dev/null
            "${PYTHON_BIN}" -c "
import importlib.metadata as md
nvidia_pkgs = []
dsl_pkgs = []
for dist in md.distributions():
    name = (dist.metadata.get('Name') or '').lower()
    if name.startswith('nvidia-'):
        loc = str(getattr(dist, '_path', 'unknown'))
        if '/opt/spack_store/view' not in loc:
            # Known acceptable DSL wrappers
            if name in ('nvidia-cudnn-frontend', 'nvidia-cutlass-dsl'):
                dsl_pkgs.append(name)
            else:
                nvidia_pkgs.append(name)
if nvidia_pkgs:
    print('NVIDIA_FOUND: ' + ','.join(nvidia_pkgs))
elif dsl_pkgs:
    print('DSL_ONLY: ' + ','.join(dsl_pkgs))
else:
    print('NO_NVIDIA_OK')
" 2>&1 | tail -1
        )" || result="ERROR"

        if [[ "${result}" == "NO_NVIDIA_OK" ]]; then
            tap_ok "No nvidia-* pip in ${venv_name}"
        elif [[ "${result}" == DSL_ONLY:* ]]; then
            dsl_list="${result#DSL_ONLY: }"
            if echo " ${VALIDATION_SOFT_FAIL_ON} " | grep -q " nvidia_dsl_leak "; then
                tap_warn "nvidia DSL wrappers in ${venv_name}: ${dsl_list}"
            else
                tap_not_ok "nvidia DSL wrappers in ${venv_name}" "${dsl_list}"
            fi
        else
            tap_not_ok "nvidia-* pip leak in ${venv_name}" "${result}"
            if echo " ${VALIDATION_HARD_FAIL_ON} " | grep -q " core_conflict "; then
                HARD_FAIL=true
            fi
        fi
    done
fi

fi  # end of GPU_ONLY skip

# ============================================================================
# Check 4: GPU functional tests
# ============================================================================

if [[ "${GPU_TESTS}" != "true" ]]; then
    echo ""
    echo "# --- GPU tests skipped (--no-gpu) ---"
elif [[ "${VALIDATION_GPU_SCRIPT_COUNT}" -gt 0 ]]; then
    echo ""
    echo "# --- GPU functional tests ---"

    for i in $(seq 0 $((VALIDATION_GPU_SCRIPT_COUNT - 1))); do
        name_var="VALIDATION_GPU_SCRIPT_${i}_NAME"
        script_var="VALIDATION_GPU_SCRIPT_${i}"
        patterns_var="VALIDATION_GPU_SCRIPT_${i}_HARD_FAIL_PATTERNS"

        test_name="${!name_var}"
        test_script="${!script_var}"
        hard_fail_patterns="${!patterns_var}"

        # Find appropriate venv — for multi-venv, match test name to venv name
        target_venv=""
        for j in $(seq 0 $((VENV_COUNT - 1))); do
            vn_var="VENV_${j}_NAME"
            vn="${!vn_var}"
            if [[ "${test_name}" == *"${vn}"* ]] || [[ "${VENV_COUNT}" -eq 1 ]]; then
                target_venv="$(get_venv_dir "${vn}")"
                break
            fi
        done
        if [[ -z "${target_venv}" ]]; then
            # Fallback to first venv
            first_var="VENV_0_NAME"
            target_venv="$(get_venv_dir "${!first_var}")"
        fi

        if [[ ! -d "${target_venv}" ]]; then
            tap_not_ok "${test_name}" "venv not found: ${target_venv}"
            continue
        fi

        # Write test script to temp file
        tmp_script="/tmp/gpu-test-${ENV_NAME}-${test_name}.py"
        # Expand escaped newlines back to real newlines
        echo -e "${test_script}" > "${tmp_script}"

        # Use venv Python so test scripts can import venv-installed packages
        result="$(
            "${target_venv}/bin/python3" "${tmp_script}" 2>&1
        )" || true

        rm -f "${tmp_script}"

        # Check for hard-fail patterns
        pattern_matched=false
        if [[ -n "${hard_fail_patterns}" ]]; then
            for pattern in ${hard_fail_patterns}; do
                if echo "${result}" | grep -qi "${pattern}"; then
                    pattern_matched=true
                    tap_not_ok "${test_name}" "HARD_FAIL pattern '${pattern}' matched in output"
                    if echo " ${VALIDATION_HARD_FAIL_ON} " | grep -q " abi_mismatch "; then
                        HARD_FAIL=true
                    fi
                    break
                fi
            done
        fi

        if [[ "${pattern_matched}" == "false" ]]; then
            # Check if the last line indicates success (not an error/traceback)
            last_line="$(echo "${result}" | tail -1)"
            if echo "${result}" | grep -qE "(Error|Traceback|assert|FAIL)"; then
                tap_not_ok "${test_name}" "${last_line}"
            else
                tap_ok "${test_name}: ${last_line}"
            fi
        fi
    done
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "1..${TEST_NUM}"
echo ""
echo "# =============================="
echo "# Validation Summary: ${ENV_NAME}"
echo "# =============================="
echo "# Tests:    ${TEST_NUM}"
echo "# Passed:   ${PASS_COUNT}"
echo "# Failed:   ${FAIL_COUNT}"
echo "# Warnings: ${WARN_COUNT}"
if [[ "${HARD_FAIL}" == "true" ]]; then
    echo "# HARD FAIL: Critical safety check failed"
fi
echo "# =============================="

if [[ "${HARD_FAIL}" == "true" ]]; then
    exit 1
fi

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    exit 1
fi

exit 0
