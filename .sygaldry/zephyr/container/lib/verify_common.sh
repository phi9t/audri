#!/bin/bash
# Shared helper utilities for Zephyr verification scripts.
#
# Provides both legacy pass/fail output and TAP primitives.
# New scripts should prefer the TAP functions from test_primitives.sh.

if [[ -n "${ZEPHYR_VERIFY_COMMON_SH_LOADED:-}" ]]; then
    return 0
fi
readonly ZEPHYR_VERIFY_COMMON_SH_LOADED=1

# Load TAP primitives (available for scripts that want standard TAP)
_VERIFY_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_VERIFY_LIB_DIR}/test_primitives.sh"

# Legacy counters (for backward compat with existing verify scripts)
verify_reset_counters() {
    TESTS_RUN=0
    TESTS_PASSED=0
    FAILURES=0
}

pass() {
    ((TESTS_PASSED++)) || true
    ((TESTS_RUN++)) || true
    echo "  PASS: $1"
}

fail() {
    ((FAILURES++)) || true
    ((TESTS_RUN++)) || true
    echo "  FAIL: $1"
    if [[ -n "${2:-}" ]]; then
        echo "        $2"
    fi
}

verify_require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker not found." >&2
        return 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker daemon not running." >&2
        return 1
    fi
    return 0
}

verify_require_image() {
    local image="$1"
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        echo "ERROR: Image ${image} not found. Build or pull it first." >&2
        return 1
    fi
    return 0
}

verify_image_size_gb() {
    local image="$1"
    docker image inspect "${image}" --format='{{.Size}}' \
        | awk '{printf "%.1f GB", $1/1073741824}'
}

verify_has_nvidia_runtime() {
    docker info 2>/dev/null | grep -q nvidia
}

verify_run_no_gpu() {
    local image="$1"
    local cmd="$2"
    timeout 300 docker run --rm --entrypoint /bin/bash "${image}" -c "${cmd}"
}

verify_run_with_gpu() {
    local image="$1"
    local cmd="$2"
    timeout 300 docker run --rm --runtime=nvidia --gpus=all --entrypoint /bin/bash "${image}" -c "${cmd}"
}

verify_print_summary() {
    local title="$1"
    local image="${2:-}"
    echo ""
    echo "========================================"
    echo " ${title}"
    echo "========================================"
    if [[ -n "${image}" ]]; then
        echo " Image:        ${image}"
    fi
    echo " Tests run:    ${TESTS_RUN}"
    echo " Passed:       ${TESTS_PASSED}"
    echo " Failed:       ${FAILURES}"
    echo "========================================"
}

verify_exit_on_failures() {
    if [[ ${FAILURES} -ne 0 ]]; then
        exit 1
    fi
    exit 0
}
