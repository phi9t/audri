#!/bin/bash
# TAP (Test Anything Protocol) test primitives for Sygaldry verification.
#
# Provides standardized test output across all verification scripts.
# Source this file; do not execute it directly.
#
# TAP spec: https://testanything.org/tap-specification.html
#
# Usage:
#   source container/lib/test_primitives.sh
#   tap_plan 5
#   tap_ok "some test passed"
#   tap_not_ok "some test failed" "expected X, got Y"
#   tap_skip "test skipped" "no GPU available"
#   tap_bail_out "cannot continue"
#   tap_summary  # prints pass/fail counts

if [[ -n "${_SYGALDRY_TAP_SH_LOADED:-}" ]]; then
    return 0
fi
readonly _SYGALDRY_TAP_SH_LOADED=1

_TAP_TEST_NUM=0
_TAP_PASS=0
_TAP_FAIL=0
_TAP_SKIP=0

tap_plan() {
    local count="$1"
    echo "1..${count}"
}

tap_ok() {
    local description="$1"
    ((_TAP_TEST_NUM++)) || true
    ((_TAP_PASS++)) || true
    echo "ok ${_TAP_TEST_NUM} - ${description}"
}

tap_not_ok() {
    local description="$1"
    local detail="${2:-}"
    ((_TAP_TEST_NUM++)) || true
    ((_TAP_FAIL++)) || true
    echo "not ok ${_TAP_TEST_NUM} - ${description}"
    if [[ -n "${detail}" ]]; then
        echo "  # ${detail}"
    fi
}

tap_skip() {
    local description="$1"
    local reason="${2:-}"
    ((_TAP_TEST_NUM++)) || true
    ((_TAP_SKIP++)) || true
    if [[ -n "${reason}" ]]; then
        echo "ok ${_TAP_TEST_NUM} - ${description} # SKIP ${reason}"
    else
        echo "ok ${_TAP_TEST_NUM} - ${description} # SKIP"
    fi
}

tap_bail_out() {
    local reason="${1:-}"
    echo "Bail out! ${reason}"
    exit 1
}

tap_diag() {
    local msg="$1"
    echo "# ${msg}"
}

tap_summary() {
    echo ""
    echo "# Tests: ${_TAP_TEST_NUM}  Pass: ${_TAP_PASS}  Fail: ${_TAP_FAIL}  Skip: ${_TAP_SKIP}"
    if [[ ${_TAP_FAIL} -ne 0 ]]; then
        return 1
    fi
    return 0
}

tap_exit() {
    if [[ ${_TAP_FAIL} -ne 0 ]]; then
        exit 1
    fi
    exit 0
}
