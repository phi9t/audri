#!/bin/bash
# Shared error and logging helpers for Sygaldry container scripts.
#
# Provides consistent error formatting with remediation hints.
# Source this file; do not execute it directly.

if [[ -n "${_SYGALDRY_ERRORS_SH_LOADED:-}" ]]; then
    return 0
fi
readonly _SYGALDRY_ERRORS_SH_LOADED=1

# error_with_hint <message> <hint>
# Outputs:
#   ERROR: <message>
#   HINT:  <hint>
error_with_hint() {
    local msg="$1"
    local hint="${2:-}"
    echo "ERROR: ${msg}" >&2
    if [[ -n "${hint}" ]]; then
        echo "HINT:  ${hint}" >&2
    fi
}

# die_with_hint <message> <hint> [exit_code]
# Same as error_with_hint but exits.
die_with_hint() {
    local msg="$1"
    local hint="${2:-}"
    local code="${3:-1}"
    error_with_hint "${msg}" "${hint}"
    exit "${code}"
}

# sygaldry_log [prefix] <message>
# Timestamped log line to stderr.
sygaldry_log() {
    local prefix=""
    if [[ $# -gt 1 ]]; then
        prefix="[${1}] "
        shift
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ${prefix}$*" >&2
}
