#!/bin/bash
set -euo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)" \
    || _LIB_DIR="/opt/container_entrypoints/../lib"
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

sygaldry_full_init

if [[ $# -lt 1 ]]; then
    echo "Usage: run-job.sh <command...>" >&2
    exit 2
fi

# Avoid cross-version contamination when job commands activate their own venvs.
if [[ "${SYGALDRY_CLEAR_PYTHONPATH_ON_EXEC:-1}" == "1" ]]; then
    unset PYTHONPATH
fi

exec "$@"
