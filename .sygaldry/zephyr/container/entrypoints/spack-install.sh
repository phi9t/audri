#!/bin/bash
#
# Spack install entrypoint
# ========================
#
# Runs `spack install` (with any provided args) after initializing Spack,
# then drops into an interactive shell.

set -euo pipefail

if [[ "${SYGALDRY_BUILD_ROLE:-consumer}" != "builder" ]]; then
    echo "ERROR: spack install is restricted to builder role." >&2
    echo "HINT:  Set SYGALDRY_BUILD_ROLE=builder in the sygaldry build repo to proceed." >&2
    exit 1
fi

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)" \
    || _LIB_DIR="/opt/container_entrypoints/../lib"
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

if ! sygaldry_init_spack; then
    echo "ERROR: Spack setup script not found at /opt/spack_src" >&2
    echo "HINT:  Are you inside the container? Use sygaldry or launch_container.sh." >&2
    exit 1
fi

if [[ -d "/workspace" ]]; then
    cd /workspace
fi

spack install "$@"
exec bash -i
