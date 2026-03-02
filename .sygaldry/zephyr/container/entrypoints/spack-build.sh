#!/bin/bash
set -euo pipefail

if [[ "${SYGALDRY_BUILD_ROLE:-consumer}" != "builder" ]]; then
    echo "ERROR: Spack builds are restricted to the builder role." >&2
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

# Resolve build environment root.
SPACK_ENV_ROOT="${SYGALDRY_SPACK_ENV:-/opt/spack_env/default}"
if [[ ! -d "${SPACK_ENV_ROOT}" ]]; then
    SPACK_ENV_ROOT="/opt/spack_env/zephyr"
fi
if [[ ! -d "${SPACK_ENV_ROOT}" ]]; then
    SPACK_ENV_ROOT="${SYGALDRY_ROOT:-/workspace}/pkg/zephyr"
fi
cd "${SPACK_ENV_ROOT}"
./build.sh
