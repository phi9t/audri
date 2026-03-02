#!/bin/bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_LIB_DIR=""
for _candidate in \
    "${_SCRIPT_DIR}/../lib" \
    "/opt/lib" \
    "/opt/container_entrypoints/../lib"; do
    if [[ -f "${_candidate}/spack_init.sh" ]]; then
        _LIB_DIR="${_candidate}"
        break
    fi
done
if [[ -z "${_LIB_DIR}" ]]; then
    echo "ERROR: runtime kit incomplete: missing lib/spack_init.sh" >&2
    echo "HINT: Re-vendor MLSys runtime (zephyr_mlsys_vendor.sh update)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

# Source Spack environment (for spack env activate, if needed)
sygaldry_init_spack || true

# Activate Spack environment if available
if [[ -f "spack.yaml" ]] || [[ -f "spack.lock" ]]; then
    spack env activate . 2>/dev/null || true
fi

# Find Python - prefer Spack view Python
SPACK_PY="/opt/spack_store/view/bin/python3"
if [[ -x "${SPACK_PY}" ]]; then
    PYTHON_BIN="${SPACK_PY}"
else
    PYTHON_BIN=$(command -v python3 2>/dev/null || echo "python3")
fi

# ---------------------------------------------------------------------------
# Read Spack-owned package list from canonical config (with hardcoded fallback)
# ---------------------------------------------------------------------------
SPACK_OWNED_CONF=""
for candidate in \
    /opt/container_entrypoints/spack_owned_packages.conf \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../spack_owned_packages.conf"; do
    if [[ -f "${candidate}" ]]; then
        SPACK_OWNED_CONF="${candidate}"
        break
    fi
done

if [[ -n "${SPACK_OWNED_CONF}" ]]; then
    SPACK_PKGS=$(grep -v '^\s*#' "${SPACK_OWNED_CONF}" | grep -v '^\s*$' | tr '\n' ',' | sed 's/,$//')
else
    SPACK_PKGS="torch,torchvision,torchaudio,jax,jaxlib,triton,numpy,scipy,scikit-learn,numba,llvmlite,matplotlib,pandas,soundfile,jupyterlab"
fi

export UV_NO_BUILD_ISOLATION_PACKAGE="${SPACK_PKGS}"

# ---------------------------------------------------------------------------
# Locate NVIDIA override file
# ---------------------------------------------------------------------------
NVIDIA_OVERRIDES=""
for candidate in \
    /opt/container_entrypoints/nvidia_overrides.txt \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../nvidia_overrides.txt"; do
    if [[ -f "${candidate}" ]]; then
        NVIDIA_OVERRIDES="${candidate}"
        break
    fi
done

if [[ $# -lt 1 ]]; then
    echo "Usage: uv-install.sh <package> [package ...]" >&2
    echo ""
    echo "Creates a venv and installs packages using uv."
    echo "Uses Spack Python + constraints to avoid overriding Spack packages."
    exit 2
fi

export PATH="/usr/local:/usr/local/bin:${PATH}"
unset PYTHONPATH

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found in PATH" >&2
    echo "HINT:  uv is installed in the container image; ensure you are inside." >&2
    exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-/tmp/spack-constraints.txt}"

uv venv --python "${PYTHON_BIN}" --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

PY_VER="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
SITE_PACKAGES="${VENV_DIR}/lib/python${PY_VER}/site-packages"
mkdir -p "${SITE_PACKAGES}"
if [[ -d "/opt/spack_store/view/lib/python${PY_VER}/site-packages" ]]; then
    echo "/opt/spack_store/view/lib/python${PY_VER}/site-packages" > "${SITE_PACKAGES}/spack-view.pth"
else
    echo "WARNING: Spack view site-packages not found for python${PY_VER}" >&2
fi

"${PYTHON_BIN}" - <<PY > "${CONSTRAINTS_FILE}"
import importlib.metadata as md, os, re
spack_owned = set()
conf_path = "${SPACK_OWNED_CONF}"
if conf_path and os.path.isfile(conf_path):
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                spack_owned.add(line.lower())
pins = {}
for dist in sorted(md.distributions(), key=lambda d: (d.metadata.get("Name", "").lower(), d.version)):
    name = dist.metadata.get("Name")
    if name and name.lower() in spack_owned:
        # Keep a single pin per package name to avoid unsat duplicate constraints.
        pins[name.lower()] = (name, re.sub(r'\+.*$', '', dist.version))
for _, (name, version) in sorted(pins.items()):
    print(f"{name}=={version}")
PY

UV_INSTALL_ARGS=(pip install --constraint "${CONSTRAINTS_FILE}")
if [[ -n "${NVIDIA_OVERRIDES}" ]]; then
    UV_INSTALL_ARGS+=(--override "${NVIDIA_OVERRIDES}")
fi
if [[ -n "${UV_EXTRA_OVERRIDES:-}" ]] && [[ -f "${UV_EXTRA_OVERRIDES}" ]]; then
    UV_INSTALL_ARGS+=(--override "${UV_EXTRA_OVERRIDES}")
fi
UV_INSTALL_ARGS+=("$@")

uv "${UV_INSTALL_ARGS[@]}"

echo "Installed packages in ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
