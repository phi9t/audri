#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/zephyr_layer.sh <env-name> <package...>
  scripts/zephyr_layer.sh <env-name> "<package1 package2 ...>"

Example:
  scripts/zephyr_layer.sh qwen3-tts "sounddevice"
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

ENV_NAME="$1"
shift

if [[ $# -eq 1 ]] && [[ "$1" == *" "* ]]; then
  read -r -a PKGS <<<"$1"
else
  PKGS=("$@")
fi

if [[ ${#PKGS[@]} -eq 0 ]]; then
  echo "ERROR: at least one package is required" >&2
  exit 2
fi

normalize_pkg() {
  local raw="$1"
  raw="${raw%%[*}"
  raw="${raw%%[<>=!~]*}"
  echo "${raw,,}"
}

is_spack_owned() {
  local pkg="$1"
  case "$pkg" in
    torch|torchaudio|torchvision|jax|jaxlib|triton|numpy|scipy|numba|llvmlite|python|python3)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

for pkg in "${PKGS[@]}"; do
  base="$(normalize_pkg "$pkg")"
  if is_spack_owned "$base"; then
    echo "ERROR: '$pkg' is Spack-owned and must not be installed via uv layering." >&2
    echo "HINT: Remove it from extra packages and rely on /opt/spack_store/view." >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOCTL="${REPO_ROOT}/.sygaldry/zephyr/bin/repoctl"
UV_INSTALL="/workspace/audri/.codex-zephyr-mlsys/container_entrypoints/uv-install.sh"
VENV_DIR="/workspace/audri/.venv-mlsys/${ENV_NAME}"

if [[ ! -x "${REPOCTL}" ]]; then
  echo "ERROR: repoctl not found: ${REPOCTL}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/.codex-zephyr-mlsys/container_entrypoints/uv-install.sh" ]]; then
  echo "ERROR: uv installer not found in vendored runtime kit" >&2
  exit 1
fi

printf -v PKG_ARGS "%q " "${PKGS[@]}"
CMD=$(cat <<EOF
set -euo pipefail
export VENV_DIR="${VENV_DIR}"
"${UV_INSTALL}" ${PKG_ARGS}
EOF
)

"${REPOCTL}" run -- bash -lc "${CMD}"

echo "Layered packages installed into ${VENV_DIR}"
echo "Use with:"
echo "  .sygaldry/zephyr/bin/repoctl run -- bash -lc 'source ${VENV_DIR}/bin/activate && python3 /workspace/audri/scripts/smoke_test_tts.py'"
