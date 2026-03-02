#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KIT_ROOT="${REPO_ROOT}/.codex-zephyr-mlsys"

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/launch_mlsys_hf.sh <env-name|env-file> [uv-env-build options...]

Env vars:
  SYGALDRY_MLSYS_IMAGE  Container image for MLSys builds
                        (default: ghcr.io/phi9t/sygaldry/zephyr:hf-20260226)
  MLSYS_VENV_ROOT       Venv root path inside container (default: /tmp/mlsys-envs)
USAGE
}

ENV_INPUT="${1:-}"
if [[ -z "${ENV_INPUT}" ]] || [[ "${ENV_INPUT}" == "--help" ]] || [[ "${ENV_INPUT}" == "-h" ]]; then
  usage
  exit 2
fi
shift || true

IMAGE="${SYGALDRY_MLSYS_IMAGE:-ghcr.io/phi9t/sygaldry/zephyr:hf-20260226}"
VENV_ROOT="${MLSYS_VENV_ROOT:-/tmp/mlsys-envs}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found" >&2
  exit 1
fi

docker run --rm --runtime=nvidia --gpus=all --user root \
  -v "${KIT_ROOT}:/opt/codex-zephyr-mlsys:ro" \
  -v "${REPO_ROOT}:/workspace/audri" \
  -w /workspace/audri \
  "${IMAGE}" \
  bash -c '
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends libsndfile1 libportaudio2 portaudio19-dev
  /opt/codex-zephyr-mlsys/scripts/uv-env-build.sh "$1" --venv-root "$2" "${@:3}"
' _ "${ENV_INPUT}" "${VENV_ROOT}" "$@"
