#!/bin/bash
set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
KIT_ROOT="${SCRIPT_DIR}"
readonly KIT_ROOT

DEFAULT_REPO_ROOT="$(cd "${KIT_ROOT}/../.." 2>/dev/null && pwd || pwd)"
readonly DEFAULT_REPO_ROOT
CONFIG_PATH="${ZEPHYR_INFRA_CONFIG:-${KIT_ROOT}/infra.yaml}"

# shellcheck disable=SC1091
source "${KIT_ROOT}/lib/infra_config.sh"

usage() {
    cat <<'USAGE'
Usage:
  build_repo_image.sh [--repo <path>] [--runtime-image <tag>] [--no-cache]

Builds a repo-derived runtime image using Dockerfile.zephyr:
  FROM <base_image_ref>

Values default from infra.yaml:
- base_image_ref
- runtime_image
USAGE
}

log() {
    echo "[build-repo-image] $*" >&2
}

REPO_ROOT="${DEFAULT_REPO_ROOT}"
RUNTIME_IMAGE_OVERRIDE=""
NO_CACHE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_ROOT="${2:-}"
            shift 2
            ;;
        --runtime-image)
            RUNTIME_IMAGE_OVERRIDE="${2:-}"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

REPO_ROOT="$(realpath "${REPO_ROOT}")"
[[ -d "${REPO_ROOT}" ]] || infra_err "Repo path does not exist: ${REPO_ROOT}"
[[ -f "${CONFIG_PATH}" ]] || infra_err "Config not found: ${CONFIG_PATH}"
[[ -f "${KIT_ROOT}/Dockerfile.zephyr" ]] || infra_err "Dockerfile.zephyr not found in ${KIT_ROOT}"

infra_load "${CONFIG_PATH}"

BASE_IMAGE_REF="${ZEPHYR_INFRA_BASE_IMAGE_REF}"
RUNTIME_IMAGE="${RUNTIME_IMAGE_OVERRIDE:-${ZEPHYR_INFRA_RUNTIME_IMAGE}}"

[[ -n "${RUNTIME_IMAGE}" ]] || infra_err "runtime_image is empty; set it in infra.yaml or pass --runtime-image"

repo_rev="unknown"
if command -v git >/dev/null 2>&1; then
    repo_rev="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi
created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

build_args=(
    --file "${KIT_ROOT}/Dockerfile.zephyr"
    --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF}"
    --build-arg "REPO_SOURCE_REV=${repo_rev}"
    --build-arg "REPO_IMAGE_CREATED_AT=${created_at}"
    --tag "${RUNTIME_IMAGE}"
)
if [[ ${NO_CACHE} -eq 1 ]]; then
    build_args+=(--no-cache)
fi

log "Building runtime image: ${RUNTIME_IMAGE}"
log "Base image ref: ${BASE_IMAGE_REF}"
log "Repo root: ${REPO_ROOT}"

docker build "${build_args[@]}" "${REPO_ROOT}"

log "Build complete"
log "Run verification: ${KIT_ROOT}/bin/repoctl verify image --repo ${REPO_ROOT}"
