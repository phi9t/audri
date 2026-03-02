#!/bin/bash
set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
KIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly KIT_ROOT
REPO_ROOT="$(cd "${KIT_ROOT}/.." && pwd)"
readonly REPO_ROOT
RUNTIME_CONFIG="${KIT_ROOT}/runtime.yaml"
readonly RUNTIME_CONFIG

usage() {
    cat <<'USAGE'
Usage:
  launch-mlsys.sh <env-name|env-file> [uv-env-build options...]

Env vars:
  SYGALDRY_SNAPSHOT_REF  Override snapshot ref from runtime.yaml.
  MLSYS_VENV_ROOT        Venv root in container (default: /tmp/mlsys-envs).
  MLSYS_DISABLE_GPU      Set to 1 to skip GPU docker flags.
USAGE
}

read_yaml_value() {
    local file="$1"
    local key="$2"
    awk -F ':' -v want="${key}" '
        $0 ~ "^[[:space:]]*" want "[[:space:]]*:[[:space:]]*" {
            sub("^[[:space:]]*" want "[[:space:]]*:[[:space:]]*", "", $0)
            sub(/[[:space:]]+#.*/, "", $0)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
            gsub(/^"|"$/, "", $0)
            print $0
            exit
        }
    ' "${file}"
}

require_digest_ref() {
    local ref="$1"
    [[ -n "${ref}" ]] || {
        echo "ERROR: snapshot ref is empty" >&2
        exit 1
    }
    [[ "${ref}" == *@sha256:* ]] || {
        echo "ERROR: snapshot ref must be digest-pinned: ${ref}" >&2
        exit 1
    }
    local digest
    digest="${ref##*@sha256:}"
    [[ "${digest}" =~ ^[a-f0-9]{64}$ ]] || {
        echo "ERROR: invalid digest in snapshot ref: ${digest}" >&2
        exit 1
    }
}

ENV_INPUT="${1:-}"
if [[ -z "${ENV_INPUT}" ]] || [[ "${ENV_INPUT}" == "--help" ]] || [[ "${ENV_INPUT}" == "-h" ]]; then
    usage
    exit 2
fi
shift || true

[[ -f "${RUNTIME_CONFIG}" ]] || {
    echo "ERROR: runtime config not found: ${RUNTIME_CONFIG}" >&2
    exit 1
}

SNAPSHOT_REF="${SYGALDRY_SNAPSHOT_REF:-}"
if [[ -z "${SNAPSHOT_REF}" ]]; then
    SNAPSHOT_REF="$(read_yaml_value "${RUNTIME_CONFIG}" snapshot_ref)"
fi
require_digest_ref "${SNAPSHOT_REF}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found" >&2
    exit 1
fi

docker_args=(--rm)
if [[ "${MLSYS_DISABLE_GPU:-0}" != "1" ]]; then
    docker_args+=(--runtime=nvidia --gpus=all)
fi

docker run "${docker_args[@]}" \
    -v "${KIT_ROOT}:/opt/codex-zephyr-mlsys:ro" \
    -v "${REPO_ROOT}:/repo" \
    -w /repo \
    "${SNAPSHOT_REF}" \
    /opt/codex-zephyr-mlsys/scripts/uv-env-build.sh \
    "${ENV_INPUT}" \
    --venv-root "${MLSYS_VENV_ROOT:-/tmp/mlsys-envs}" \
    "$@"
