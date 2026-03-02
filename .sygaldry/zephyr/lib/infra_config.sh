#!/bin/bash
set -eu -o pipefail

infra_err() {
    echo "[infra-config] ERROR: $*" >&2
    exit 1
}

infra_read_value() {
    local file="$1"
    local key="$2"
    awk -F ':' -v want="${key}" '
        $0 ~ "^[[:space:]]*" want "[[:space:]]*:[[:space:]]*" {
            sub("^[[:space:]]*" want "[[:space:]]*:[[:space:]]*", "", $0)
            sub(/[[:space:]]+#.*/, "", $0)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
            gsub(/^"|"$/, "", $0)
            gsub(/^\047|\047$/, "", $0)
            print $0
            exit
        }
    ' "${file}"
}

infra_require_digest_pin() {
    local image_ref="$1"
    local label="${2:-image_ref}"
    if [[ -z "${image_ref}" ]]; then
        infra_err "${label} is empty"
    fi
    if [[ "${image_ref}" != *@sha256:* ]]; then
        infra_err "${label} must be digest-pinned (<repo>:<tag>@sha256:<digest>): ${image_ref}"
    fi
    local digest_part
    digest_part="${image_ref##*@sha256:}"
    if [[ ! "${digest_part}" =~ ^[a-f0-9]{64}$ ]]; then
        infra_err "${label} digest must be 64 lowercase hex chars: ${image_ref}"
    fi
}

infra_require_mode() {
    local image_mode="$1"
    case "${image_mode}" in
        standard|auto|derived)
            ;;
        *)
            infra_err "image_mode must be one of standard|auto|derived, got '${image_mode}'"
            ;;
    esac
}

infra_image_exists() {
    local image_ref="$1"
    command -v docker >/dev/null 2>&1 || return 1
    docker image inspect "${image_ref}" >/dev/null 2>&1
}

infra_select_effective_image() {
    local allow_missing_runtime="${1:-0}"

    local effective_image=""
    local effective_source=""

    case "${ZEPHYR_INFRA_IMAGE_MODE}" in
        standard)
            effective_image="${ZEPHYR_INFRA_IMAGE_REF}"
            effective_source="standard"
            ;;
        auto)
            if [[ -n "${ZEPHYR_INFRA_RUNTIME_IMAGE}" ]] && infra_image_exists "${ZEPHYR_INFRA_RUNTIME_IMAGE}"; then
                effective_image="${ZEPHYR_INFRA_RUNTIME_IMAGE}"
                effective_source="derived"
            else
                effective_image="${ZEPHYR_INFRA_IMAGE_REF}"
                effective_source="standard-fallback"
            fi
            ;;
        derived)
            [[ -n "${ZEPHYR_INFRA_RUNTIME_IMAGE}" ]] || infra_err "runtime_image must be set when image_mode=derived"
            if [[ "${allow_missing_runtime}" != "1" ]] && ! infra_image_exists "${ZEPHYR_INFRA_RUNTIME_IMAGE}"; then
                infra_err "runtime_image not found locally (${ZEPHYR_INFRA_RUNTIME_IMAGE}). Build it with repoctl image build."
            fi
            effective_image="${ZEPHYR_INFRA_RUNTIME_IMAGE}"
            effective_source="derived"
            ;;
        *)
            infra_err "Invalid image_mode at runtime: ${ZEPHYR_INFRA_IMAGE_MODE}"
            ;;
    esac

    ZEPHYR_INFRA_EFFECTIVE_IMAGE="${effective_image}"
    ZEPHYR_INFRA_EFFECTIVE_IMAGE_SOURCE="${effective_source}"
    export ZEPHYR_INFRA_EFFECTIVE_IMAGE
    export ZEPHYR_INFRA_EFFECTIVE_IMAGE_SOURCE
}

infra_verify_runtime_label() {
    local image_ref="$1"
    local expected_base_ref="$2"
    local actual_base_ref

    if ! infra_image_exists "${image_ref}"; then
        infra_err "Image not found for label verification: ${image_ref}"
    fi

    actual_base_ref="$(docker inspect --format='{{index .Config.Labels "sygaldry.base_image_ref"}}' "${image_ref}" 2>/dev/null || true)"
    if [[ -z "${actual_base_ref}" || "${actual_base_ref}" == "<no value>" ]]; then
        infra_err "Derived image '${image_ref}' missing label sygaldry.base_image_ref"
    fi
    if [[ "${actual_base_ref}" != "${expected_base_ref}" ]]; then
        infra_err "Derived image base label mismatch: expected '${expected_base_ref}', got '${actual_base_ref}'"
    fi
}

infra_default_repo_root() {
    if command -v git >/dev/null 2>&1; then
        git rev-parse --show-toplevel 2>/dev/null || pwd
    else
        pwd
    fi
}

infra_load() {
    local config_path="$1"

    [[ -f "${config_path}" ]] || infra_err "Config file not found: ${config_path}"

    local image_ref
    image_ref="$(infra_read_value "${config_path}" image_ref || true)"

    local base_image_ref
    base_image_ref="$(infra_read_value "${config_path}" base_image_ref || true)"

    local runtime_image
    runtime_image="$(infra_read_value "${config_path}" runtime_image || true)"

    local image_mode
    image_mode="$(infra_read_value "${config_path}" image_mode || true)"

    local project_id
    project_id="$(infra_read_value "${config_path}" project_id || true)"

    local cache_root
    cache_root="$(infra_read_value "${config_path}" cache_root || true)"

    local lease_mode
    lease_mode="$(infra_read_value "${config_path}" lease_mode || true)"

    local cache_profile
    cache_profile="$(infra_read_value "${config_path}" cache_profile || true)"

    local extra_docker_args
    extra_docker_args="$(infra_read_value "${config_path}" extra_docker_args || true)"

    local entrypoint_default
    entrypoint_default="$(infra_read_value "${config_path}" entrypoint_default || true)"

    local default_project_id
    default_project_id="$(basename "$(infra_default_repo_root)")"

    if [[ -z "${image_mode}" ]]; then
        image_mode="auto"
    fi
    infra_require_mode "${image_mode}"

    if [[ -n "${SYGALDRY_IMAGE:-}" ]]; then
        image_ref="${SYGALDRY_IMAGE}"
    fi
    if [[ -z "${base_image_ref}" ]]; then
        base_image_ref="${image_ref}"
    fi
    if [[ -z "${runtime_image}" ]]; then
        runtime_image="${default_project_id}/zephyr:dev"
    fi

    infra_require_digest_pin "${image_ref}" "image_ref"
    if [[ "${image_mode}" == "auto" || "${image_mode}" == "derived" ]]; then
        infra_require_digest_pin "${base_image_ref}" "base_image_ref"
        [[ -n "${runtime_image}" ]] || infra_err "runtime_image must be set when image_mode=${image_mode}"
    fi

    ZEPHYR_INFRA_IMAGE_REF="${image_ref}"
    ZEPHYR_INFRA_BASE_IMAGE_REF="${base_image_ref}"
    ZEPHYR_INFRA_RUNTIME_IMAGE="${runtime_image}"
    ZEPHYR_INFRA_IMAGE_MODE="${image_mode}"
    ZEPHYR_INFRA_PROJECT_ID="${SYGALDRY_PROJECT_ID:-${project_id:-${default_project_id}}}"
    ZEPHYR_INFRA_CACHE_ROOT="${ZEPHYR_CACHE_ROOT:-${cache_root:-/mnt/data_infra/zephyr_container_infra}}"
    ZEPHYR_INFRA_LEASE_MODE="${ZEPHYR_LEASE_MODE:-${lease_mode:-warn}}"
    ZEPHYR_INFRA_CACHE_PROFILE="${ZEPHYR_CACHE_PROFILE:-${cache_profile:-shared}}"
    ZEPHYR_INFRA_EXTRA_DOCKER_ARGS="${SYGALDRY_EXTRA_DOCKER_ARGS:-${extra_docker_args:-}}"
    ZEPHYR_INFRA_ENTRYPOINT_DEFAULT="${entrypoint_default:-default}"

    export ZEPHYR_INFRA_IMAGE_REF
    export ZEPHYR_INFRA_BASE_IMAGE_REF
    export ZEPHYR_INFRA_RUNTIME_IMAGE
    export ZEPHYR_INFRA_IMAGE_MODE
    export ZEPHYR_INFRA_PROJECT_ID
    export ZEPHYR_INFRA_CACHE_ROOT
    export ZEPHYR_INFRA_LEASE_MODE
    export ZEPHYR_INFRA_CACHE_PROFILE
    export ZEPHYR_INFRA_EXTRA_DOCKER_ARGS
    export ZEPHYR_INFRA_ENTRYPOINT_DEFAULT
}

infra_apply_env() {
    local allow_missing_runtime="${1:-0}"

    infra_select_effective_image "${allow_missing_runtime}"

    export SYGALDRY_IMAGE="${ZEPHYR_INFRA_EFFECTIVE_IMAGE}"
    export SYGALDRY_BUILD_IMAGE="never"

    export SYGALDRY_PROJECT_ID="${ZEPHYR_INFRA_PROJECT_ID}"
    export ZEPHYR_CACHE_ROOT="${ZEPHYR_INFRA_CACHE_ROOT}"
    export ZEPHYR_LEASE_MODE="${ZEPHYR_INFRA_LEASE_MODE}"
    export ZEPHYR_CACHE_PROFILE="${ZEPHYR_INFRA_CACHE_PROFILE}"

    if [[ -n "${ZEPHYR_INFRA_EXTRA_DOCKER_ARGS}" ]]; then
        export SYGALDRY_EXTRA_DOCKER_ARGS="${ZEPHYR_INFRA_EXTRA_DOCKER_ARGS}"
    fi
}

infra_print() {
    infra_select_effective_image 1

    echo "image_ref=${ZEPHYR_INFRA_IMAGE_REF}"
    echo "base_image_ref=${ZEPHYR_INFRA_BASE_IMAGE_REF}"
    echo "runtime_image=${ZEPHYR_INFRA_RUNTIME_IMAGE}"
    echo "image_mode=${ZEPHYR_INFRA_IMAGE_MODE}"
    echo "effective_image=${ZEPHYR_INFRA_EFFECTIVE_IMAGE}"
    echo "effective_image_source=${ZEPHYR_INFRA_EFFECTIVE_IMAGE_SOURCE}"
    echo "project_id=${ZEPHYR_INFRA_PROJECT_ID}"
    echo "cache_root=${ZEPHYR_INFRA_CACHE_ROOT}"
    echo "lease_mode=${ZEPHYR_INFRA_LEASE_MODE}"
    echo "cache_profile=${ZEPHYR_INFRA_CACHE_PROFILE}"
    echo "entrypoint_default=${ZEPHYR_INFRA_ENTRYPOINT_DEFAULT}"
}
