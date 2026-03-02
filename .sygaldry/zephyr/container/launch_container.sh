#!/bin/bash

# Zephyr container launcher
set -eu -o pipefail

SCRIPT_DIR="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
readonly SCRIPT_DIR
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
readonly PROJECT_ROOT

SYGALDRY_HOME="${SYGALDRY_HOME:-${PROJECT_ROOT}}"
readonly SYGALDRY_HOME

PROJECT_ID="${SYGALDRY_PROJECT_ID:-$(basename "${PWD}")}"
readonly PROJECT_ID

readonly ZEPHYR_CACHE_ROOT="${ZEPHYR_CACHE_ROOT:-/mnt/data_infra/zephyr_container_infra}"
readonly ZEPHYR_SHARED_ROOT="${ZEPHYR_SHARED_ROOT:-${ZEPHYR_CACHE_ROOT}/shared}"
readonly ZEPHYR_BUILD_ROOT="${ZEPHYR_BUILD_ROOT:-${ZEPHYR_CACHE_ROOT}/sygaldry}"
readonly ZEPHYR_PROJECTS_ROOT="${ZEPHYR_PROJECTS_ROOT:-${ZEPHYR_CACHE_ROOT}/projects}"
readonly ZEPHYR_PROJECT_ROOT="${ZEPHYR_PROJECT_ROOT:-${ZEPHYR_PROJECTS_ROOT}/${PROJECT_ID}}"
readonly ZEPHYR_META_ROOT="${ZEPHYR_META_ROOT:-${ZEPHYR_CACHE_ROOT}/meta}"

# Back-compat alias for older scripts/docs.
readonly SYGALDRY_CONTAINER_ROOT="${SYGALDRY_CONTAINER_ROOT:-${ZEPHYR_PROJECT_ROOT}}"

readonly HOST_MONOREPO_HOME="${ZEPHYR_SHARED_MONOREPO_HOME:-${ZEPHYR_PROJECT_ROOT}/home}"
readonly HOST_PROJECT_CONFIG_DIR="${ZEPHYR_SHARED_CONFIG_HOME:-${ZEPHYR_PROJECT_ROOT}/config}"
readonly HOST_PROJECT_LOCAL_SHARE="${ZEPHYR_SHARED_LOCAL_SHARE:-${ZEPHYR_PROJECT_ROOT}/local_share}"
readonly HOST_OUTPUT_ROOT="${ZEPHYR_SHARED_OUTPUT_ROOT:-${ZEPHYR_PROJECT_ROOT}/outputs}"
readonly HOST_WORKSPACE="${ZEPHYR_SHARED_WORKSPACE:-${ZEPHYR_PROJECT_ROOT}/workspace}"
readonly HOST_RUNS_DIR="${ZEPHYR_PROJECT_ROOT}/runs"
readonly HOST_LEASE_DIR="${ZEPHYR_PROJECT_ROOT}/leases"
readonly HOST_LOGS_DIR="${ZEPHYR_PROJECT_ROOT}/logs"

# Shared caches default to shared root; legacy SYGALDRY_* vars still override.
readonly HOST_SPACK_STORE="${ZEPHYR_SHARED_SPACK_STORE:-${SYGALDRY_SPACK_STORE:-${ZEPHYR_CACHE_ROOT}/sygaldry/spack_store}}"
readonly HOST_BAZEL_CACHE="${ZEPHYR_SHARED_BAZEL_CACHE:-${SYGALDRY_BAZEL_CACHE:-${ZEPHYR_SHARED_ROOT}/bazel_cache}}"
readonly HOST_HF_CACHE="${ZEPHYR_SHARED_HF_CACHE:-${SYGALDRY_HF_CACHE:-${ZEPHYR_SHARED_ROOT}/hf_cache}}"
readonly HOST_UV_CACHE="${ZEPHYR_SHARED_UV_CACHE:-${SYGALDRY_UV_CACHE:-${ZEPHYR_SHARED_ROOT}/uv_cache}}"
readonly HOST_TORCH_CACHE="${ZEPHYR_SHARED_TORCH_CACHE:-${ZEPHYR_SHARED_ROOT}/torch_cache}"
readonly HOST_TRITON_CACHE="${ZEPHYR_SHARED_TRITON_CACHE:-${ZEPHYR_SHARED_ROOT}/triton_cache}"
readonly HOST_NV_COMPUTE_CACHE="${ZEPHYR_SHARED_NV_COMPUTE_CACHE:-${ZEPHYR_SHARED_ROOT}/nv_compute_cache}"
readonly HOST_JAX_CACHE="${ZEPHYR_SHARED_JAX_CACHE:-${ZEPHYR_SHARED_ROOT}/jax_cache}"

readonly CONTAINER_HOME="/home/kvothe"
readonly CONTAINER_CONFIG_HOME="${CONTAINER_HOME}/.config"
readonly CONTAINER_LOCAL_SHARE="${CONTAINER_HOME}/.local/share"
readonly CONTAINER_OUTPUT_ROOT="/work""space/outputs"
readonly CONTAINER_SPACK_STORE="/opt/spack_store"
readonly CONTAINER_BAZEL_CACHE="/opt/bazel_cache"
readonly CONTAINER_HF_CACHE="/opt/hf_cache"
readonly CONTAINER_UV_CACHE="/opt/uv_cache"
readonly CONTAINER_TORCH_CACHE="/opt/torch_cache"
readonly CONTAINER_TRITON_CACHE="/opt/triton_cache"
readonly CONTAINER_NV_COMPUTE_CACHE="/opt/nv_compute_cache"
readonly CONTAINER_JAX_CACHE="/opt/jax_cache"
readonly CONTAINER_WORKSPACE="/work""space"
readonly CONTAINER_SYGALDRY="/opt/sygaldry"
readonly CONTAINER_ENTRYPOINT_DIR="/opt/container_entrypoints"

readonly REQUIRED_CUDA_VERSION="${SYGALDRY_REQUIRED_CUDA_VERSION:-12.9}"
readonly CONTAINER_NET="${SYGALDRY_NET:-host}"
readonly CONTAINER_IPC="${SYGALDRY_IPC:-host}"
readonly BUILD_IMAGE_POLICY="${SYGALDRY_BUILD_IMAGE:-auto}"
readonly EXTRA_DOCKER_ARGS="${SYGALDRY_EXTRA_DOCKER_ARGS:-}"
DEFAULT_CACHE_PROFILE="${ZEPHYR_CACHE_PROFILE:-shared}"
readonly DEFAULT_CACHE_PROFILE
readonly LEASE_MODE_DEFAULT="${ZEPHYR_LEASE_MODE:-warn}"

readonly DEFAULT_CONTAINER_IMAGE="sygaldry/zephyr:base"
readonly CONTAINER_IMAGE="${SYGALDRY_IMAGE:-${DEFAULT_CONTAINER_IMAGE}}"
readonly CONTAINER_USER="kvothe"

readonly BAZEL_VERSION="${BAZEL_VERSION:-6.4.0}"
readonly PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
readonly RUST_VERSION="${RUST_VERSION:-1.79.0}"
readonly GO_VERSION="${GO_VERSION:-1.21.5}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [launch:${BASH_LINENO[0]}] $*" >&2
}

error() {
    log "ERROR: $*"
    exit 1
}

version_lt() {
    local a="$1"
    local b="$2"
    local a_major="${a%%.*}"
    local a_minor="${a#*.}"
    local b_major="${b%%.*}"
    local b_minor="${b#*.}"
    if [[ "${a_major}" -lt "${b_major}" ]]; then return 0; fi
    if [[ "${a_major}" -gt "${b_major}" ]]; then return 1; fi
    if [[ "${a_minor:-0}" -lt "${b_minor:-0}" ]]; then return 0; fi
    return 1
}

detect_host_cuda_version() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    local cuda_line
    if command -v rg >/dev/null 2>&1; then
        cuda_line="$(nvidia-smi 2>/dev/null | rg -o "CUDA Version: [0-9]+\\.[0-9]+" -m 1 || true)"
    else
        cuda_line="$(nvidia-smi 2>/dev/null | grep -Eo "CUDA Version: [0-9]+\\.[0-9]+" | head -n 1 || true)"
    fi
    if [[ -z "${cuda_line}" ]]; then
        return 1
    fi
    echo "${cuda_line##*CUDA Version: }"
}

resolve_mount_path() {
    local path="$1"
    local parent_dir
    parent_dir="$(dirname "${path}")"
    if [[ ! -d "${parent_dir}" ]]; then
        mkdir -p "${parent_dir}"
    fi
    if [[ ! -e "${path}" ]]; then
        mkdir -p "${path}"
    fi
    realpath "${path}"
}

check_requirements() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
    fi
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running or not accessible"
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        log "WARNING: NVIDIA runtime not listed in docker info; continuing and relying on docker --gpus validation."
    fi
}

warn_legacy_overrides() {
    if [[ -n "${SYGALDRY_SPACK_STORE:-}" ]]; then
        log "DEPRECATED: SYGALDRY_SPACK_STORE set; prefer ZEPHYR_SHARED_SPACK_STORE"
    fi
    if [[ -n "${SYGALDRY_HF_CACHE:-}" ]]; then
        log "DEPRECATED: SYGALDRY_HF_CACHE set; prefer ZEPHYR_SHARED_HF_CACHE"
    fi
    if [[ -n "${SYGALDRY_UV_CACHE:-}" ]]; then
        log "DEPRECATED: SYGALDRY_UV_CACHE set; prefer ZEPHYR_SHARED_UV_CACHE"
    fi
}

setup_host_directories() {
    local dirs=(
        "${ZEPHYR_SHARED_ROOT}"
        "${ZEPHYR_BUILD_ROOT}"
        "${ZEPHYR_PROJECT_ROOT}"
        "${ZEPHYR_META_ROOT}"
        "${HOST_MONOREPO_HOME}"
        "${HOST_PROJECT_CONFIG_DIR}"
        "${HOST_PROJECT_LOCAL_SHARE}"
        "${HOST_OUTPUT_ROOT}"
        "${HOST_WORKSPACE}"
        "${HOST_RUNS_DIR}"
        "${HOST_LEASE_DIR}"
        "${HOST_LOGS_DIR}"
        "${HOST_SPACK_STORE}"
        "${HOST_BAZEL_CACHE}"
        "${HOST_HF_CACHE}"
        "${HOST_UV_CACHE}"
        "${HOST_TORCH_CACHE}"
        "${HOST_TRITON_CACHE}"
        "${HOST_NV_COMPUTE_CACHE}"
        "${HOST_JAX_CACHE}"
    )
    local dir
    for dir in "${dirs[@]}"; do
        [[ -d "${dir}" ]] || mkdir -p "${dir}"
    done

    cat > "${ZEPHYR_META_ROOT}/layout_version.json" <<'JSON'
{"layout_version":2,"layout_name":"unified-shared-cache-project-isolation"}
JSON
}

resolve_common_mount_paths() {
    RESOLVED_MONOREPO_HOME="$(resolve_mount_path "${HOST_MONOREPO_HOME}")"
    RESOLVED_SPACK_STORE="$(resolve_mount_path "${HOST_SPACK_STORE}")"
    RESOLVED_BAZEL_CACHE="$(resolve_mount_path "${HOST_BAZEL_CACHE}")"
    RESOLVED_HF_CACHE="$(resolve_mount_path "${HOST_HF_CACHE}")"
    RESOLVED_UV_CACHE="$(resolve_mount_path "${HOST_UV_CACHE}")"
    RESOLVED_PROJECT_CONFIG="$(resolve_mount_path "${HOST_PROJECT_CONFIG_DIR}")"
    RESOLVED_PROJECT_LOCAL_SHARE="$(resolve_mount_path "${HOST_PROJECT_LOCAL_SHARE}")"
    RESOLVED_OUTPUT_ROOT="$(resolve_mount_path "${HOST_OUTPUT_ROOT}")"
    RESOLVED_TORCH_CACHE="$(resolve_mount_path "${HOST_TORCH_CACHE}")"
    RESOLVED_TRITON_CACHE="$(resolve_mount_path "${HOST_TRITON_CACHE}")"
    RESOLVED_NV_COMPUTE_CACHE="$(resolve_mount_path "${HOST_NV_COMPUTE_CACHE}")"
    RESOLVED_JAX_CACHE="$(resolve_mount_path "${HOST_JAX_CACHE}")"
}

resolve_entrypoint_container_dir() {
    local spack_baked="$1"
    local entrypoint_name="$2"
    local entrypoints_baked
    entrypoints_baked="$(docker inspect --format='{{index .Config.Labels "sygaldry.entrypoints.baked"}}' "${CONTAINER_IMAGE}" 2>/dev/null || echo "")"

    if [[ "${entrypoints_baked}" == "true" ]]; then
        echo "${CONTAINER_ENTRYPOINT_DIR}"
        return 0
    fi
    if [[ "${spack_baked}" == "true" ]]; then
        echo "/opt/spack_env/entrypoints"
        return 0
    fi
    if [[ ! -f "${SYGALDRY_HOME}/container/entrypoints/${entrypoint_name}.sh" ]]; then
        error "Entrypoint not found: ${SYGALDRY_HOME}/container/entrypoints/${entrypoint_name}.sh"
    fi
    echo ""
}

build_container_image() {
    local build_image=false
    local is_default_image=false
    if [[ "${CONTAINER_IMAGE}" == "${DEFAULT_CONTAINER_IMAGE}" ]]; then
        is_default_image=true
    fi
    if [[ "${BUILD_IMAGE_POLICY}" != "auto" && "${BUILD_IMAGE_POLICY}" != "always" && "${BUILD_IMAGE_POLICY}" != "never" ]]; then
        error "Invalid SYGALDRY_BUILD_IMAGE value: ${BUILD_IMAGE_POLICY} (expected auto|always|never)"
    fi

    if docker image inspect "${CONTAINER_IMAGE}" >/dev/null 2>&1; then
        if [[ "${BUILD_IMAGE_POLICY}" == "always" ]]; then
            build_image=true
        elif [[ "${BUILD_IMAGE_POLICY}" == "auto" && "${is_default_image}" == "true" ]]; then
            local dockerfile_path="${SCRIPT_DIR}/dev_container.dockerfile"
            if [[ -f "${dockerfile_path}" ]]; then
                local dockerfile_mtime
                dockerfile_mtime=$(stat -c %Y "${dockerfile_path}" 2>/dev/null || echo 0)
                local image_created
                image_created=$(docker image inspect "${CONTAINER_IMAGE}" --format='{{.Created}}' 2>/dev/null)
                local image_timestamp
                image_timestamp=$(date -d "${image_created}" +%s 2>/dev/null || echo 0)
                if [[ ${dockerfile_mtime} -gt ${image_timestamp} ]]; then
                    build_image=true
                fi
            fi
        fi
    else
        if [[ "${BUILD_IMAGE_POLICY}" == "never" ]]; then
            error "Docker image ${CONTAINER_IMAGE} not found and SYGALDRY_BUILD_IMAGE=never"
        fi
        if [[ "${BUILD_IMAGE_POLICY}" == "always" || "${is_default_image}" == "true" ]]; then
            build_image=true
        else
            error "Docker image ${CONTAINER_IMAGE} not found. Pull/build it first, or use SYGALDRY_BUILD_IMAGE=always explicitly."
        fi
    fi

    if [[ "${build_image}" == "true" ]]; then
        local dockerfile_path="${SCRIPT_DIR}/dev_container.dockerfile"
        [[ -f "${dockerfile_path}" ]] || error "Dockerfile not found at ${dockerfile_path}"

        local host_uid
        host_uid=$(id -u)
        local host_gid
        host_gid=$(id -g)

        docker build \
            --file "${dockerfile_path}" \
            --tag "${CONTAINER_IMAGE}" \
            --build-arg "BAZEL_VERSION=${BAZEL_VERSION}" \
            --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
            --build-arg "RUST_VERSION=${RUST_VERSION}" \
            --build-arg "GO_VERSION=${GO_VERSION}" \
            --build-arg "HOST_UID=${host_uid}" \
            --build-arg "HOST_GID=${host_gid}" \
            "${PROJECT_ROOT}" || error "Failed to build Docker image"
    fi
}

lease_file_for() {
    local lease_dir="$1"
    local resource="$2"
    echo "${lease_dir}/${resource}.lease"
}

read_lease_expiry() {
    local lease_file="$1"
    if [[ ! -f "${lease_file}" ]]; then
        echo ""
        return
    fi
    awk -F= '$1=="expires_epoch"{print $2}' "${lease_file}" 2>/dev/null || true
}

acquire_lease() {
    local lease_mode="$1"
    local lease_dir="$2"
    local resource="$3"
    local owner="$4"
    local ttl_s="$5"
    local run_id="$6"

    mkdir -p "${lease_dir}"
    local lease_file
    lease_file="$(lease_file_for "${lease_dir}" "${resource}")"
    local now
    now=$(date +%s)

    if [[ -f "${lease_file}" ]]; then
        local expiry
        expiry="$(read_lease_expiry "${lease_file}")"
        if [[ -n "${expiry}" && "${expiry}" -ge "${now}" ]]; then
            local msg="Resource lease exists (${resource}) at ${lease_file}"
            if [[ "${lease_mode}" == "enforce" ]]; then
                error "${msg}"
            fi
            if [[ "${lease_mode}" == "warn" ]]; then
                log "WARNING: ${msg}"
            fi
        fi
    fi

    local expires
    expires=$((now + ttl_s))
    cat > "${lease_file}" <<EOF_LEASE
resource=${resource}
owner=${owner}
run_id=${run_id}
pid=$$
created_epoch=${now}
expires_epoch=${expires}
EOF_LEASE
    echo "${lease_file}"
}

release_lease() {
    local lease_file="$1"
    [[ -n "${lease_file}" && -f "${lease_file}" ]] && rm -f "${lease_file}"
}

# build_docker_args CFG
# Reads a named associative array (passed by name) to build the docker CLI args.
# Required keys: monorepo_home, spack_store, bazel_cache, hf_cache, uv_cache,
#   entrypoint_path, mode, run_id, lease_mode, cache_profile
# Optional keys: sygaldry_home, workspace, repo, repo_name, project_root,
#   spack_baked, project_config, project_local_share, output_root,
#   torch_cache, triton_cache, nv_compute_cache, jax_cache
build_docker_args() {
    local -n _cfg=$1

    local docker_args=()
    docker_args+=("--rm" "--init")
    if [[ -t 0 ]]; then
        docker_args+=("--interactive" "--tty")
    fi

    [[ -n "${CONTAINER_NET}" ]] || error "SYGALDRY_NET must be non-empty"
    [[ -n "${CONTAINER_IPC}" ]] || error "SYGALDRY_IPC must be non-empty"
    docker_args+=("--net=${CONTAINER_NET}" "--ipc=${CONTAINER_IPC}")

    local host_uid
    host_uid=$(id -u)
    local host_gid
    host_gid=$(id -g)
    local user_spec="${host_uid}:${host_gid}"
    if docker info 2>/dev/null | grep -q rootless; then
        user_spec="0:0"
    fi
    docker_args+=("--user=${user_spec}")

    if [[ "${SYGALDRY_MOUNT_HOST_IDENTITY:-0}" == "1" ]]; then
        docker_args+=("--volume=/etc/passwd:/etc/passwd:ro" "--volume=/etc/group:/etc/group:ro")
    fi

    docker_args+=("--volume=${_cfg[monorepo_home]}:${CONTAINER_HOME}")
    if [[ -n "${_cfg[project_config]:-}" ]]; then
        docker_args+=("--volume=${_cfg[project_config]}:${CONTAINER_CONFIG_HOME}")
    fi
    if [[ -n "${_cfg[project_local_share]:-}" ]]; then
        docker_args+=("--volume=${_cfg[project_local_share]}:${CONTAINER_LOCAL_SHARE}")
    fi
    if [[ -n "${_cfg[output_root]:-}" ]]; then
        docker_args+=("--volume=${_cfg[output_root]}:${CONTAINER_OUTPUT_ROOT}")
    fi
    docker_args+=(
        "--volume=${_cfg[bazel_cache]}:${CONTAINER_BAZEL_CACHE}"
        "--volume=${_cfg[hf_cache]}:${CONTAINER_HF_CACHE}"
        "--volume=${_cfg[uv_cache]}:${CONTAINER_UV_CACHE}"
    )

    if [[ -n "${_cfg[torch_cache]:-}" ]]; then
        docker_args+=("--volume=${_cfg[torch_cache]}:${CONTAINER_TORCH_CACHE}")
    fi
    if [[ -n "${_cfg[triton_cache]:-}" ]]; then
        docker_args+=("--volume=${_cfg[triton_cache]}:${CONTAINER_TRITON_CACHE}")
    fi
    if [[ -n "${_cfg[nv_compute_cache]:-}" ]]; then
        docker_args+=("--volume=${_cfg[nv_compute_cache]}:${CONTAINER_NV_COMPUTE_CACHE}")
    fi
    if [[ -n "${_cfg[jax_cache]:-}" ]]; then
        docker_args+=("--volume=${_cfg[jax_cache]}:${CONTAINER_JAX_CACHE}")
    fi

    if [[ "${_cfg[spack_baked]:-false}" != "true" ]]; then
        docker_args+=("--volume=${_cfg[spack_store]}:${CONTAINER_SPACK_STORE}")
    fi

    # Compatibility overlay for baked images whose entrypoints expect ../lib helpers.
    if [[ -d "${SYGALDRY_HOME}/container/entrypoints" ]]; then
        docker_args+=(
            "--volume=$(resolve_mount_path "${SYGALDRY_HOME}/container/entrypoints"):/opt/container_entrypoints:ro"
            "--volume=$(resolve_mount_path "${SYGALDRY_HOME}/container/entrypoints"):/opt/spack_env/entrypoints:ro"
        )
    fi
    if [[ -d "${SYGALDRY_HOME}/container/lib" ]]; then
        docker_args+=(
            "--volume=$(resolve_mount_path "${SYGALDRY_HOME}/container/lib"):/opt/lib:ro"
            "--volume=$(resolve_mount_path "${SYGALDRY_HOME}/container/lib"):/opt/spack_env/lib:ro"
        )
    fi

    local sygaldry_root_in_container
    if [[ "${_cfg[mode]}" == "multi-repo" ]]; then
        if [[ -n "${_cfg[sygaldry_home]:-}" ]]; then
            docker_args+=("--volume=${_cfg[sygaldry_home]}:${CONTAINER_SYGALDRY}:ro")
        fi
        docker_args+=(
            "--volume=${_cfg[workspace]}:${CONTAINER_WORKSPACE}"
            "--volume=${_cfg[repo]}:${CONTAINER_WORKSPACE}/${_cfg[repo_name]}"
            "--workdir=${CONTAINER_WORKSPACE}/${_cfg[repo_name]}"
        )
        sygaldry_root_in_container="${CONTAINER_WORKSPACE}/${_cfg[repo_name]}"
    else
        docker_args+=("--volume=${_cfg[project_root]}:${CONTAINER_WORKSPACE}" "--workdir=${CONTAINER_WORKSPACE}")
        sygaldry_root_in_container="${CONTAINER_WORKSPACE}"
    fi

    docker_args+=("--entrypoint=${_cfg[entrypoint_path]}")

    local host_cuda_version
    host_cuda_version="$(detect_host_cuda_version || true)"
    log "Host CUDA version: ${host_cuda_version}"
    log "Required CUDA version: ${REQUIRED_CUDA_VERSION}"

    if ! docker info 2>/dev/null | grep -q nvidia; then
        error "NVIDIA Docker runtime not detected. This is a GPU-only container infrastructure."
    fi
    if [[ -n "${host_cuda_version}" ]] && version_lt "${host_cuda_version}" "${REQUIRED_CUDA_VERSION}"; then
        error "Host CUDA ${host_cuda_version} < required ${REQUIRED_CUDA_VERSION}"
    fi

    docker_args+=("--runtime=nvidia" "--gpus=all")

    docker_args+=(
        "--env=SYGALDRY_IN_CONTAINER=1"
        "--env=SYGALDRY_ROOT=${sygaldry_root_in_container}"
        "--env=SYGALDRY_PROJECT_ID=${PROJECT_ID}"
        "--env=SYGALDRY_RUN_ID=${_cfg[run_id]}"
        "--env=ZEPHYR_LEASE_MODE=${_cfg[lease_mode]}"
        "--env=ZEPHYR_CACHE_PROFILE=${_cfg[cache_profile]}"
        "--env=USER=${CONTAINER_USER}"
        "--env=HOME=${CONTAINER_HOME}"
        "--env=XDG_CONFIG_HOME=${CONTAINER_CONFIG_HOME}"
        "--env=XDG_DATA_HOME=${CONTAINER_LOCAL_SHARE}"
        "--env=XDG_CACHE_HOME=${CONTAINER_UV_CACHE}"
        "--env=HF_HOME=${CONTAINER_HF_CACHE}"
        "--env=UV_CACHE_DIR=${CONTAINER_UV_CACHE}"
        "--env=TORCH_HOME=${CONTAINER_TORCH_CACHE}"
        "--env=TRITON_CACHE_DIR=${CONTAINER_TRITON_CACHE}"
        "--env=CUDA_CACHE_PATH=${CONTAINER_NV_COMPUTE_CACHE}"
        "--env=JAX_COMPILATION_CACHE_DIR=${CONTAINER_JAX_CACHE}"
    )

    local passthru_env_vars=(
        "TERM" "LANG" "LC_ALL" "BAZEL_VERSION"
        "SYGALDRY_BUILD_ROLE" "SYGALDRY_SPACK_ENV"
        "SYGALDRY_MLSYS_ENV" "SYGALDRY_MLSYS_VENV_ROOT" "SYGALDRY_MLSYS_TARGET"
    )
    local var
    for var in "${passthru_env_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            docker_args+=("--env=${var}=${!var}")
        fi
    done

    if [[ -n "${EXTRA_DOCKER_ARGS}" ]]; then
        local extra_args=()
        read -r -a extra_args <<<"${EXTRA_DOCKER_ARGS}"
        docker_args+=("${extra_args[@]}")
    fi

    printf '%s\n' "${docker_args[@]}"
}

print_effective_config() {
    local mode="$1"
    local run_id="$2"
    local lease_mode="$3"
    local cache_profile="$4"
    log "Effective config:"
    log "  Mode: ${mode}"
    log "  Project ID: ${PROJECT_ID}"
    log "  Run ID: ${run_id}"
    log "  Lease mode: ${lease_mode}"
    log "  Cache profile: ${cache_profile}"
    log "  Cache root: ${ZEPHYR_CACHE_ROOT}"
    log "  Shared root: ${ZEPHYR_SHARED_ROOT}"
    log "  Build root: ${ZEPHYR_BUILD_ROOT}"
    log "  Project root: ${ZEPHYR_PROJECT_ROOT}"
    log "  Shared caches: hf=${HOST_HF_CACHE} uv=${HOST_UV_CACHE} bazel=${HOST_BAZEL_CACHE}"
    log "  Spack store: host=${HOST_SPACK_STORE} -> container=${CONTAINER_SPACK_STORE}"
    log "  Extra caches: torch=${HOST_TORCH_CACHE} triton=${HOST_TRITON_CACHE} nv_compute=${HOST_NV_COMPUTE_CACHE} jax=${HOST_JAX_CACHE}"
}

main() {
    log "Starting Sygaldry container launcher..."

    local entrypoint_name="${SYGALDRY_ENTRYPOINT:-default}"
    local repo_path="${SYGALDRY_REPO:-}"
    local passthrough_args=()
    local run_id="${SYGALDRY_RUN_ID:-}"
    local lease_mode="${LEASE_MODE_DEFAULT}"
    local cache_profile="${DEFAULT_CACHE_PROFILE}"
    local print_config=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo=*) repo_path="${1#*=}"; shift ;;
            --repo) repo_path="${2:-}"; [[ -n "${repo_path}" ]] || error "Missing value for --repo"; shift 2 ;;
            --entrypoint=*) entrypoint_name="${1#*=}"; shift ;;
            --entrypoint|-e) entrypoint_name="${2:-}"; [[ -n "${entrypoint_name}" ]] || error "Missing value for --entrypoint"; shift 2 ;;
            --run-id=*) run_id="${1#*=}"; shift ;;
            --run-id) run_id="${2:-}"; [[ -n "${run_id}" ]] || error "Missing value for --run-id"; shift 2 ;;
            --lease-mode=*) lease_mode="${1#*=}"; shift ;;
            --lease-mode) lease_mode="${2:-}"; [[ -n "${lease_mode}" ]] || error "Missing value for --lease-mode"; shift 2 ;;
            --cache-profile=*) cache_profile="${1#*=}"; shift ;;
            --cache-profile) cache_profile="${2:-}"; [[ -n "${cache_profile}" ]] || error "Missing value for --cache-profile"; shift 2 ;;
            --print-effective-config) print_config=1; shift ;;
            --) shift; passthrough_args+=("$@"); break ;;
            *) passthrough_args+=("$1"); shift ;;
        esac
    done

    entrypoint_name="${entrypoint_name%.sh}"
    if [[ -z "${run_id}" ]]; then
        run_id="run-$(date +%Y%m%d-%H%M%S)-$$"
    fi

    if [[ "${lease_mode}" != "off" && "${lease_mode}" != "warn" && "${lease_mode}" != "enforce" ]]; then
        error "Invalid --lease-mode: ${lease_mode} (expected off|warn|enforce)"
    fi
    if [[ "${cache_profile}" != "shared" && "${cache_profile}" != "isolated" && "${cache_profile}" != "hybrid" ]]; then
        error "Invalid --cache-profile: ${cache_profile} (expected shared|isolated|hybrid)"
    fi

    local mode="legacy"
    if [[ -n "${repo_path}" ]]; then
        mode="multi-repo"
        repo_path="$(realpath "${repo_path}")"
        [[ -d "${repo_path}" ]] || error "Repo path does not exist: ${repo_path}"
    fi

    warn_legacy_overrides
    if [[ ${print_config} -eq 1 ]]; then
        print_effective_config "${mode}" "${run_id}" "${lease_mode}" "${cache_profile}"
    fi

    check_requirements
    setup_host_directories
    build_container_image

    local spack_baked
    spack_baked="$(docker inspect --format='{{index .Config.Labels "sygaldry.spack.baked"}}' "${CONTAINER_IMAGE}" 2>/dev/null || echo "")"

    resolve_common_mount_paths
    local entrypoint_container_dir
    entrypoint_container_dir="$(resolve_entrypoint_container_dir "${spack_baked}" "${entrypoint_name}")"

    local lease_file=""
    if [[ "${lease_mode}" != "off" ]]; then
        lease_file="$(acquire_lease "${lease_mode}" "${HOST_LEASE_DIR}" "gpu-all" "${PROJECT_ID}" 21600 "${run_id}")"
    fi

    # Build config struct for docker args
    declare -A CFG=(
        [monorepo_home]="${RESOLVED_MONOREPO_HOME}"
        [spack_store]="${RESOLVED_SPACK_STORE}"
        [bazel_cache]="${RESOLVED_BAZEL_CACHE}"
        [hf_cache]="${RESOLVED_HF_CACHE}"
        [uv_cache]="${RESOLVED_UV_CACHE}"
        [spack_baked]="${spack_baked}"
        [project_config]="${RESOLVED_PROJECT_CONFIG}"
        [project_local_share]="${RESOLVED_PROJECT_LOCAL_SHARE}"
        [output_root]="${RESOLVED_OUTPUT_ROOT}"
        [torch_cache]="${RESOLVED_TORCH_CACHE}"
        [triton_cache]="${RESOLVED_TRITON_CACHE}"
        [nv_compute_cache]="${RESOLVED_NV_COMPUTE_CACHE}"
        [jax_cache]="${RESOLVED_JAX_CACHE}"
        [run_id]="${run_id}"
        [lease_mode]="${lease_mode}"
        [cache_profile]="${cache_profile}"
        [mode]="${mode}"
    )

    local docker_args
    if [[ "${mode}" == "multi-repo" ]]; then
        local repo_name
        repo_name="$(basename "${repo_path}")"
        if [[ -n "${entrypoint_container_dir}" ]]; then
            CFG[entrypoint_path]="${entrypoint_container_dir}/${entrypoint_name}.sh"
        else
            CFG[entrypoint_path]="${CONTAINER_SYGALDRY}/container/entrypoints/${entrypoint_name}.sh"
        fi

        if [[ -z "${entrypoint_container_dir}" ]]; then
            CFG[sygaldry_home]="$(resolve_mount_path "${SYGALDRY_HOME}")"
        fi
        CFG[workspace]="$(resolve_mount_path "${HOST_WORKSPACE}")"
        CFG[repo]="$(realpath "${repo_path}")"
        CFG[repo_name]="${repo_name}"

        readarray -t docker_args < <(build_docker_args CFG)
    else
        if [[ -n "${entrypoint_container_dir}" ]]; then
            CFG[entrypoint_path]="${entrypoint_container_dir}/${entrypoint_name}.sh"
        else
            CFG[entrypoint_path]="${CONTAINER_WORKSPACE}/container/entrypoints/${entrypoint_name}.sh"
        fi
        CFG[project_root]="$(resolve_mount_path "${SYGALDRY_WORKSPACE_SOURCE:-${PWD}}")"

        readarray -t docker_args < <(build_docker_args CFG)
    fi

    set +e
    docker run "${docker_args[@]}" "${CONTAINER_IMAGE}" "${passthrough_args[@]}"
    local rc=$?
    set -e

    release_lease "${lease_file}"
    exit ${rc}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
