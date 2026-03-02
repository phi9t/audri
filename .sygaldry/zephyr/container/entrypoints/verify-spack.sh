#!/bin/bash
#
# Spack Verification Entrypoint
# =============================
#
# Fast verification that Spack packages are installed and functional.
# Does NOT trigger rebuilds - only checks existing installations.
#
# Exit codes:
#   0 - All checks passed
#   1 - Spack packages not found or verification failed

set -euo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)" \
    || _LIB_DIR="/opt/container_entrypoints/../lib"
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

log() {
    echo "[verify-spack] $*" >&2
}

error() {
    log "ERROR: $1"
    if [[ -n "${2:-}" ]]; then
        log "HINT:  $2"
    fi
    exit 1
}

if ! sygaldry_init_spack; then
    error "Spack setup script not found at /opt/spack_src" \
          "Are you inside the container? Use 'sygaldry' or 'launch_container.sh'."
fi

if ! sygaldry_activate_env; then
    error "Spack environment not found" \
          "Set SYGALDRY_SPACK_ENV to override, or build with spack-build.sh."
fi

log "=== Step 1: Checking Spack view and packages ==="

if [[ -d "/opt/spack_store/view/bin" ]]; then
    log "Spack view: /opt/spack_store/view exists"
else
    error "Spack view not found at /opt/spack_store/view" \
          "Build with spack-build.sh or use a snapshot image."
fi

if /opt/spack_store/view/bin/python3 -c "import torch" 2>/dev/null; then
    _torch_ver="$(/opt/spack_store/view/bin/python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    log "py-torch: found (${_torch_ver})"
else
    error "py-torch not importable from Spack view" \
          "Rebuild with spack-build.sh or check spack.lock for py-torch."
fi

if /opt/spack_store/view/bin/python3 -c "import jax" 2>/dev/null; then
    _jax_ver="$(/opt/spack_store/view/bin/python3 -c 'import jax; print(jax.__version__)' 2>/dev/null)"
    log "py-jax: found (${_jax_ver})"
else
    error "py-jax not importable from Spack view" \
          "Rebuild with spack-build.sh or check spack.lock for py-jax."
fi

if spack find py-torch >/dev/null 2>&1; then
    log "spack find py-torch: OK"
else
    log "NOTE: spack find py-torch unavailable (missing spack.lock); packages verified via view"
fi

log "=== Step 2-4: Python verification (import + tensor ops + NN ops) ==="

python3 - <<'PY'
import sys

print("[verify-spack] Importing torch and jax...")

import torch
import jax
import jax.numpy as jnp

errors = []

# ============================================================================
# Torch verification
# ============================================================================
print("[verify-spack] --- PyTorch verification ---")
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()

    print(f"[verify-spack] torch.cuda.is_available(): True")
    print(f"[verify-spack] Device: {device_name}")
    print(f"[verify-spack] CUDA version: {cuda_version}")
    print(f"[verify-spack] cuDNN version: {cudnn_version}")

    print("[verify-spack] Running tensor ops...")
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (100, 100), f"Expected shape (100, 100), got {c.shape}"
    print("[verify-spack] Tensor matmul: OK")

    print("[verify-spack] Running neural network ops...")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    ).to(device)

    x = torch.randn(32, 100, device=device)
    y = model(x)
    assert y.shape == (32, 10), f"Expected shape (32, 10), got {y.shape}"

    loss = y.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"No gradient for {name}")

    print("[verify-spack] Neural network forward + backward: OK")
    print(f"[verify-spack] torch: PASSED")

except Exception as e:
    errors.append(f"torch: {e}")
    print(f"[verify-spack] torch: FAILED - {e}", file=sys.stderr)

# ============================================================================
# JAX verification
# ============================================================================
print("[verify-spack] --- JAX verification ---")
try:
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]

    if not gpu_devices:
        raise RuntimeError(f"No GPU devices found (available: {devices})")

    print(f"[verify-spack] JAX devices: {devices}")
    print(f"[verify-spack] GPU devices: {gpu_devices}")

    print("[verify-spack] Running tensor ops...")
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (100, 100))
    b = jax.random.normal(key, (100, 100))
    c = jnp.matmul(a, b)
    assert c.shape == (100, 100), f"Expected shape (100, 100), got {c.shape}"
    print("[verify-spack] Tensor matmul: OK")

    print("[verify-spack] Running neural network ops...")

    def forward(params, x):
        W1, b1, W2, b2 = params
        h = jax.nn.relu(x @ W1 + b1)
        return h @ W2 + b2

    def loss_fn(params, x, y):
        pred = forward(params, x)
        return jnp.mean((pred - y) ** 2)

    key1, key2, key3, key4 = jax.random.split(key, 4)
    W1 = jax.random.normal(key1, (100, 64)) * 0.01
    b1 = jnp.zeros(64)
    W2 = jax.random.normal(key2, (64, 10)) * 0.01
    b2 = jnp.zeros(10)
    params = (W1, b1, W2, b2)

    x = jax.random.normal(key3, (32, 100))
    y = jax.random.normal(key4, (32, 10))

    grads = jax.grad(loss_fn)(params, x, y)

    assert grads[0].shape == (100, 64), f"W1 grad shape mismatch"
    assert grads[1].shape == (64,), f"b1 grad shape mismatch"
    assert grads[2].shape == (64, 10), f"W2 grad shape mismatch"
    assert grads[3].shape == (10,), f"b2 grad shape mismatch"

    print("[verify-spack] Neural network forward + gradient: OK")
    print(f"[verify-spack] jax: PASSED")

except Exception as e:
    errors.append(f"jax: {e}")
    print(f"[verify-spack] jax: FAILED - {e}", file=sys.stderr)

# ============================================================================
# Summary
# ============================================================================
print("")
if errors:
    print("[verify-spack] === VERIFICATION FAILED ===", file=sys.stderr)
    for e in errors:
        print(f"[verify-spack] ERROR: {e}", file=sys.stderr)
    sys.exit(1)

print("[verify-spack] === VERIFICATION PASSED ===")
print("[verify-spack] All Spack packages installed and functional.")
PY

log "=== Verification complete ==="
