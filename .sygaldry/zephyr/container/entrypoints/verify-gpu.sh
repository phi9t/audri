#!/bin/bash
set -euo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)" \
    || _LIB_DIR="/opt/container_entrypoints/../lib"
# shellcheck disable=SC1091
source "${_LIB_DIR}/spack_init.sh"

echo "=== GPU Verification ==="
echo ""

# CUDA driver
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# CUDA toolkit
echo "CUDA Toolkit:"
nvcc --version 2>/dev/null | grep "release" || echo "nvcc not available"
echo ""

# Initialize Spack + CUDA + view fallback
sygaldry_init_spack || true
if ! sygaldry_activate_env; then
    if [[ -d "/opt/spack_store/view" ]]; then
        echo "WARNING: Spack env activation failed; using view fallback" >&2
        sygaldry_ensure_view_fallback || true
    fi
fi
sygaldry_setup_cuda

# PyTorch
echo "PyTorch CUDA:"
python3 -c "
import torch
print(f'  Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  cuDNN Version: {torch.backends.cudnn.version()}')
" 2>/dev/null || echo "  PyTorch not available"
echo ""

# JAX
echo "JAX GPU:"
python3 -c "
import jax
devices = jax.devices()
print(f'  Devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
print(f'  GPU count: {len(gpu_devices)}')
" 2>/dev/null || echo "  JAX not available"
echo ""

echo "=== Verification Complete ==="
