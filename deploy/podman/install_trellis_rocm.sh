#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export TARGET_GFX="${TARGET_GFX:-gfx1100}"
export GPU_ARCHS="${GPU_ARCHS:-$TARGET_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$TARGET_GFX}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required inside the container." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    uv venv .venv --python /usr/bin/python3 --seed
fi

. .venv/bin/activate

mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    --find-links "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/" \
    -r deploy/requirements/rocm-7.2.2.txt

bash ./setup.sh --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec

echo "Install complete. Run: bash deploy/podman/verify_rocm.sh"
