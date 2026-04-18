#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .venv/bin/activate ]; then
    echo "Error: Missing .venv. Run bash deploy/podman/install_trellis_rocm.sh first." >&2
    exit 1
fi

. .venv/bin/activate

export TARGET_GFX="${TARGET_GFX:-gfx1100}"
export GPU_ARCHS="${GPU_ARCHS:-$TARGET_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$TARGET_GFX}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

python - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "No HIP GPUs are available inside the container. "
        "Run bash deploy/podman/verify_rocm.sh for device diagnostics."
    )
PY

exec python app.py "$@"
