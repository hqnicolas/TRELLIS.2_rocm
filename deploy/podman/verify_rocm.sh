#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REQUIRE_INFO=false
if [ "${1:-}" = "--require-info" ]; then
    REQUIRE_INFO=true
fi

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
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

echo "=== identity ==="
id || true
groups || true
echo

echo "=== device nodes ==="
ls -ld /dev/dri || true
ls -l /dev/kfd || true
ls -l /dev/dri/render* 2>/dev/null || true
echo

echo "=== rocminfo ==="
rocminfo | grep -E 'gfx|Name:' | head -n 20 || true
echo

echo "=== torch preflight ==="
python - <<'PY'
import os
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("TARGET_GFX", os.environ.get("TARGET_GFX"))
print("GPU_ARCHS", os.environ.get("GPU_ARCHS"))
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

echo
echo "=== extension imports ==="
python - <<'PY'
import cumesh
import flex_gemm
import o_voxel
import nvdiffrast.torch as dr
import nvdiffrec_render.renderutils as renderutils

print("imports", "ok")
PY

if curl -sf "http://127.0.0.1:${GRADIO_SERVER_PORT:-7860}/info" >/dev/null; then
    echo "Gradio /info endpoint is reachable."
elif [ "$REQUIRE_INFO" = true ]; then
    echo "Error: Gradio /info endpoint is not reachable." >&2
    exit 1
else
    echo "Gradio /info endpoint not reachable yet. Start the app with bash deploy/podman/run_trellis.sh"
fi
