#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "=== podman container gpu diagnostics ==="
echo "pwd: $(pwd)"
echo

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

echo "=== huggingface env ==="
printf 'HF_HOME=%s\n' "${HF_HOME:-}"
printf 'HF_HUB_CACHE=%s\n' "${HF_HUB_CACHE:-}"
printf 'HF_TOKEN=%s\n' "${HF_TOKEN:+set}"
printf 'HUGGINGFACE_HUB_TOKEN=%s\n' "${HUGGINGFACE_HUB_TOKEN:+set}"
printf 'TRELLIS_IMAGE_COND_MODEL=%s\n' "${TRELLIS_IMAGE_COND_MODEL:-}"
echo

if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
    echo "=== torch preflight ==="
    python - <<'PY'
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
else
    echo ".venv is missing; run bash deploy/podman/install_trellis_rocm.sh first."
fi
