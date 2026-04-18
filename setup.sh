#!/usr/bin/env bash
set -euo pipefail

script_exit() {
    local code="${1:-0}"
    return "${code}" 2>/dev/null || exit "${code}"
}

TEMP=$(getopt -o h --long help,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@")
eval set -- "$TEMP"

HELP=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false

if [ "$#" -eq 1 ]; then
    HELP=true
fi

while true; do
    case "$1" in
        -h|--help) HELP=true; shift ;;
        --basic) BASIC=true; shift ;;
        --flash-attn) FLASHATTN=true; shift ;;
        --cumesh) CUMESH=true; shift ;;
        --o-voxel) OVOXEL=true; shift ;;
        --flexgemm) FLEXGEMM=true; shift ;;
        --nvdiffrast) NVDIFFRAST=true; shift ;;
        --nvdiffrec) NVDIFFREC=true; shift ;;
        --) shift; break ;;
        *) ERROR=true; break ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPENDENCY_DIR="${ROOT_DIR}/Dependency"
STAGE_ROOT="${TMPDIR:-/tmp}/trellis2-stage"
TARGET_GFX="${TARGET_GFX:-gfx1100}"
GPU_ARCHS="${GPU_ARCHS:-$TARGET_GFX}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$TARGET_GFX}"
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

export TARGET_GFX
export GPU_ARCHS
export PYTORCH_ROCM_ARCH
export HSA_OVERRIDE_GFX_VERSION
export ROCM_PATH

show_help() {
    cat <<EOF
Usage: ./setup.sh [OPTIONS]

Options:
  -h, --help              Display this help message
  --basic                 Install Python/runtime dependencies
  --flash-attn            Install flash-attention from remote source
  --cumesh                Install CuMesh from ./Dependency/CuMesh
  --o-voxel               Install o-voxel from ./Dependency/o-voxel
  --flexgemm              Install FlexGEMM from ./Dependency/FlexGEMM-rocm
  --nvdiffrast            Install nvdiffrast (CUDA remote / ROCm local nvdiffrast-hip)
  --nvdiffrec             Install nvdiffrec from ./Dependency/nvdiffrec

This repo is transport-friendly: anything stored in ./Dependency is treated as
the local source of truth and is installed without git clone or submodule update.
Only flash-attention, utils3d, Python wheels, and model weights are fetched remotely.

Recommended ROCm bootstrap:
  bash deploy/podman/install_trellis_rocm.sh

Manual CUDA bootstrap:
  python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

ROCm defaults:
  TARGET_GFX=${TARGET_GFX}
  GPU_ARCHS=${GPU_ARCHS}
  PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
  HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}
  ROCM_PATH=${ROCM_PATH}
EOF
}

if [ "$ERROR" = true ]; then
    echo "Error: Invalid argument" >&2
    HELP=true
fi

if [ "$HELP" = true ]; then
    show_help
    script_exit 0
fi

if [ ! -d "$DEPENDENCY_DIR" ]; then
    echo "Error: Missing Dependency directory at $DEPENDENCY_DIR" >&2
    script_exit 1
fi

copy_tree() {
    local src="$1"
    local dst="$2"
    mkdir -p "$dst"
    cp -a "${src}/." "$dst/"
}

strip_stage_metadata() {
    local path="$1"
    find "$path" -type d \( -name .git -o -name __pycache__ \) -prune -exec rm -rf {} +
}

stage_dependency() {
    local name="$1"
    local src="${DEPENDENCY_DIR}/${name}"
    local dst="${STAGE_ROOT}/${name}"

    if [ ! -d "$src" ]; then
        echo "Error: Missing local dependency ${name} at ${src}" >&2
        script_exit 1
    fi

    rm -rf "$dst"
    mkdir -p "$dst"
    copy_tree "$src" "$dst"
    strip_stage_metadata "$dst"
    printf '%s\n' "$dst"
}

overlay_dependency() {
    local name="$1"
    local dst="$2"
    local src="${DEPENDENCY_DIR}/${name}"

    if [ ! -d "$src" ]; then
        echo "Error: Missing local dependency ${name} at ${src}" >&2
        script_exit 1
    fi

    rm -rf "$dst"
    mkdir -p "$dst"
    copy_tree "$src" "$dst"
    strip_stage_metadata "$dst"
}

install_system_package_if_available() {
    local package="$1"
    if ! command -v apt-get >/dev/null 2>&1; then
        return
    fi
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "$package"
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "$package"
    else
        echo "[SETUP] Skipping ${package}: apt-get requires root or sudo." >&2
    fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
    PLATFORM="cuda"
elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU toolchain found (expected nvidia-smi or rocminfo)." >&2
    script_exit 1
fi

echo "[SETUP] repo=${ROOT_DIR}"
echo "[SETUP] dependency_dir=${DEPENDENCY_DIR}"
echo "[SETUP] platform=${PLATFORM}"
echo "[SETUP] TARGET_GFX=${TARGET_GFX} GPU_ARCHS=${GPU_ARCHS} PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} ROCM_PATH=${ROCM_PATH}"

if [ "$BASIC" = true ]; then
    python -m pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "transformers==4.56.0" gradio==6.0.1 tensorboard pandas lpips zstandard pyfqmr matplotlib
    python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    install_system_package_if_available libjpeg-dev
    python -m pip install pillow-simd
    python -m pip install kornia timm
fi

if [ "$FLASHATTN" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        python -m pip install flash-attn==2.7.3
    elif [ "$PLATFORM" = "hip" ]; then
        FLASH_STAGE="/tmp/extensions/flash-attention"
        echo "[FLASHATTN] Building flash-attention from remote ROCm source for ${GPU_ARCHS}..."
        rm -rf "$FLASH_STAGE"
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git "$FLASH_STAGE"
        (
            cd "$FLASH_STAGE"
            git checkout tags/v2.7.3-cktile
            GPU_ARCHS="$GPU_ARCHS" python setup.py install
        )
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM" >&2
        script_exit 1
    fi
fi

if [ "$NVDIFFRAST" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        CUDA_STAGE="/tmp/extensions/nvdiffrast"
        rm -rf "$CUDA_STAGE"
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$CUDA_STAGE"
        python -m pip install "$CUDA_STAGE" --no-build-isolation
    elif [ "$PLATFORM" = "hip" ]; then
        HIP_STAGE="$(stage_dependency nvdiffrast-hip)"
        env ROCM_PATH="$ROCM_PATH" PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" python -m pip install "$HIP_STAGE" --no-build-isolation
    fi
fi

if [ "$NVDIFFREC" = true ]; then
    NVDIFFREC_STAGE="$(stage_dependency nvdiffrec)"
    env ROCM_PATH="$ROCM_PATH" PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" python -m pip install "$NVDIFFREC_STAGE" --no-build-isolation
fi

if [ "$CUMESH" = true ]; then
    CUMESH_STAGE="$(stage_dependency CuMesh)"
    overlay_dependency cubvh "${CUMESH_STAGE}/third_party/cubvh"
    overlay_dependency eigen "${CUMESH_STAGE}/third_party/cubvh/third_party/eigen"
    if [ "$PLATFORM" = "hip" ]; then
        env BUILD_TARGET=rocm GPU_ARCHS="$GPU_ARCHS" ROCM_PATH="$ROCM_PATH" PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" python -m pip install "$CUMESH_STAGE" --no-build-isolation
    else
        env BUILD_TARGET=cuda python -m pip install "$CUMESH_STAGE" --no-build-isolation
    fi
fi

if [ "$FLEXGEMM" = true ]; then
    FLEXGEMM_STAGE="$(stage_dependency FlexGEMM-rocm)"
    if [ "$PLATFORM" = "hip" ]; then
        env BUILD_TARGET=rocm GPU_ARCHS="$GPU_ARCHS" ROCM_PATH="$ROCM_PATH" PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" python -m pip install "$FLEXGEMM_STAGE" --no-build-isolation
    else
        env BUILD_TARGET=cuda python -m pip install "$FLEXGEMM_STAGE" --no-build-isolation
    fi
fi

if [ "$OVOXEL" = true ]; then
    OVOXEL_STAGE="$(stage_dependency o-voxel)"
    overlay_dependency eigen "${OVOXEL_STAGE}/third_party/eigen"
    if [ "$PLATFORM" = "hip" ]; then
        env BUILD_TARGET=rocm GPU_ARCHS="$GPU_ARCHS" ROCM_PATH="$ROCM_PATH" PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH" python -m pip install "$OVOXEL_STAGE" --no-build-isolation
    else
        env BUILD_TARGET=cuda python -m pip install "$OVOXEL_STAGE" --no-build-isolation
    fi
fi
