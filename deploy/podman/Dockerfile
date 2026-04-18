FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    cmake \
    curl \
    espeak \
    ffmpeg \
    g++ \
    git \
    git-lfs \
    gnupg2 \
    libdw1 \
    libjpeg-dev \
    libopenmpi3 \
    libsparsehash-dev \
    libvulkan-dev \
    libxml2 \
    make \
    nano \
    nodejs \
    npm \
    pipx \
    pkg-config \
    python3-dev \
    python3-setuptools \
    python3-tk \
    python3-venv \
    python3-wheel \
    sudo \
    tar \
    vulkan-tools \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/31.20/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.2 noble main" > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' > /etc/apt/preferences.d/rocm-pin

RUN apt-get update && apt-get install -y \
    amd-smi-lib \
    hipblas \
    hipcc \
    hipcub \
    hipcub-dev \
    hipfft \
    hipfft-dev \
    hipify-clang \
    hiprand \
    hiprand-dev \
    hipsolver \
    hipsolver-dev \
    hipsparse \
    hipsparse-dev \
    hipsparselt \
    hipsparselt-dev \
    hsa-amd-aqlprofile \
    miopen-hip \
    miopen-hip-dev \
    rocfft \
    rocfft-dev \
    rocminfo \
    rocm \
    rocm-cmake \
    rocm-hip-runtime \
    rocm-hip-runtime-dev \
    rocm-hip-sdk \
    rocm-smi \
    rocm-smi-lib \
    rocprim \
    rocprim-dev \
    rocprofiler-sdk \
    rocrand \
    rocrand-dev \
    rocthrust \
    rocthrust-dev && \
    rm -rf /var/lib/apt/lists/*

RUN getent group render || groupadd -r render
RUN mkdir -p /AI/TRELLIS.2_rocm && chmod 777 /AI /AI/TRELLIS.2_rocm

COPY deploy/podman/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh && \
    PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install uv

ENV PATH="/usr/local/bin:/opt/rocm/bin:/opt/rocm/opencl/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}"
ENV ROCM_PATH="/opt/rocm"
ENV ROCM_HOME="/opt/rocm"
ENV TARGET_GFX="gfx1100"
ENV GPU_ARCHS="gfx1100"
ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
ENV ATTN_BACKEND="sdpa"

WORKDIR /AI/TRELLIS.2_rocm

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
