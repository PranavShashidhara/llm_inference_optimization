# Jetson Orin — CUDA + Python LLM inference dev container
# Base: NVIDIA L4T base image (JetPack 6.x / CUDA 12.6, ARM64)
FROM nvcr.io/nvidia/l4t-base:r36.2.0

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    cmake \
    ninja-build \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    libopenblas-dev \
    openssh-client \
    procps \
    lsb-release \
    libglib2.0-0 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA paths ────────────────────────────────────────────────────────────────
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/aarch64-linux-gnu:/opt/cusparselt/lib
ENV CUDA_HOME=/usr/local/cuda

# ── cuSPARSELt — required by torch nv24.06+ ───────────────────────────────────
# /usr/local/cuda doesn't exist at build time (it's a runtime mount).
# Install into /opt/cusparselt instead, and add to LD_LIBRARY_PATH above.
RUN CUSPARSELT_VER="0.7.1.0" && \
    CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-${CUSPARSELT_VER}-archive" && \
    wget -q "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUSPARSELT_NAME}.tar.xz" && \
    tar xf "${CUSPARSELT_NAME}.tar.xz" && \
    mkdir -p /opt/cusparselt/include /opt/cusparselt/lib && \
    cp -a "${CUSPARSELT_NAME}/include/"* /opt/cusparselt/include/ && \
    cp -a "${CUSPARSELT_NAME}/lib/"* /opt/cusparselt/lib/ && \
    ldconfig && \
    rm -rf "${CUSPARSELT_NAME}" "${CUSPARSELT_NAME}.tar.xz"

# ── Stage 1: base pip tooling ─────────────────────────────────────────────────
RUN pip3 install --upgrade pip setuptools wheel

# ── Stage 2: PyTorch — JetPack 6.1 / CUDA 12.6 / cuDNN 9 ────────────────────
RUN pip3 install --no-cache-dir \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

# ── Stage 3: project dependencies ─────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .

RUN grep -vE "^(torch|torchvision)" requirements.txt \
    | pip3 install --no-cache-dir -r /dev/stdin

# ── Stage 4: Triton ───────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/e45/8f7f91a8e7396/triton-3.3.0-cp310-cp310-linux_aarch64.whl" \
    || echo "Triton wheel unavailable — skipping."

# ── Runtime environment ────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TRITON_CACHE_DIR=/workspace/.triton_cache

RUN mkdir -p /workspace/profiling /workspace/.triton_cache /root/.cache/huggingface

CMD ["/bin/bash"]
