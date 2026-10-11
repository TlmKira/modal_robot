#!/usr/bin/env bash
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Keep the login credential path when moving large caches into the project.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-/root/.cache/huggingface}/token}"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
export XDG_RUNTIME_DIR="$PROJECT_ROOT/cache/runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MAX_JOBS=3
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
