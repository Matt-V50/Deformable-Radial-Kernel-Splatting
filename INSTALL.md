#!/usr/bin/env fish
# ============================================================
# Deformable Radial Kernel Splatting 环境配置脚本 (Fish Shell)
# 基于已有的3DGS环境配置,保持 Python 3.10.12, PyTorch 2.4.1, CUDA 12.1
# ============================================================

# 环境名称
set ENV_NAME "drk_gs"

echo "=========================================="
echo "开始配置 Deformable Radial Kernel Splatting 环境"
echo "=========================================="

# 创建conda环境
echo "创建 conda 环境: $ENV_NAME (Python 3.10.12)..."
conda create -y -n $ENV_NAME python=3.10.12
conda activate $ENV_NAME

# ============================================================
# PyTorch + CUDA 12.1 (保持与你的设置一致)
# ============================================================
echo "=========================================="
echo "安装 PyTorch 2.4.1 + CUDA 12.1..."
echo "=========================================="
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install cuda-toolkit -c nvidia/label/cuda-12.1.0 --yes

# 解决 undefined symbol: iJIT_NotifyEvent
conda install mkl==2023.1.0 mkl-include -c conda-forge --yes

# 解决 cannot find -lcudart
conda install cuda-cudart=12.1.55 -c nvidia/label/cuda-12.1.0 --yes

# ============================================================
# Deformable Radial Kernel Splatting 核心依赖
# ============================================================
echo "=========================================="
echo "安装 Deformable Radial Kernel Splatting 核心依赖..."
echo "=========================================="

# 基础科学计算库
pip install numpy \
    tqdm \
    plyfile \
    Pillow==11.0.0 \
    opencv-python==4.10.0.84 \
    scipy \
    lpips==0.1.4 \
    tqdm \
    mediapy \
    numpy==1.26.4 \
    matplotlib==3.5.3 \
    einops \
    imageio==2.27.0 \
    torchmetrics 

# Submodule dependencies
pip install --no-build-isolation \
    submodules/depth-diff-gaussian-rasterization \
    submodules/drk_splatting \
    submodules/simple-knn

