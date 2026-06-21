#!/bin/bash
# =============================================================
# setup.sh - 环境安装脚本
# 在集群终端执行: bash setup.sh
# =============================================================
set -e

echo "==================================="
echo "  C-视觉工程师 环境安装"
echo "==================================="

# ---------- 检测 CUDA ----------
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA 驱动已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[ERROR] 未检测到 NVIDIA GPU，请确认集群有 GPU 节点"
    exit 1
fi

# ---------- Conda 环境 ----------
ENV_NAME="c-visual"

if conda env list | grep -q "$ENV_NAME"; then
    echo "[OK] Conda 环境 $ENV_NAME 已存在，跳过创建"
else
    echo "[创建] Conda 环境: $ENV_NAME (Python 3.10)"
    conda create -y -n "$ENV_NAME" python=3.10
fi

# ---------- 激活环境 ----------
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ---------- PyTorch (CUDA 11.8) ----------
echo "[安装] PyTorch 2.1.0 + CUDA 11.8"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# ---------- 其他依赖 ----------
echo "[安装] diffusers + 其他依赖"
pip install \
    diffusers>=0.27.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    peft>=0.6.0 \
    opencv-python>=4.8.0 \
    moviepy>=1.0.3 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    safetensors>=0.4.0 \
    controlnet-aux>=0.0.7

# ---------- 创建必要目录 ----------
mkdir -p output/{scenes,animations,frames,video} model_cache input/contours

echo ""
echo "==================================="
echo "  安装完成！"
echo "  用法:"
echo "    conda activate $ENV_NAME"
echo "    python scene_generator.py storyboard_example.json"
echo "    python animation_generator.py storyboard_example.json"
echo "    python video_composer.py storyboard_example.json"
echo "==================================="
