#!/bin/bash
# =============================================================
# run_pipeline.sh - 一键运行完整管线
# 用法: bash run_pipeline.sh <storyboard.json>
# 默认: storyboard_example.json
# =============================================================
set -e

STORYBOARD="${1:-storyboard_example.json}"
ENV_NAME="c-visual"

echo "==================================="
echo "  全链路 AI 视频生成管线"
echo "  角色 C - 视觉工程师"
echo "  分镜脚本: $STORYBOARD"
echo "==================================="

# ---------- 检测环境 ----------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda 未安装，请先运行 bash setup.sh"
    exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "[ERROR] Conda 环境 $ENV_NAME 不存在，请先运行 bash setup.sh"
    exit 1
fi
conda activate "$ENV_NAME"
echo "[OK] 环境: $ENV_NAME"

# ---------- GPU 检测 ----------
echo "[GPU]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"

# ---------- Step 1: 场景图生成 ----------
echo ""
echo "==================================="
echo "  Step 1/3: 场景图生成"
echo "==================================="
time python scene_generator.py "$STORYBOARD"

# ---------- Step 2: 动画生成 ----------
echo ""
echo "==================================="
echo "  Step 2/3: 动画生成"
echo "==================================="
time python animation_generator.py "$STORYBOARD"

# ---------- Step 3: 视频合成 ----------
echo ""
echo "==================================="
echo "  Step 3/3: 视频合成"
echo "==================================="
time python video_composer.py "$STORYBOARD"

# ---------- 完成 ----------
echo ""
echo "==================================="
echo "  管线运行完成！"
echo "  最终视频: $(pwd)/output/video/final_video.mp4"
echo "==================================="
