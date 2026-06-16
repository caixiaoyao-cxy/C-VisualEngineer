import os
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
SCENES_DIR = OUTPUT_DIR / "scenes"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_DIR = OUTPUT_DIR / "video"
CACHE_DIR = ROOT / "model_cache"

for d in [OUTPUT_DIR, SCENES_DIR, ANIMATIONS_DIR, FRAMES_DIR, VIDEO_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 模型配置（HuggingFace 模型 ID）
# ============================================================
# 场景生成模型（SD1.5 写实偏二次元风格）
SCENE_BASE_MODEL = "Lykon/dreamshaper-8"
SCENE_VAE = "stabilityai/sd-vae-ft-mse"

# ControlNet（Canny 边缘约束）
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"

# AnimateDiff 运动适配器
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
ANIMATION_BASE_MODEL = "Lykon/dreamshaper-8"

# LoRA 风格（二次元强化，可选）
STYLE_LORA = None  # 或 "ntc-ai/SD-LoRA-anime-2"

# ============================================================
# 生成参数
# ============================================================
# 场景图
SCENE_WIDTH = 768
SCENE_HEIGHT = 512
SCENE_STEPS = 25
SCENE_GUIDANCE = 7.5
SCENE_BATCH_SIZE = 1

# 动画
NUM_FRAMES = 12
FPS = 6
ANIMATION_STEPS = 20
ANIMATION_GUIDANCE = 7.5

# 视频合成
TRANSITION_DURATION = 0.3  # 转场时长（秒）
FINAL_FPS = 24

# ============================================================
# Prompt 模板
# ============================================================
STYLE_PREFIX = "masterpiece, best quality, anime style, hand-drawn style, fairy tale style, vibrant colors, detailed background, "
STYLE_SUFFIX = ", cinematic lighting, depth of field, high resolution"

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
    "landscape, sky, ground, grass, trees, building exterior, scenery, background detail"
)

# ============================================================
# CUDA / 内存优化
# ============================================================
DEVICE = "cuda"
DTYPE = "float16"  # 半精度节省显存
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
ENABLE_VAE_TILING = True
ENABLE_MODEL_CPU_OFFLOAD = False  # 24G 显存可以关掉 offload
