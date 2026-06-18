import os
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
SCENES_DIR = OUTPUT_DIR / "scenes"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_DIR = OUTPUT_DIR / "video"
CACHE_DIR = ROOT / "model_cache"
ELEMENTS_DIR = OUTPUT_DIR / "elements"

for d in [OUTPUT_DIR, SCENES_DIR, ANIMATIONS_DIR, FRAMES_DIR, VIDEO_DIR, CACHE_DIR, ELEMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SCENE_BASE_MODEL = "Lykon/dreamshaper-8"
SCENE_VAE = "stabilityai/sd-vae-ft-mse"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
CONTROLNET_SCALE = 0.65
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
ANIMATION_BASE_MODEL = "Lykon/dreamshaper-8"
STYLE_LORA = None

SCENE_WIDTH = 768
SCENE_HEIGHT = 512
SCENE_STEPS = 25
SCENE_GUIDANCE = 7.5
SCENE_BATCH_SIZE = 1

NUM_FRAMES = 12
FPS = 6
ANIMATION_STEPS = 20
ANIMATION_GUIDANCE = 7.5

TRANSITION_DURATION = 0.3
FINAL_FPS = 24

# 元素生成
ELEMENT_SIZE = 320
ELEMENT_STEPS = 15
NUM_CONTOUR_POINTS = 16
CONTOUR_SCALE_RATIO = 0.55  # 轮廓占屏幕比例

# ============================================================
# Prompt 模板
# ============================================================
STYLE_PREFIX = (
    "best quality, pencil sketch, watercolor, marker rendering, "
    "low saturation, retro mint, macaron palette, sage green, beige, cream, cornflower blue, "
    "rough sketchy lines, hand-drawn, paper texture, "
)
STYLE_SUFFIX = (
    ", explosion composition, character at center, life objects surrounding radially, "
    "geometric wireframe border, film edge frame, "
    "decorative dots, cross stars, flowers, journal stickers, white highlights, water bloom"
)

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
    "smooth, CG, digital art, photorealistic, vibrant, high contrast, "
    "plain background, empty space, 3d render, photographic"
)

DEVICE = "cuda"
DTYPE = "float16"
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True
ENABLE_VAE_TILING = True
ENABLE_MODEL_CPU_OFFLOAD = False
