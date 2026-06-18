"""
animation_generator.py
角色 C - 动画片段生成
AnimateDiff → 每个场景 8-12 帧小动画
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
)
from diffusers.utils import export_to_gif
from PIL import Image

from config import *

def load_storyboard(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(scene: dict) -> str:
    scene_prompt = scene.get("scene_prompt", "")
    if scene_prompt:
        core = scene_prompt
    else:
        element = scene.get("culture_element", "")
        desc = scene.get("description", "")
        core = (element or desc)[:12]
    return f"{STYLE_PREFIX}{core}, minimalist, flat illustration{STYLE_SUFFIX}"

def load_pipeline():
    print(f"[加载 AnimateDiff] 基础模型: {ANIMATION_BASE_MODEL}")
    print(f"[加载 AnimateDiff] 运动适配器: {MOTION_ADAPTER}")

    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER,
        torch_dtype=torch.float16,
        cache_dir=str(CACHE_DIR),
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        ANIMATION_BASE_MODEL,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=str(CACHE_DIR),
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        ANIMATION_BASE_MODEL,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    if ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
    if ENABLE_VAE_SLICING:
        pipe.enable_vae_slicing()

    # 24GB 显存无需 offload，但保留选项
    if ENABLE_MODEL_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    return pipe

def generate_animations(pipe, storyboard: list[dict]):
    for i, scene in enumerate(storyboard):
        print(f"\n[动画 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        prompt = build_prompt(scene)
        print(f"  Prompt: {prompt[:80]}...")

        seed = scene.get("seed", 42) + i * 100
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # 尝试加载场景图作为 conditioning（可选）
        scene_img = None
        scene_id = scene.get("scene_id", f"{i+1:02d}")
        scene_paths = list(SCENES_DIR.glob(f"scene_{i+1:02d}_*"))
        if scene_paths:
            scene_img_path = scene_paths[0]
            scene_img = Image.open(scene_img_path).resize((SCENE_WIDTH, SCENE_HEIGHT))
            print(f"  参考场景图: {scene_img_path.name}")

        # 加载填充 mask 用于裁切（轮廓外留白，凸显物件拼出的轮廓边界）
        contour_path = scene.get("contour_map")
        fill_mask = None
        if contour_path and Path(contour_path).exists():
            m = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
            _, fill_mask = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=NUM_FRAMES,
            guidance_scale=ANIMATION_GUIDANCE,
            num_inference_steps=ANIMATION_STEPS,
            generator=generator,
            width=SCENE_WIDTH,
            height=SCENE_HEIGHT,
        )

        frames = output.frames[0]
        out_dir = ANIMATIONS_DIR / f"animation_{i+1:02d}_{scene_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cropped_frames = []
        for j, frame in enumerate(frames):
            if fill_mask is not None:
                fnp = np.array(frame)
                m = cv2.resize(fill_mask, (fnp.shape[1], fnp.shape[0]))
                m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB) / 255.0
                bg = np.ones_like(fnp) * 255
                fnp = (fnp * m3 + bg * (1 - m3)).astype(np.uint8)
                out_frame = Image.fromarray(fnp)
                out_frame.save(out_dir / f"frame_{j:04d}.png")
                cropped_frames.append(out_frame)
            else:
                frame.save(out_dir / f"frame_{j:04d}.png")
                cropped_frames.append(frame)

        gif_path = ANIMATIONS_DIR / f"animation_{i+1:02d}_{scene_id}.gif"
        export_to_gif(cropped_frames, str(gif_path))
        print(f"  保存: {gif_path} ({len(frames)} 帧)")

def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    if len(args) < 1:
        print("用法: python animation_generator.py <storyboard.json>")
        sys.exit(1)

    storyboard = load_storyboard(args[0])
    print(f"加载分镜脚本: {len(storyboard)} 个场景")

    pipe = load_pipeline()
    generate_animations(pipe, storyboard)
    print("\n✅ 动画生成完成！")

if __name__ == "__main__":
    main()
