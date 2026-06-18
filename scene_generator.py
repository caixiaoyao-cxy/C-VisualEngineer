"""
scene_generator.py
角色 C - 场景图生成
SD1.5 + ControlNet(Canny) + 地图轮廓约束 → 高质量场景图
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image

from config import *

def load_storyboard(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_contour(mask_path: str) -> Image.Image:
    """地图填充图 → ControlNet 条件：白色区域=可生成，黑色=禁止，硬约束"""
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, filled = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    colored = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(colored)

def build_prompt(scene: dict) -> str:
    element = scene.get("culture_element", "")
    desc = scene.get("description", "")
    core = (element or desc)[:12]
    return f"{STYLE_PREFIX}{core}, minimalist, flat illustration{STYLE_SUFFIX}"

def load_pipeline():
    print(f"[加载模型] 基础: {SCENE_BASE_MODEL}")
    print(f"[加载模型] ControlNet: {CONTROLNET_MODEL}")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=torch.float16,
        cache_dir=str(CACHE_DIR),
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SCENE_BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=str(CACHE_DIR),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        final_sigmas_type="sigma_min",
    )

    if SCENE_VAE:
        pipe.vae = AutoencoderKL.from_pretrained(SCENE_VAE, torch_dtype=torch.float16)

    if ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
    if ENABLE_VAE_SLICING:
        pipe.enable_vae_slicing()

    pipe.to(DEVICE)
    return pipe

def generate_scenes(pipe, storyboard: list[dict]):
    for i, scene in enumerate(storyboard):
        print(f"\n[场景 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        prompt = build_prompt(scene)
        print(f"  Prompt: {prompt[:80]}...")

        # 轮廓边缘条件
        contour_path = scene.get("contour_map")
        if contour_path and Path(contour_path).exists():
            control_image = extract_contour(contour_path)
            print(f"  使用轮廓约束: {contour_path}")
        else:
            control_image = Image.new("RGB", (SCENE_WIDTH, SCENE_HEIGHT), color="white")
            print(f"  无轮廓约束，生成纯白条件图")

        # 加载地图 mask 用于裁切（所有分支都执行）
        mask_img = None
        if contour_path and Path(contour_path).exists():
            mask_img = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
            _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        else:
            mask_img = np.ones((SCENE_HEIGHT, SCENE_WIDTH), dtype=np.uint8) * 255

        # 多张变体
        variants = scene.get("variants", 1)
        for v in range(variants):
                seed = scene.get("seed", 42) + v
                generator = torch.Generator(device=DEVICE).manual_seed(seed)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=control_image,
                    width=SCENE_WIDTH,
                    height=SCENE_HEIGHT,
                    num_inference_steps=SCENE_STEPS,
                    guidance_scale=SCENE_GUIDANCE,
                    controlnet_conditioning_scale=0.9,  # 高权重 → 元素严格分布在地图区域内
                    generator=generator,
                    num_images_per_prompt=SCENE_BATCH_SIZE,
                ).images[0]

                # 用地图 mask 裁切：轮廓外变白色背景
                result_np = np.array(result)
                mask_resized = cv2.resize(mask_img, (result_np.shape[1], result_np.shape[0]))
                mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB) / 255.0
                white_bg = np.ones_like(result_np) * 255
                result_np = (result_np * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
                result_masked = Image.fromarray(result_np)

                out_name = f"scene_{i+1:02d}_{scene.get('scene_id', '')}"
                if variants > 1:
                    out_name += f"_v{v+1}"
                out_path = SCENES_DIR / f"{out_name}.png"
                result_masked.save(out_path)
                print(f"  保存: {out_path}")

def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    if len(args) < 1:
        print("用法: python scene_generator.py <storyboard.json>")
        sys.exit(1)

    storyboard = load_storyboard(args[0])
    print(f"加载分镜脚本: {len(storyboard)} 个场景")

    pipe = load_pipeline()
    generate_scenes(pipe, storyboard)
    print("\n✅ 场景图生成完成！")

if __name__ == "__main__":
    main()
