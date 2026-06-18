import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw

from config import *

MACARON = [
    (180, 210, 180),  # sage
    (230, 200, 170),  # beige
    (200, 190, 160),  # cream
    (160, 180, 210),  # cornflower
    (210, 180, 190),  # dusty rose
    (190, 200, 180),  # mint
]

def load_storyboard(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pipeline():
    print(f"[加载模型] {SCENE_BASE_MODEL}")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16, cache_dir=str(CACHE_DIR),
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SCENE_BASE_MODEL, controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None, cache_dir=str(CACHE_DIR),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="sigma_min",
    )
    if SCENE_VAE:
        pipe.vae = AutoencoderKL.from_pretrained(SCENE_VAE, torch_dtype=torch.float16)
    if ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
    if ENABLE_VAE_SLICING:
        pipe.enable_vae_slicing()
    pipe.to(DEVICE)
    return pipe

def generate_element(pipe, prompt: str, size: tuple[int, int], seed: int = 42) -> Image.Image:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    control = Image.new("RGB", size, "white")
    result = pipe(
        prompt=f"{STYLE_PREFIX}{prompt}, isolated object, white background{STYLE_SUFFIX}",
        negative_prompt=NEGATIVE_PROMPT,
        image=control,
        width=size[0], height=size[1],
        num_inference_steps=ELEMENT_STEPS,
        guidance_scale=SCENE_GUIDANCE,
        controlnet_conditioning_scale=0.0,
        generator=gen,
    ).images[0]
    return result

def remove_bg(img: Image.Image, threshold: int = 245) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img)
    white = (data[:, :, :3] > threshold).all(axis=2)
    data[white, 3] = 0
    return Image.fromarray(data)

def build_scene_prompt(scene: dict) -> str:
    sp = scene.get("scene_prompt", "")
    if sp:
        return f"{STYLE_PREFIX}{sp}{STYLE_SUFFIX}"
    element = scene.get("culture_element", "")
    desc = scene.get("description", "")
    core = (element or desc)[:12]
    return f"{STYLE_PREFIX}{core}, scenery, landscape{STYLE_SUFFIX}"

def compose_scene(pipe, storyboard: list[dict]):
    W, H = SCENE_WIDTH, SCENE_HEIGHT

    for i, scene in enumerate(storyboard):
        print(f"\n[场景 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        contour_path = scene.get("contour_map")
        if not contour_path or not Path(contour_path).exists():
            img = Image.new("RGB", (W, H), "white")
            img.save(SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png")
            continue

        # ── 加载居中轮廓掩码，缩放到画布尺寸 ──
        mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # ── ControlNet 条件：轮廓边缘线 ──
        edges = cv2.Canny(binary, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        control_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # ── 生成完整场景 ──
        prompt = build_scene_prompt(scene)
        print(f"  Prompt: {prompt[:80]}...")

        gen = torch.Generator(device=DEVICE).manual_seed(scene.get("seed", 42))
        scene_img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=control_img,
            width=W, height=H,
            num_inference_steps=SCENE_STEPS,
            guidance_scale=SCENE_GUIDANCE,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            generator=gen,
        ).images[0]

        # ── Mask 裁切：轮廓内保留场景，轮廓外白色 ──
        scene_np = np.array(scene_img)
        m3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) / 255.0
        bg = np.ones_like(scene_np) * 255
        cropped_np = (scene_np * m3 + bg * (1 - m3)).astype(np.uint8)
        canvas = Image.fromarray(cropped_np).convert("RGBA")
        draw = ImageDraw.Draw(canvas)

        # ── 获取轮廓边缘点（画布坐标）用于元素放置 ──
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            canvas.convert("RGB").save(SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png")
            continue

        cx, cy = int(xs.mean()), int(ys.mean())
        bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
        pts_idx = np.linspace(0, len(xs) - 1, NUM_CONTOUR_POINTS, endpoint=False, dtype=int)
        pts = [(int(xs[j]), int(ys[j])) for j in pts_idx]

        rnd = random.Random(scene.get("seed", 42))
        element_name = scene.get("culture_element", "local object")[:20]

        # ── 角色（轮廓中心） ──
        char_prompt = (
            f"character placeholder silhouette, standing, simple outline, {element_name}"
        )
        char_raw = generate_element(pipe, char_prompt, (ELEMENT_SIZE, ELEMENT_SIZE), seed=scene.get("seed", 42))
        char_no_bg = remove_bg(char_raw)
        c_size = int(ELEMENT_SIZE * 0.4)
        char_resized = char_no_bg.resize((c_size, c_size), Image.LANCZOS)
        cpx = int(cx - c_size / 2)
        cpy = int(cy - c_size / 2)
        canvas.paste(char_resized, (cpx, cpy), char_resized)

        # ── 生成物件图标 ──
        obj_imgs = []
        for k in range(OBJ_COUNT):
            prompts = [
                f"{element_name}, hand-drawn icon, isolated",
                f"{element_name} illustration, sticker style",
                f"traditional {element_name}, sketch, simple",
                f"small {element_name}, minimalist icon",
                f"decorative {element_name}, line art",
            ]
            raw = generate_element(
                pipe, prompts[k % len(prompts)], (OBJ_SIZE, OBJ_SIZE),
                seed=scene.get("seed", 42) + 100 + k,
            )
            obj_imgs.append(remove_bg(raw))

        # ── 沿轮廓边界放置物件 + 装饰圆点 ──
        dot_r = max(2, int(min(W, H) * 0.008))

        for j, (px, py) in enumerate(pts):
            if j % 3 == 0:
                obj = obj_imgs[(j // 3) % OBJ_COUNT]
                sz = int(OBJ_SIZE * 0.3 * rnd.uniform(0.7, 1.1))
                obj_resized = obj.resize((sz, sz), Image.LANCZOS)
                if rnd.random() > 0.3:
                    obj_resized = obj_resized.rotate(rnd.randint(-15, 15), expand=True, fillcolor=(0, 0, 0, 0))
                canvas.paste(
                    obj_resized,
                    (px - obj_resized.width // 2, py - obj_resized.height // 2),
                    obj_resized,
                )
            else:
                c = MACARON[j % len(MACARON)]
                r = dot_r * rnd.uniform(0.8, 2.0)
                draw.ellipse([px - r, py - r, px + r, py + r], fill=c + (200,))

        # ── 散布十字星装饰 ──
        for _ in range(NUM_CONTOUR_POINTS * 2):
            dx = rnd.randint(-20, 20) + int(W * 0.03)
            dy = rnd.randint(-20, 20) + int(H * 0.03)
            for bx, by in pts:
                sx, sy = bx + dx, by + dy
                if 0 <= sx < W and 0 <= sy < H:
                    ss = rnd.randint(2, 5)
                    clr = MACARON[rnd.randint(0, len(MACARON) - 1)] + (160,)
                    draw.line([sx - ss, sy, sx + ss, sy], fill=clr, width=1)
                    draw.line([sx, sy - ss, sx, sy + ss], fill=clr, width=1)
                    break

        out_path = SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
        canvas.convert("RGB").save(out_path)
        print(f"  保存: {out_path} (场景 + {(len(pts) + 2) // 3} 个边缘物件)")

def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    if len(args) < 1:
        print("用法: python scene_generator.py <storyboard.json>")
        sys.exit(1)
    storyboard = load_storyboard(args[0])
    print(f"加载分镜脚本: {len(storyboard)} 个场景")
    pipe = load_pipeline()
    compose_scene(pipe, storyboard)
    print("\n✅ 场景图生成完成！")

if __name__ == "__main__":
    main()
