import io
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

def get_contour_data(mask_path: str):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 100, 200)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return None, None, binary
    cx, cy = int(xs.mean()), int(ys.mean())
    bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    return (xs, ys), (cx, cy, bw, bh), binary

def sample_contour_points(xs, ys, n: int):
    if len(xs) == 0:
        return []
    indices = np.linspace(0, len(xs) - 1, n, endpoint=False, dtype=int)
    return [(int(xs[i]), int(ys[i])) for i in indices]

def compose_scene(pipe, storyboard: list[dict]):
    W, H = SCENE_WIDTH, SCENE_HEIGHT
    macaron_colors = [
        (180, 210, 180),  # sage
        (230, 200, 170),  # beige
        (200, 190, 160),  # cream
        (160, 180, 210),  # cornflower
        (210, 180, 190),  # dusty rose
        (190, 200, 180),  # mint
    ]

    for i, scene in enumerate(storyboard):
        print(f"\n[场景 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        contour_path = scene.get("contour_map")
        if not contour_path or not Path(contour_path).exists():
            img = Image.new("RGB", (W, H), "white")
            img.save(SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png")
            continue

        xs_ys, (cx, cy, bw, bh), binary = get_contour_data(contour_path)
        if xs_ys is None:
            continue

        scale = min(W * CONTOUR_SCALE_RATIO / bw, H * CONTOUR_SCALE_RATIO / bh) * 0.85
        ox, oy = (W - bw * scale) // 2, (H - bh * scale) // 2

        pts = sample_contour_points(xs_ys[0], xs_ys[1], NUM_CONTOUR_POINTS)
        rnd = random.Random(scene.get("seed", 42))

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # ── 1. 角色（中心） ──
        char_prompt = (
            f"character placeholder silhouette, standing, simple outline, "
            f"{scene.get('culture_element', 'person')[:20]}"
        )
        char_raw = generate_element(
            pipe, char_prompt, (ELEMENT_SIZE, ELEMENT_SIZE),
            seed=scene.get("seed", 42),
        )
        char_no_bg = remove_bg(char_raw)
        c_size = int(ELEMENT_SIZE * 0.45)
        char_resized = char_no_bg.resize((c_size, c_size), Image.LANCZOS)
        cpx = ox + int((bw * scale - c_size) / 2)
        cpy = oy + int((bh * scale - c_size) / 2)
        canvas.paste(char_resized, (cpx, cpy), char_resized)

        # ── 2. 生成 OBJ_COUNT 个独特物件 ──
        element_name = scene.get("culture_element", "local object")[:20]
        obj_imgs: list[Image.Image] = []
        for k in range(OBJ_COUNT):
            prompts = [
                f"{element_name}, hand-drawn icon, isolated",
                f"{element_name} illustration, sticker style",
                f"traditional {element_name}, sketch, simple",
                f"small {element_name}, minimalist icon",
                f"decorative {element_name}, line art",
            ]
            p = prompts[k % len(prompts)]
            raw = generate_element(
                pipe, p, (OBJ_SIZE, OBJ_SIZE),
                seed=scene.get("seed", 42) + 100 + k,
            )
            nobg = remove_bg(raw)
            obj_imgs.append(nobg)

        # ── 3. 沿轮廓密集放置 ──
        dot_r = max(3, int(scale * 3))

        for j, (px, py) in enumerate(pts):
            sx = int((px - (cx - bw // 2)) * scale + ox)
            sy = int((py - (cy - bh // 2)) * scale + oy)

            if j % 3 == 0:
                # SD 物件（每3个点放一个）
                obj = obj_imgs[(j // 3) % OBJ_COUNT]
                base = int(OBJ_SIZE * 0.35)
                sz = int(base * rnd.uniform(0.7, 1.1))
                obj_resized = obj.resize((sz, sz), Image.LANCZOS)
                if rnd.random() > 0.3:
                    obj_resized = obj_resized.rotate(rnd.randint(-15, 15), expand=True, fillcolor=(0, 0, 0, 0))
                canvas.paste(
                    obj_resized,
                    (sx - obj_resized.width // 2, sy - obj_resized.height // 2),
                    obj_resized,
                )
            else:
                # 装饰圆点（macaron 色盘）
                c = macaron_colors[j % len(macaron_colors)]
                r = dot_r * rnd.uniform(0.8, 2.0)
                draw.ellipse(
                    [sx - r, sy - r, sx + r, sy + r],
                    fill=c + (180,),
                )

        # ── 4. 散布十字星装饰 ──
        for _ in range(NUM_CONTOUR_POINTS * 2):
            dx = rnd.randint(-25, 25) + int(W * 0.04)
            dy = rnd.randint(-25, 25) + int(H * 0.04)
            for bx, by in pts:
                sx = int((bx - (cx - bw // 2)) * scale + ox) + dx
                sy = int((by - (cy - bh // 2)) * scale + oy) + dy
                if 0 <= sx < W and 0 <= sy < H:
                    ss = rnd.randint(2, 5)
                    clr = macaron_colors[rnd.randint(0, len(macaron_colors) - 1)] + (160,)
                    draw.line([sx - ss, sy, sx + ss, sy], fill=clr, width=1)
                    draw.line([sx, sy - ss, sx, sy + ss], fill=clr, width=1)
                    break

        out_path = SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
        canvas.convert("RGB").save(out_path)
        print(f"  保存: {out_path} ({len(pts)} 个轮廓点, {OBJ_COUNT} 种物件)")

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
