import json
import random
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
        prompt=f"{STYLE_PREFIX}{prompt}, isolated on white{STYLE_SUFFIX}",
        negative_prompt=NEGATIVE_PROMPT,
        image=control,
        width=size[0], height=size[1],
        num_inference_steps=ELEMENT_STEPS,
        guidance_scale=SCENE_GUIDANCE,
        controlnet_conditioning_scale=0.0,
        generator=gen,
    ).images[0]
    return result

def remove_bg(img: Image.Image, threshold: int = 240) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img)
    white = (data[:, :, :3] > threshold).all(axis=2)
    data[white, 3] = 0
    return Image.fromarray(data)

def compose_scene(pipe, storyboard: list[dict]):
    W, H = SCENE_WIDTH, SCENE_HEIGHT

    for i, scene in enumerate(storyboard):
        print(f"\n[场景 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        contour_path = scene.get("contour_map")
        if not contour_path or not Path(contour_path).exists():
            img = Image.new("RGB", (W, H), "white")
            img.save(SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png")
            continue

        # ── 加载轮廓掩码 ──
        mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # ── 轮廓边缘点（加随机抖动，保持大致形状但不精确追踪） ──
        edges = cv2.Canny(binary, 100, 200)
        ys, xs = np.where(edges > 0)
        if len(xs) < 10:
            Image.new("RGB", (W, H), "white").save(
                SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
            )
            continue

        rnd = random.Random(scene.get("seed", 42))
        n_pts = min(40, len(xs))
        idx = rnd.sample(range(len(xs)), n_pts)
        # 边缘点加抖动自然散布
        all_pts = []
        for j in idx:
            px = int(xs[j]) + rnd.randint(-18, 18)
            py = int(ys[j]) + rnd.randint(-18, 18)
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))
            all_pts.append((px, py))

        cy, cx = int(ys.mean()), int(xs.mean())

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

        element_name = scene.get("culture_element", "local object")[:20]
        place_name = scene.get("place_name", "")

        # ── 生成角色（中心） ──
        char_prompt = f"character placeholder silhouette, {place_name} local, standing, simple outline, {element_name}"
        char_raw = generate_element(
            pipe, char_prompt, (ELEMENT_SIZE, ELEMENT_SIZE),
            seed=scene.get("seed", 42),
        )
        char_no_bg = remove_bg(char_raw)
        c_size = int(ELEMENT_SIZE * 0.35)
        char_resized = char_no_bg.resize((c_size, c_size), Image.LANCZOS)
        cpx = int(cx - c_size / 2)
        cpy = int(cy - c_size / 2)
        canvas.paste(char_resized, (cpx, cpy), char_resized)

        # ── 生成 24 个独特物件（关联地名 + 文化元素） ──
        obj_topics = [
            f"{place_name} {element_name}, hand-drawn icon",
            f"{place_name} {element_name} illustration, sticker",
            f"traditional {place_name} {element_name}, sketch",
            f"small {place_name} {element_name}, minimalist",
            f"decorative {place_name} {element_name}, line art",
            f"cute {place_name} {element_name} mascot, doodle",
            f"vintage {place_name} badge, emblem",
            f"flat {place_name} cultural vector",
            f"{place_name} {element_name} with leaves",
            f"ornamental {place_name} pattern",
            f"simple {place_name} symbol, logo",
            f"hand-drawn {place_name} flowers decor",
            f"minimal {place_name} line drawing",
            f"watercolor {place_name} splash",
            f"pencil sketch {place_name} study",
            f"folk art {place_name} motif",
            f"tiny {place_name} icon style",
            f"{place_name} cultural bud, sprout, doodle",
            f"round {place_name} badge, sticker",
            f"{place_name} silhouette, minimal",
            f"stamped {place_name} postmark style",
            f"woven {place_name} pattern",
            f"folded {place_name} origami style",
            f"baked {place_name} clay charm",
        ]
        obj_imgs = []
        for k in range(24):
            raw = generate_element(
                pipe, obj_topics[k], (OBJ_SIZE, OBJ_SIZE),
                seed=scene.get("seed", 42) + 100 + k * 7,
            )
            obj_imgs.append(remove_bg(raw))

        # ── 物件散布（边缘大物件 + 内部小点缀） ──
        n_boundary = min(len(all_pts), 30)
        boundary = all_pts[:n_boundary]
        interior = all_pts[n_boundary:]

        obj_idx = 0
        for px, py in boundary:
            src = obj_imgs[obj_idx % len(obj_imgs)]
            obj_idx += 1
            sz = int(OBJ_SIZE * 0.35 * rnd.uniform(0.7, 1.1))
            obj_resized = src.resize((sz, sz), Image.LANCZOS)
            obj_resized = obj_resized.rotate(rnd.randint(-30, 30), expand=True, fillcolor=(0, 0, 0, 0))
            canvas.paste(
                obj_resized,
                (px - obj_resized.width // 2, py - obj_resized.height // 2),
                obj_resized,
            )

        for px, py in interior:
            src = obj_imgs[rnd.randint(0, len(obj_imgs) - 1)]
            sz = int(OBJ_SIZE * 0.25 * rnd.uniform(0.5, 0.9))
            obj_resized = src.resize((sz, sz), Image.LANCZOS)
            obj_resized = obj_resized.rotate(rnd.randint(-20, 20), expand=True, fillcolor=(0, 0, 0, 0))
            canvas.paste(
                obj_resized,
                (px - obj_resized.width // 2, py - obj_resized.height // 2),
                obj_resized,
            )

        out_path = SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
        canvas.convert("RGB").save(out_path)
        print(f"  保存: {out_path} (1 角色 + 24 物件 + {len(all_pts)} 散布点)")

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
