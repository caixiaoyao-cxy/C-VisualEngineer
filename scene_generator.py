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
    (180, 210, 180), (230, 200, 170), (200, 190, 160),
    (160, 180, 210), (210, 180, 190), (190, 200, 180),
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

def sample_interior(mask: np.ndarray, n: int, margin: int = 15):
    kernel = np.ones((margin, margin), np.uint8)
    eroded = cv2.erode(mask, kernel)
    ys, xs = np.where(eroded > 0)
    if len(xs) == 0:
        return []
    if n >= len(xs):
        return list(zip(xs, ys))
    idx = np.random.choice(len(xs), n, replace=False)
    return [(int(xs[j]), int(ys[j])) for j in idx]

def compose_scene(pipe, storyboard: list[dict]):
    W, H = SCENE_WIDTH, SCENE_HEIGHT
    bg_color = (248, 245, 240)  # warm macaron

    for i, scene in enumerate(storyboard):
        print(f"\n[场景 {i+1}/{len(storyboard)}] {scene.get('description', '')}")

        contour_path = scene.get("contour_map")
        if not contour_path or not Path(contour_path).exists():
            img = Image.new("RGB", (W, H), bg_color)
            img.save(SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png")
            continue

        # ── 加载轮廓掩码 ──
        mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # ── 获取边缘点和内部点（画布坐标） ──
        edges = cv2.Canny(binary, 100, 200)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            Image.new("RGB", (W, H), bg_color).save(
                SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
            )
            continue

        pts_idx = np.linspace(0, len(xs) - 1, NUM_CONTOUR_POINTS, endpoint=False, dtype=int)
        edge_pts = [(int(xs[j]), int(ys[j])) for j in pts_idx]

        cx, cy = int(xs.mean()), int(ys.mean())
        rnd = random.Random(scene.get("seed", 42))

        canvas = Image.new("RGBA", (W, H), bg_color + (255,))
        draw = ImageDraw.Draw(canvas)

        element_name = scene.get("culture_element", "local object")[:20]

        # ── 生成角色（中心） ──
        char_raw = generate_element(
            pipe,
            f"character placeholder silhouette, standing, simple outline, {element_name}",
            (ELEMENT_SIZE, ELEMENT_SIZE),
            seed=scene.get("seed", 42),
        )
        char_no_bg = remove_bg(char_raw)
        c_size = int(ELEMENT_SIZE * 0.35)
        char_resized = char_no_bg.resize((c_size, c_size), Image.LANCZOS)
        cpx = int(cx - c_size / 2)
        cpy = int(cy - c_size / 2)
        canvas.paste(char_resized, (cpx, cpy), char_resized)

        # ── 生成 24 个独特物件（全为具体物件，不生成场景） ──
        obj_topics = [
            f"{element_name}, hand-drawn icon", f"{element_name} illustration, sticker",
            f"traditional {element_name}, sketch", f"small {element_name}, minimalist",
            f"decorative {element_name}, line art", f"cute {element_name} mascot, doodle",
            f"vintage {element_name} badge, emblem", f"flat {element_name} vector",
            f"{element_name} with leaves, nature", f"ornamental {element_name}, pattern",
            f"simple {element_name} symbol, logo", f"hand-drawn {element_name} flowers",
            f"minimal {element_name} line drawing", f"watercolor {element_name} splash",
            f"pencil sketch {element_name} study", f"folk art {element_name} motif",
            f"tiny {element_name}, icon style", f"{element_name} bud, sprout, doodle",
            f"round {element_name} badge, sticker", f"{element_name} silhouette, minimal",
            f"stamped {element_name}, postmark style", f"woven {element_name} pattern",
            f"folded {element_name}, origami style", f"baked {element_name}, clay charm",
        ]
        obj_imgs = []
        for k in range(24):
            raw = generate_element(
                pipe, obj_topics[k], (OBJ_SIZE, OBJ_SIZE),
                seed=scene.get("seed", 42) + 100 + k * 7,
            )
            obj_imgs.append(remove_bg(raw))

        # ── 物件密铺轮廓边界（48 点全放）+ 内部散布填补 ──
        inside_pts = sample_interior(binary, 10, margin=20)
        obj_idx = 0

        for j, (px, py) in enumerate(edge_pts):
            src = obj_imgs[obj_idx % len(obj_imgs)]
            obj_idx += 1
            if j % 3 == 0:
                sz = int(OBJ_SIZE * 0.35 * rnd.uniform(0.8, 1.1))
            else:
                sz = int(OBJ_SIZE * 0.22 * rnd.uniform(0.6, 0.9))
                px += rnd.randint(-6, 6)
                py += rnd.randint(-6, 6)
            obj_resized = src.resize((sz, sz), Image.LANCZOS)
            obj_resized = obj_resized.rotate(rnd.randint(-25, 25), expand=True, fillcolor=(0, 0, 0, 0))
            canvas.paste(
                obj_resized,
                (px - obj_resized.width // 2, py - obj_resized.height // 2),
                obj_resized,
            )

        # 内部散布大号物件（充当场景填充，不像场景贴片那样不可控）
        for pt in inside_pts:
            src = obj_imgs[rnd.randint(0, len(obj_imgs) - 1)]
            sz = int(OBJ_SIZE * 0.4 * rnd.uniform(0.8, 1.3))
            obj_resized = src.resize((sz, sz), Image.LANCZOS)
            obj_resized = obj_resized.rotate(rnd.randint(-20, 20), expand=True, fillcolor=(0, 0, 0, 0))
            canvas.paste(
                obj_resized,
                (pt[0] - obj_resized.width // 2, pt[1] - obj_resized.height // 2),
                obj_resized,
            )

        # ── 散布十字星 ──
        for _ in range(NUM_CONTOUR_POINTS * 2):
            dx = rnd.randint(-18, 18) + int(W * 0.03)
            dy = rnd.randint(-18, 18) + int(H * 0.03)
            for bx, by in edge_pts:
                sx, sy = bx + dx, by + dy
                if 0 <= sx < W and 0 <= sy < H:
                    ss = rnd.randint(2, 4)
                    clr = MACARON[rnd.randint(0, len(MACARON) - 1)] + (160,)
                    draw.line([sx - ss, sy, sx + ss, sy], fill=clr, width=1)
                    draw.line([sx, sy - ss, sx, sy + ss], fill=clr, width=1)
                    break

        out_path = SCENES_DIR / f"scene_{i+1:02d}_{scene.get('scene_id', '')}.png"
        canvas.convert("RGB").save(out_path)
        print(f"  保存: {out_path} (6 场景贴片 + 1 角色 + 16 物件)")

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
