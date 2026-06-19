from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from mapgen.config import get_settings
from mapgen.place.osm import get_osm_contour


class ScatterLayoutAgent:
    """Agent 3: 地图散点排版。

    下载地图剪影 → 将 24 张贴纸按主题散落在地图区域内。
    输出 layout.json（含每个贴纸的位置、缩放、相位）供 Agent 4 使用。
    """

    CANVAS_W = 1080
    CANVAS_H = 768
    STICKER_BASE_SIZE = 170
    MIN_SCALE = 0.60
    MAX_SCALE = 1.20
    PADDING = 8
    CHARACTER_W = 400
    CHARACTER_H = 400

    def __init__(self, place: str, sticker_layouts: list[dict], output_dir: str | Path = "",
                 character_images: list[dict] | None = None):
        """sticker_layouts 来自 Agent 2 输出: [{ theme, items: [{item, path}, ...] }, ...]
        character_images 来自 Agent 2: [{ theme, prompt, path }, ...]"""
        self.place = place
        self.themes = sticker_layouts
        self.characters = character_images or []
        self.out_dir = Path(output_dir) if output_dir else get_settings().output_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def compose(self) -> dict[str, Any]:
        """主流程: 获取地图掩码 → 定位人物 → 贴纸均匀覆盖地图形状 → 输出 layout + 预览图。"""
        contour = get_osm_contour(self.place, {
            "output_width": self.CANVAS_W,
            "output_height": self.CANVAS_H,
            "output_dir": str(self.out_dir),
            "zoom": 14,
            "canvas_size": 800,
        })
        mask_img = Image.open(contour["mask_path"]).convert("L")
        mask_arr = np.array(mask_img)
        ys, xs_ = np.where(mask_arr > 0)
        if len(xs_) == 0:
            raise RuntimeError(f"地图掩码为空: {contour['mask_path']}")
        xs, ys = xs_, ys

        char_lookup = {c["theme"]: c["path"] for c in self.characters}

        # 在掩码内均匀采样与 sticker 数量相同的锚点，让贴纸铺满地图形状
        total_stickers = sum(len(t["items"]) for t in self.themes)
        anchors = self._sample_mesh_anchors(xs, ys, total_stickers)
        anchor_idx = 0

        scenes_layout = []
        placed_rects: list[tuple[int, int, int, int]] = []

        for t_idx, theme in enumerate(self.themes):
            theme_name = theme["theme"]
            items = theme["items"]
            stickers_info = []

            # 定位人物大图：放在地图中心偏下
            char_path = char_lookup.get(theme_name, "")
            if char_path:
                cw, ch = self.CHARACTER_W, self.CHARACTER_H
                cxm = int(xs.mean() - cw // 2)
                cym = int(ys.mean() - ch // 2)
                cxm = max(0, min(cxm, self.CANVAS_W - cw))
                cym = max(0, min(cym, self.CANVAS_H - ch))
                placed_rects.append((cxm, cym, cxm + cw, cym + ch))
            else:
                cxm, cym, cw, ch = 0, 0, 0, 0

            for item_idx, sticker in enumerate(items):
                img_path = sticker.get("path", "")
                scale = random.uniform(self.MIN_SCALE, self.MAX_SCALE)
                sw = int(self.STICKER_BASE_SIZE * scale)
                sh = int(self.STICKER_BASE_SIZE * scale)

                if anchor_idx < len(anchors):
                    ax, ay = anchors[anchor_idx]
                    anchor_idx += 1
                else:
                    ax, ay = int(xs.mean()), int(ys.mean())

                px = ax - sw // 2
                py = ay - sh // 2
                px = max(0, min(px, self.CANVAS_W - sw))
                py = max(0, min(py, self.CANVAS_H - sh))
                px = int(px)
                py = int(py)

                # 若与已放置的矩形重叠，小范围随机偏移
                for _ in range(30):
                    r = (px, py, px + sw, py + sh)
                    if not self._overlaps_any(r, placed_rects):
                        break
                    px += random.randint(-20, 20)
                    py += random.randint(-20, 20)
                    px = max(0, min(px, self.CANVAS_W - sw))
                    py = max(0, min(py, self.CANVAS_H - sh))

                placed_rects.append((px, py, px + sw, py + sh))

                phase = random.uniform(0, 2 * math.pi)
                resolved_path = str(Path(img_path).resolve()) if img_path else ""
                stickers_info.append({
                    "image_path": resolved_path,
                    "item": sticker["item"],
                    "theme": theme_name,
                    "x": px,
                    "y": py,
                    "width": sw,
                    "height": sh,
                    "scale": scale,
                    "phase": phase,
                    "float_amplitude": random.uniform(2.0, 5.0),
                    "float_frequency": random.uniform(0.6, 1.8),
                })

            scenes_layout.append({
                "theme": theme_name,
                "character": {
                    "image_path": str(Path(char_path).resolve()) if char_path else "",
                    "x": cxm,
                    "y": cym,
                    "width": cw,
                    "height": ch,
                    "phase": random.uniform(0, 2 * math.pi),
                    "float_amplitude": 3.0,
                    "float_frequency": 0.5,
                },
                "stickers": stickers_info,
            })

        layout = {
            "place": self.place,
            "canvas_width": self.CANVAS_W,
            "canvas_height": self.CANVAS_H,
            "map_mask_path": str(Path(contour["mask_path"]).resolve()),
            "contour": contour,
            "scenes": scenes_layout,
        }

        layout_path = self.out_dir / f"{self._sanitize(self.place)}_layout_{uuid4().hex[:10]}.json"
        layout_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")

        preview = self._render_preview(mask_img, scenes_layout)
        preview_path = self.out_dir / f"{self._sanitize(self.place)}_preview_{uuid4().hex[:10]}.png"
        preview.save(preview_path)

        layout["preview_path"] = str(preview_path.resolve())
        layout["layout_path"] = str(layout_path.resolve())
        return layout

    def _find_position(
        self, xs: np.ndarray, ys: np.ndarray,
        sw: int, sh: int,
        placed: list[tuple[int, int, int, int]],
        max_attempts: int = 1000,
    ) -> tuple[int, int] | None:
        # Phase 1: random sampling
        for _ in range(max_attempts):
            idx = random.randint(0, len(xs) - 1)
            cx, cy = int(xs[idx]), int(ys[idx])
            px = cx - sw // 2
            py = cy - sh // 2
            if px < 0 or py < 0 or px + sw > self.CANVAS_W or py + sh > self.CANVAS_H:
                continue
            if not self._overlaps(px, py, sw, sh, placed):
                return (px, py)
        # Phase 2: systematic scan (shuffle all coords)
        indices = list(range(len(xs)))
        random.shuffle(indices)
        for idx in indices:
            cx, cy = int(xs[idx]), int(ys[idx])
            px = cx - sw // 2
            py = cy - sh // 2
            if px < 0 or py < 0 or px + sw > self.CANVAS_W or py + sh > self.CANVAS_H:
                continue
            if not self._overlaps(px, py, sw, sh, placed):
                return (px, py)
        return None

    @staticmethod
    def _overlaps(px: int, py: int, sw: int, sh: int,
                  placed: list[tuple[int, int, int, int]]) -> bool:
        pad = ScatterLayoutAgent.PADDING
        r1 = (px - pad, py - pad, px + sw + pad, py + sh + pad)
        for r2 in placed:
            if r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1]:
                return True
        return False

    @staticmethod
    def _overlaps_any(r: tuple[int, int, int, int],
                      placed: list[tuple[int, int, int, int]],
                      pad: int = 8) -> bool:
        for r2 in placed:
            if r[0] < r2[2] and r[2] > r2[0] and r[1] < r2[3] and r[3] > r2[1]:
                return True
        return False

    @staticmethod
    def _sample_mesh_anchors(xs: np.ndarray, ys: np.ndarray, n: int) -> list[tuple[int, int]]:
        """在掩码内均匀采样 n 个锚点，用于贴纸摆放，使整体近似地图轮廓。"""
        if n <= 0:
            return []
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cols = max(1, int(round(math.sqrt(n * (x_max - x_min) / max(y_max - y_min, 1)))))
        rows = max(1, (n + cols - 1) // cols)

        cell_w = (x_max - x_min) / cols
        cell_h = (y_max - y_min) / rows

        points = []
        for r in range(rows):
            for c in range(cols):
                if len(points) >= n:
                    break
                cx = int(x_min + (c + 0.5) * cell_w)
                cy = int(y_min + (r + 0.5) * cell_h)
                dists = (xs - cx) ** 2 + (ys - cy) ** 2
                idx = int(dists.argmin())
                px, py = int(xs[idx]), int(ys[idx])
                # 加小随机偏移，避免过于整齐
                px += random.randint(-8, 8)
                py += random.randint(-8, 8)
                points.append((px, py))
            if len(points) >= n:
                break

        random.shuffle(points)
        return points[:n]

    @staticmethod
    def _near_edge(cx: int, cy: int, xs: np.ndarray, ys: np.ndarray,
                   threshold: int = 30) -> bool:
        """检查点是否靠近掩码边界。"""
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                nx, ny = cx + dx, cy + dy
                # 如果邻居不在掩码内，说明该点在边缘附近
                matches = np.where((xs == nx) & (ys == ny))[0]
                if len(matches) == 0:
                    return True
        return False

    @staticmethod
    def _edge_pixels(mask: np.ndarray, width: int = 3) -> np.ndarray:
        """返回掩码边缘的像素坐标。"""
        from scipy.ndimage import binary_erosion
        binary = mask > 0
        eroded = binary_erosion(binary, iterations=width)
        edge = binary & ~eroded
        return np.column_stack(np.where(edge))

    @staticmethod
    def _render_preview(mask_img: Image.Image,
                        scenes: list[dict]) -> Image.Image:
        """渲染一张预览图：米色背景 + 地图浅影 + 人物 + 贴纸。"""
        w, h = mask_img.size
        bg = Image.new("RGBA", (w, h), (255, 250, 240, 255))
        shadow = mask_img.convert("L").filter(
            __import__("PIL").ImageFilter.GaussianBlur(radius=4)
        )
        shadow_rgba = Image.new("RGBA", (w, h), (200, 190, 175, 255))
        shadow_rgba.putalpha(shadow)
        bg = Image.alpha_composite(bg, shadow_rgba)

        for scene in scenes:
            # 先画人物大图
            char = scene.get("character", {})
            cpath = char.get("image_path", "")
            if cpath and Path(cpath).exists():
                try:
                    cimg = Image.open(cpath).convert("RGBA")
                    cw, ch = char.get("width", 300), char.get("height", 400)
                    cimg = cimg.resize((cw, ch), __import__("PIL").Image.LANCZOS)
                    bg.paste(cimg, (char.get("x", 0), char.get("y", 0)), cimg)
                except FileNotFoundError:
                    pass

            # 再画小贴纸
            for s in scene["stickers"]:
                ipath = s.get("image_path", "")
                if not ipath or not Path(ipath).exists():
                    d = ImageDraw.Draw(bg)
                    d.rectangle([s["x"], s["y"],
                                 s["x"] + s["width"], s["y"] + s["height"]],
                                fill=(200, 180, 160))
                    continue
                try:
                    sticker = Image.open(ipath).convert("RGBA")
                    sticker = sticker.resize((s["width"], s["height"]),
                                             __import__("PIL").Image.LANCZOS)
                    bg.paste(sticker, (s["x"], s["y"]), sticker)
                except FileNotFoundError:
                    d = ImageDraw.Draw(bg)
                    d.rectangle([s["x"], s["y"],
                                 s["x"] + s["width"], s["y"] + s["height"]],
                                fill=(200, 180, 160))
        return bg

    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower() or "place"
