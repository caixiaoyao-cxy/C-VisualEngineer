from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter

from mapgen.config import get_settings
from mapgen.place.osm import get_osm_contour


def compose_video(place_name: str, image_paths: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a crossfade video with each photo masked to the map contour shape.

    Steps:
      1. Get OSM contour mask for place_name.
      2. Mask each image to the contour shape (with glow edge).
      3. Render crossfade frames into an MP4.

    Returns dict with video_path and metadata.
    """
    opts = options or {}
    output_dir = Path(opts.get("output_dir") or get_settings().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(opts.get("fps", 24))
    total_sec = float(opts.get("total_seconds", 5.0))
    fade_sec = float(opts.get("fade_seconds", 0.5))
    bg_color = tuple(opts.get("bg_color", (255, 250, 240)))

    # 1. Contour mask
    contour_opts = {
        "output_width": opts.get("output_width", 800),
        "output_height": opts.get("output_height", 600),
        "output_dir": str(output_dir),
    }
    contour = get_osm_contour(place_name, contour_opts)
    mask_path = contour["mask_path"]
    w = contour["width"]
    h = contour["height"]

    import cv2

    # 2. Mask each image
    mask_img = Image.open(mask_path).convert("L")
    masked = []
    for p in image_paths:
        img = Image.open(p).convert("RGBA").resize((w, h), Image.LANCZOS)
        m = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        result.paste(img, (0, 0), m)

        glow = m.filter(ImageFilter.GaussianBlur(radius=6))
        glow_arr = np.array(glow, dtype=np.float32) / 255.0
        glow_color = np.zeros((h, w, 4), dtype=np.uint8)
        glow_color[:, :, 0] = 220
        glow_color[:, :, 1] = 200
        glow_color[:, :, 2] = 180
        glow_color[:, :, 3] = (glow_arr * 120).astype(np.uint8)
        result = Image.alpha_composite(Image.fromarray(glow_color), result)
        masked.append(result)

    # 3. Render crossfade frames
    total_frames = int(total_sec * fps)
    seg = total_frames // len(masked)
    fade = int(fade_sec * fps)

    video_path = output_dir / f"{_sanitize(place_name)}_video_{uuid4().hex[:10]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for fi in range(total_frames):
        s = fi // seg
        pos = fi % seg

        if s < len(masked) - 1:
            if pos < seg - fade:
                frame = masked[s]
            else:
                t = (pos - (seg - fade)) / max(fade, 1)
                frame = Image.blend(masked[s], masked[s + 1], t)
        else:
            frame = masked[-1]

        bg = Image.new("RGBA", (w, h), (*bg_color, 255))
        bg = Image.alpha_composite(bg, frame)
        writer.write(cv2.cvtColor(np.array(bg.convert("RGB")), cv2.COLOR_RGB2BGR))

    writer.release()

    return {
        "place_name": place_name,
        "video_path": str(video_path.resolve()),
        "width": w,
        "height": h,
        "fps": fps,
        "total_frames": total_frames,
        "contour": contour,
        "masked_count": len(masked),
    }


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower() or "place"
