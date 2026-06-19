from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


class MotionVideoAgent:
    """Agent 4: 微动视频制作。

    调用通义万相 I2V 为每张场景图生成 3s 动态视频片段，
    再拼接为 12s 最终 MP4。 片段间使用 白闪+模糊 过渡。
    """

    FPS = 24
    TOTAL_SEC = 12.0
    FADE_SEC = 0.6
    W = 1024
    H = 1024
    CLIP_DURATION = 3

    def __init__(self, layout: dict[str, Any] | str | Path, output_dir: str | Path = ""):
        if isinstance(layout, (str, Path)):
            with open(layout, encoding="utf-8") as f:
                self.layout = json.load(f)
        else:
            self.layout = layout
        self.out_dir = Path(output_dir) if output_dir else Path("output")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ─────────────────────────────────────────────────────

    def render(self, max_retries: int = 3) -> str:
        scenes = self.layout["scenes"]
        w = self.layout.get("canvas_width", self.W)
        h = self.layout.get("canvas_height", self.H)

        clip_paths = []
        for i, scene in enumerate(scenes):
            img_path = scene.get("image_path", "")
            prompt = scene.get("prompt", "")
            clip_path = self.out_dir / f"clip_{i+1:02d}.mp4"
            if clip_path.exists():
                print(f"  [I2V] 场景 {i+1}/{len(scenes)} 已有缓存, 跳过")
                clip_paths.append(clip_path)
                continue
            print(f"  [I2V] 生成场景 {i+1}/{len(scenes)} 动态视频...")
            ok = False
            for attempt in range(1, max_retries + 1):
                try:
                    self._generate_scene_clip(img_path, prompt, str(clip_path))
                    ok = True
                    break
                except RuntimeError as e:
                    print(f"    尝试 {attempt}/{max_retries} 失败: {e}")
                    if attempt == max_retries:
                        print(f"    I2V 失败, 退回到静态幻灯片")
                    time.sleep(3)
            if not ok:
                self._generate_static_clip(img_path, str(clip_path), w, h)
            clip_paths.append(clip_path)

        video_path = self.out_dir / f"{self._sanitize(self.layout['place'])}_motion_{self._rand()}.mp4"
        print("  [Video] 拼接片段...")
        self._concatenate_clips(clip_paths, str(video_path), w, h)
        return str(video_path.resolve())

    # ── I2V 调用 ────────────────────────────────────────────────────

    def _generate_scene_clip(self, image_path: str, prompt: str, output_path: str) -> None:
        api_key = os.getenv("ALIBABA_API_KEY", "")
        if not api_key:
            raise RuntimeError("ALIBABA_API_KEY 未设置，无法调用图生视频")

        img = Image.open(image_path).convert("RGB")
        # I2V 720P 输出是 16:9，center-crop 到 16:9 避免自动裁切偏差
        iw, ih = img.size
        iw16 = iw
        ih16 = int(iw * 9 / 16)
        if ih > ih16:
            iy = (ih - ih16) // 2
            img = img.crop((0, iy, iw16, iy + ih16))
        img.thumbnail((1280, 1280), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        img_data_uri = f"data:image/jpeg;base64,{img_b64}"

        body = {
            "model": "wanx2.1-i2v-turbo",
            "input": {
                "img_url": img_data_uri,
                "prompt": prompt[:800],
            },
            "parameters": {
                "resolution": "720P",
                "duration": self.CLIP_DURATION,
            },
        }

        session = requests.Session()
        for attempt in range(3):
            try:
                r = session.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-Async": "enable",
                    },
                    json=body,
                    timeout=(30, 180),
                )
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 2:
                    raise RuntimeError(f"I2V 连接失败 (3次重试): {e}")
                print(f"    连接超时, 重试 {attempt+2}/3...")
                time.sleep(5)
        data = r.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"I2V 提交失败: {data}")

        for _ in range(120):
            time.sleep(5)
            q = requests.get(
                f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            q.raise_for_status()
            task_data = q.json().get("output", {})
            status = task_data.get("task_status")
            if status == "SUCCEEDED":
                video_url = task_data.get("video_url", "")
                if video_url:
                    self._download_video(video_url, output_path)
                    print(f"    clip saved: {output_path}")
                    return
                raise RuntimeError(f"I2V 结果无 video_url: {task_data}")
            elif status == "FAILED":
                raise RuntimeError(f"I2V 失败: {task_data.get('message', '未知')}")
        raise RuntimeError(f"I2V 超时: scene image {image_path}")

    @staticmethod
    def _download_video(url: str, output_path: str) -> None:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        Path(output_path).write_bytes(r.content)

    # ── 静态回退 ─────────────────────────────────────────────────────

    def _generate_static_clip(self, image_path: str, output_path: str, w: int, h: int) -> None:
        """I2V 失败时生成 Ken Burns 式静态幻灯片视频。"""
        fps = self.FPS
        total = int(self.CLIP_DURATION * fps)
        img = cv2.imread(image_path)
        if img is None:
            img = np.full((h, w, 3), 200, dtype=np.uint8)
        # 与 I2V 一致的 16:9 center-crop
        ih, iw = img.shape[:2]
        iw16 = iw
        ih16 = int(iw * 9 / 16)
        if ih > ih16:
            iy = (ih - ih16) // 2
            img = img[iy:iy + ih16, 0:iw16]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for t in range(total):
            s = 1.0 + 0.04 * (t / total)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, s)
            frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            writer.write(frame)
        writer.release()
        print(f"    static clip (Ken Burns): {output_path}")

    # ── 拼接 ────────────────────────────────────────────────────────

    def _concatenate_clips(self, clip_paths: list[Path], output_path: str, w: int, h: int) -> None:
        fps = self.FPS
        target_per_clip = int(self.CLIP_DURATION * fps)          # 72
        total_frames = int(self.TOTAL_SEC * fps)                  # 288
        fade_frames = int(self.FADE_SEC * fps)                    # 14
        half_fade = fade_frames // 2

        # 加载 mask + border（锁死在画面上，每帧重新裁切+描边）
        mask_np = None
        bg = np.array(self.layout.get("bg_color", [245, 240, 230]), dtype=np.uint8)
        mask_path = self.layout.get("mask_path", "")
        if mask_path and os.path.exists(mask_path):
            pil_mask = Image.open(mask_path).convert("L")
            # 与 I2V 相同的 16:9 center-crop，保证 mask 对齐
            mw, mh = pil_mask.size
            mw16 = mw
            mh16 = int(mw * 9 / 16)
            if mh > mh16:
                my = (mh - mh16) // 2
                pil_mask = pil_mask.crop((0, my, mw16, my + mh16))
            pil_mask = pil_mask.resize((w, h), Image.LANCZOS)
            mask_np = np.array(pil_mask).astype(np.float32) / 255.0
            print(f"  [Video] 地图 mask 重新裁切: {mask_path}")

        border_np = None
        border_overlay_path = self.layout.get("border_overlay", "")
        if border_overlay_path and os.path.exists(border_overlay_path):
            pil_border = Image.open(border_overlay_path).convert("RGBA")
            # 同样 16:9 center-crop
            bw, bh = pil_border.size
            bw16 = bw
            bh16 = int(bw * 9 / 16)
            if bh > bh16:
                by = (bh - bh16) // 2
                pil_border = pil_border.crop((0, by, bw16, by + bh16))
            pil_border = pil_border.resize((w, h), Image.LANCZOS)
            border_np = np.array(pil_border).astype(np.float32) / 255.0
            print(f"  [Video] 叠加静态地图边框: {border_overlay_path}")

        def apply_mask_and_border(frame: np.ndarray) -> np.ndarray:
            f = frame.astype(np.float32) / 255.0
            # 1) 用 mask 裁切内容，超出部分变 bg_color
            if mask_np is not None:
                m = mask_np[:, :, None]  # (h,w,1)
                bg_arr = np.full((h, w, 3), bg / 255.0, dtype=np.float32)
                f = f * m + bg_arr * (1.0 - m)
            # 2) 叠加边框
            if border_np is not None:
                a = border_np[:, :, 3:4]
                f = f * (1.0 - a) + border_np[:, :, :3] * a
            return (f * 255).astype(np.uint8)

        white = np.full((h, w, 3), 255, dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        prev_frames: list[np.ndarray] = []

        for idx, clip_path in enumerate(clip_paths):
            if not clip_path.exists():
                print(f"    skip missing clip: {clip_path}")
                continue

            raw = self._read_clip(str(clip_path), w, h)

            # 重采样到 target_per_clip 帧
            if len(raw) != target_per_clip:
                raw = self._resample_frames(raw, target_per_clip)

            if idx == 0:
                for f in raw:
                    writer.write(apply_mask_and_border(f))
            else:
                for i in range(half_fade):
                    p = (i + 1) / (half_fade + 1)
                    a = cv2.addWeighted(prev_frames[-(half_fade - i)], 1 - p, white, p, 0)
                    writer.write(apply_mask_and_border(cv2.GaussianBlur(a, (0, 0), sigmaX=p * 10)))

                for i in range(fade_frames - half_fade):
                    p = (i + 1) / (fade_frames - half_fade + 1)
                    a = cv2.addWeighted(white, 1 - p, raw[i], p, 0)
                    writer.write(apply_mask_and_border(cv2.GaussianBlur(a, (0, 0), sigmaX=(1 - p) * 10)))

                for f in raw[fade_frames:]:
                    writer.write(apply_mask_and_border(f))

            prev_frames = raw

        # 截断到 total_frames
        # (因为每段固定 target_per_clip 帧 + 过渡会多出 fade_frames)
        # 实现方法：直接限制帧数
        # 更好的做法：上面每段写入时手动计数

        writer.release()

        # 如果超出总帧数，重编码截断
        final = cv2.VideoCapture(output_path)
        all_frames: list[np.ndarray] = []
        while True:
            ret, f = final.read()
            if not ret:
                break
            all_frames.append(f)
        final.release()

        if len(all_frames) > total_frames:
            writer2 = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for f in all_frames[:total_frames]:
                writer2.write(f)
            writer2.release()

        print(f"  [Video] final: {output_path} ({len(all_frames)} frames)")

    # ── 辅助 ────────────────────────────────────────────────────────

    @staticmethod
    def _read_clip(path: str, w: int, h: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _resample_frames(frames: list[np.ndarray], n_target: int) -> list[np.ndarray]:
        if not frames:
            return [np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)] * n_target
        src_idx = np.linspace(0, len(frames) - 1, n_target)
        out = []
        for si in src_idx:
            i0 = int(np.floor(si))
            i1 = min(i0 + 1, len(frames) - 1)
            t = si - i0
            if t < 1e-6:
                out.append(frames[i0])
            else:
                out.append(cv2.addWeighted(frames[i0], 1 - t, frames[i1], t, 0))
        return out

    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower() or "place"

    @staticmethod
    def _rand() -> str:
        import uuid
        return uuid.uuid4().hex[:10]
