"""
video_composer.py
角色 C - 视频合成
拼接动画片段 + 转场 + 字幕 + 最终导出 MP4
"""

import json
import sys
from pathlib import Path

import numpy as np
try:
    from moviepy import (
        ImageSequenceClip,
        CompositeVideoClip,
        concatenate_videoclips,
        TextClip,
        vfx,
    )
except ImportError:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.VideoClip import TextClip
    import moviepy.video.fx as vfx

from config import *

def load_storyboard(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_clip_from_frames(frame_dir: Path, duration: float, fps: int) -> ImageSequenceClip:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"没有找到帧图片: {frame_dir}")
    clip = ImageSequenceClip([str(p) for p in frame_paths], fps=fps)
    if clip.duration < duration:
        n_repeats = int(np.ceil(duration / clip.duration))
        clip = concatenate_videoclips([clip] * n_repeats)
    return clip.subclipped(0, duration)

def add_subtitle(video, text: str, duration: float, start: float = 0):
    txt = TextClip(
        text=text,
        font_size=28,
        color="white",
        stroke_color="black",
        stroke_width=1,
        font="SimHei",
    ).with_position(("center", "bottom")).with_start(start).with_duration(duration)
    return txt

def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]

    print("=" * 50)
    print("视频合成")
    print("=" * 50)

    storyboard = None
    if len(args) >= 1:
        storyboard = load_storyboard(args[0])

    # 收集动画片段
    anim_dirs = sorted(ANIMATIONS_DIR.glob("animation_*"))
    anim_dirs = [d for d in anim_dirs if d.is_dir()]

    if not anim_dirs:
        print("⚠️  没有动画片段，用场景图生成幻灯片")
        scene_images = sorted(SCENES_DIR.glob("scene_*.png"))
        if not scene_images:
            print("❌ 没有任何素材，先运行 scene_generator.py 和 animation_generator.py")
            sys.exit(1)
        clips = []
        for img_path in scene_images:
            clip = ImageSequenceClip([str(img_path)] * 12, fps=FINAL_FPS).with_duration(3.0)
            clips.append(clip)
    else:
        clips = []
        for d in anim_dirs:
            clip = make_clip_from_frames(d, duration=3.0, fps=FPS)
            clips.append(clip)

    print(f"共 {len(clips)} 个片段")

    # 转场处理
    processed = []
    for i, clip in enumerate(clips):
        c = clip
        if i > 0:
            c = c.with_effects([vfx.FadeIn(TRANSITION_DURATION)])
        if i < len(clips) - 1:
            c = c.with_effects([vfx.FadeOut(TRANSITION_DURATION)])
        processed.append(c)

    video = concatenate_videoclips(processed, method="compose")

    # 字幕
    subtitles = []
    if storyboard:
        dur_per_scene = video.duration / len(storyboard)
        for i, scene in enumerate(storyboard):
            narration = scene.get("narration", scene.get("description", ""))
            if narration:
                subtitles.append(
                    add_subtitle(video, narration, dur_per_scene, i * dur_per_scene)
                )

    if subtitles:
        video = CompositeVideoClip([video] + subtitles)

    # 导出
    video_path = VIDEO_DIR / "final_video.mp4"
    video.write_videofile(
        str(video_path),
        fps=FINAL_FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="5000k",
    )
    print(f"\n✅ 视频已导出: {video_path}")
    print(f"   时长: {video.duration:.1f} 秒")
    print(f"   位置: {video_path.resolve()}")

if __name__ == "__main__":
    main()
