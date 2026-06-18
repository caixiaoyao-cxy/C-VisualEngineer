"""
video_composer.py
角色 C - 视频合成
拼接动画片段 + 转场 + 字幕 + 最终导出 MP4
"""

import json
import sys
from pathlib import Path

import numpy as np

MOVIEPY_V2 = True
try:
    from moviepy import ImageSequenceClip, concatenate_videoclips
    from moviepy.video.fx import FadeIn, FadeOut
except ImportError:
    MOVIEPY_V2 = False
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.VideoClip import ImageClip
    import moviepy.video.fx.all as vfx
    from moviepy.video.fx import fadein, fadeout

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
    return clip.subclip(0, duration)


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

    processed = []
    for i, clip in enumerate(clips):
        effects = []
        if i > 0:
            effects.append(FadeIn(TRANSITION_DURATION) if MOVIEPY_V2 else None)
        if i < len(clips) - 1:
            effects.append(FadeOut(TRANSITION_DURATION) if MOVIEPY_V2 else None)
        if MOVIEPY_V2:
            effects = [e for e in effects if e is not None]
            c = clip.with_effects(effects) if effects else clip
        else:
            c = clip
            if i > 0:
                c = vfx.fadein(c, TRANSITION_DURATION)
            if i < len(clips) - 1:
                c = vfx.fadeout(c, TRANSITION_DURATION)
        processed.append(c)

    video = concatenate_videoclips(processed, method="compose")

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
