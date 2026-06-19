"""Media tools: TTS dubbing, subtitle generation, and video editing via ffmpeg."""
from __future__ import annotations

from .subtitles import (
    align_subtitle_to_audio,
    generate_subtitle,
    parse_srt,
    script_to_cues,
    write_srt,
)
from .tts import synthesize_dubbing
from .video import (
    burn_subtitle,
    clip_video,
    concat_videos,
    mux_audio,
    probe_media,
)
from .pipeline import synthesize_with_subtitle

__all__ = [
    "align_subtitle_to_audio",
    "burn_subtitle",
    "clip_video",
    "concat_videos",
    "generate_subtitle",
    "mux_audio",
    "parse_srt",
    "probe_media",
    "script_to_cues",
    "synthesize_dubbing",
    "synthesize_with_subtitle",
    "write_srt",
]
