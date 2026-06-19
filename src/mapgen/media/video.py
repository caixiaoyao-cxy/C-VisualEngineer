"""Video editing helpers: clipping, concatenation, subtitle burning, audio muxing."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..config import get_settings
from .ffmpeg import ffprobe_json, probe_duration, run_ffmpeg


# ---------- helpers ----------


def _resolve_output(path: str | Path | None, default_name: str) -> Path:
    if path is None:
        out_dir = get_settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _format_time(value: Any) -> str:
    """Accept HH:MM:SS(.ms) strings or numeric seconds, return ffmpeg-friendly string."""
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"negative time value: {value}")
        td = timedelta(seconds=float(value))
        total = td.total_seconds()
        hours, rem = divmod(int(total), 3600)
        minutes, seconds = divmod(rem, 60)
        millis = int(round((total - int(total)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    if isinstance(value, str):
        return value.strip()
    raise TypeError(f"unsupported time value type: {type(value).__name__}")


def _escape_subtitle_path(path: Path) -> str:
    """Escape a path for use inside the ffmpeg ``subtitles=`` filter."""
    posix = path.resolve().as_posix()
    # The drive letter colon needs escaping on Windows, and any single-quote.
    return posix.replace(":", r"\:").replace("'", r"\'")


# ---------- public tools ----------


def probe_media(path: str | Path) -> dict[str, Any]:
    """Return a flat description of a media file (duration, video/audio streams)."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    info = ffprobe_json(path)
    streams = info.get("streams", []) or []
    fmt = info.get("format", {}) or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "path": str(path),
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
        "format_name": fmt.get("format_name"),
        "video": {
            "codec": (video or {}).get("codec_name"),
            "width": (video or {}).get("width"),
            "height": (video or {}).get("height"),
            "fps": (video or {}).get("avg_frame_rate"),
        }
        if video
        else None,
        "audio": {
            "codec": (audio or {}).get("codec_name"),
            "sample_rate": (audio or {}).get("sample_rate"),
            "channels": (audio or {}).get("channels"),
        }
        if audio
        else None,
    }


def clip_video(
    input_path: str | Path,
    segments: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
    re_encode: bool = False,
) -> dict[str, Any]:
    """Cut ``input_path`` into one or more segments.

    Each ``segments`` entry is ``{"start": <time>, "end": <time>, "name": optional str}``.
    Times can be floats/ints (seconds) or ``HH:MM:SS(.ms)`` strings.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(input_path)
    if not segments:
        raise ValueError("segments must be a non-empty list")

    out_dir = Path(output_dir) if output_dir else get_settings().output_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        if "start" not in seg or "end" not in seg:
            raise ValueError(f"segment {idx} requires 'start' and 'end'")
        start = _format_time(seg["start"])
        end = _format_time(seg["end"])
        name = seg.get("name") or f"{src.stem}_clip{idx}"
        out = out_dir / f"{name}{src.suffix}"

        # `-ss/-to` placed *before* `-i` is fast (keyframe-based seek). With
        # `-c copy` cuts may snap to keyframes; re_encode=True for frame-accurate cuts.
        args = ["-ss", start, "-to", end, "-i", str(src)]
        if re_encode:
            args += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac"]
        else:
            args += ["-c", "copy"]
        args.append(str(out))
        run_ffmpeg(args)
        outputs.append(str(out))

    return {"clips": outputs, "count": len(outputs), "output_dir": str(out_dir)}


def concat_videos(
    video_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    *,
    re_encode: bool = True,
) -> dict[str, Any]:
    """Concatenate videos in order. Re-encodes by default for safe joining of mismatched inputs."""
    if not video_paths:
        raise ValueError("video_paths must be a non-empty list")
    paths = [Path(p) for p in video_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    out = _resolve_output(output_path, "merged.mp4")

    # ffmpeg concat demuxer needs an absolute-path list file.
    list_file = out.parent / f".concat_{out.stem}.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in paths),
        encoding="utf-8",
    )
    try:
        args = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
        if re_encode:
            args += [
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ]
        else:
            args += ["-c", "copy"]
        args.append(str(out))
        run_ffmpeg(args)
    finally:
        if list_file.exists():
            list_file.unlink()

    return {
        "output_path": str(out),
        "input_count": len(paths),
        "duration": probe_duration(out),
    }


def burn_subtitle(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path | None = None,
    *,
    font_name: str = "SimHei",
    font_size: int = 24,
    primary_color: str = "&HFFFFFF",
    outline: int = 1,
) -> dict[str, Any]:
    """Hard-burn an SRT subtitle file into a video using ffmpeg's ``subtitles`` filter."""
    src = Path(video_path)
    sub = Path(subtitle_path)
    if not src.exists():
        raise FileNotFoundError(video_path)
    if not sub.exists():
        raise FileNotFoundError(subtitle_path)

    out = _resolve_output(output_path, f"subtitled_{src.name}")

    style = (
        f"FontName={font_name},"
        f"FontSize={font_size},"
        f"Outline={outline},"
        f"PrimaryColour={primary_color}"
    )
    sub_arg = _escape_subtitle_path(sub)
    args = [
        "-i",
        str(src),
        "-vf",
        f"subtitles='{sub_arg}':force_style='{style}'",
        "-c:a",
        "copy",
        str(out),
    ]
    run_ffmpeg(args)
    return {"output_path": str(out), "duration": probe_duration(out)}


def mux_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path | None = None,
    *,
    mode: str = "replace",
    audio_volume: float = 1.0,
    shortest: bool = True,
) -> dict[str, Any]:
    """Combine an audio file with a video.

    Args:
        mode: ``"replace"`` swaps the existing audio for ``audio_path``;
              ``"mix"`` blends both tracks via amix.
        audio_volume: Volume multiplier applied to ``audio_path`` in mix mode
              (or to the new audio in replace mode).
        shortest: If True, output stops when the shorter input ends.
    """
    src = Path(video_path)
    aud = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(video_path)
    if not aud.exists():
        raise FileNotFoundError(audio_path)

    out = _resolve_output(output_path, f"dubbed_{src.stem}.mp4")

    if mode == "replace":
        args = ["-i", str(src), "-i", str(aud)]
        if audio_volume != 1.0:
            args += ["-filter:a:1", f"volume={audio_volume}"]
        args += [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
        if shortest:
            args.append("-shortest")
        args.append(str(out))
    elif mode == "mix":
        # Mix new audio with existing video's audio (if any).
        filter_complex = (
            f"[1:a]volume={audio_volume}[a1];"
            f"[0:a][a1]amix=inputs=2:duration={'shortest' if shortest else 'longest'}:dropout_transition=2[aout]"
        )
        args = [
            "-i",
            str(src),
            "-i",
            str(aud),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
        if shortest:
            args.append("-shortest")
        args.append(str(out))
    else:
        raise ValueError(f"mode must be 'replace' or 'mix', got '{mode}'")

    run_ffmpeg(args)
    return {"output_path": str(out), "duration": probe_duration(out), "mode": mode}
