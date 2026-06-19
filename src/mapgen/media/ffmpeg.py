"""Thin wrappers around ffmpeg/ffprobe binaries."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..config import get_settings


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg / ffprobe invocation fails."""

    def __init__(self, message: str, stderr: str = "", returncode: int | None = None):
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


def _resolve(binary: str) -> str:
    found = shutil.which(binary)
    return found or binary


def ffmpeg_bin() -> str:
    return _resolve(get_settings().ffmpeg_bin)


def ffprobe_bin() -> str:
    return _resolve(get_settings().ffprobe_bin)


def ensure_ffmpeg() -> None:
    """Raise FFmpegError if ffmpeg/ffprobe aren't usable on PATH."""
    settings = get_settings()
    for label, candidate in (("ffmpeg", settings.ffmpeg_bin), ("ffprobe", settings.ffprobe_bin)):
        resolved = shutil.which(candidate)
        if resolved is None and not Path(candidate).exists():
            raise FFmpegError(
                f"{label} not found. Install ffmpeg or set {label.upper()}_BIN in .env."
            )


def run_ffmpeg(args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the given args (after the binary)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"ffmpeg binary not found: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}",
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    return proc


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    """Return ffprobe JSON for a media file."""
    cmd = [
        ffprobe_bin(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"ffprobe binary not found: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed (exit {proc.returncode})",
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON: {exc}") from exc


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds (0.0 if unknown)."""
    info = ffprobe_json(path)
    fmt = info.get("format") or {}
    duration = fmt.get("duration")
    if duration is None:
        for stream in info.get("streams", []):
            if "duration" in stream:
                duration = stream["duration"]
                break
    try:
        return float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
