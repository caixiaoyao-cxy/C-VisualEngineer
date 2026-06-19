"""High-level pipelines combining TTS + subtitle generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import get_settings
from .subtitles import generate_subtitle
from .tts import synthesize_dubbing


def synthesize_with_subtitle(
    script: str,
    *,
    audio_output: str | Path | None = None,
    subtitle_output: str | Path | None = None,
    audio_format: str = "mp3",
    voice: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    speed: float | None = None,
    max_chars: int = 40,
    min_chars: int = 6,
    gap: float = 0.05,
) -> dict[str, Any]:
    """Generate a dubbing audio AND a matched .srt subtitle from one script.

    The subtitle is timed against the actual rendered audio duration so the
    cues line up with the dubbing.

    Returns combined metadata: ``{"audio": {...}, "subtitle": {...}}``.
    """
    if not script or not script.strip():
        raise ValueError("script is empty")

    settings = get_settings()
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default both outputs to a shared stem so they pair obviously.
    if audio_output is None:
        ext = "raw" if audio_format.lower() in {"raw", "pcm"} else audio_format.lower()
        audio_output = out_dir / f"dubbing.{ext}"
    if subtitle_output is None:
        subtitle_output = out_dir / "dubbing.srt"

    audio_meta = synthesize_dubbing(
        script,
        audio_output,
        audio_format=audio_format,
        voice=voice,
        model=model,
        provider=provider,
        speed=speed,
    )

    duration = audio_meta.get("duration") or None
    sub_meta = generate_subtitle(
        script,
        subtitle_output,
        audio_path=audio_meta["audio_path"] if duration else None,
        total_duration=duration,
        max_chars=max_chars,
        min_chars=min_chars,
        gap=gap,
    )
    return {"audio": audio_meta, "subtitle": sub_meta}
