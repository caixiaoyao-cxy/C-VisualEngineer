"""Pluggable TTS dubbing.

Two providers are supported, selected by the ``TTS_PROVIDER`` env var:

* ``openai`` (default) — any OpenAI-compatible TTS endpoint
  (``POST {OPENAI_BASE_URL}/audio/speech``). Works with OpenAI ``gpt-4o-mini-tts``,
  ``tts-1``, and most compatible gateways. No extra dependency.
* ``dashscope`` — Aliyun DashScope CosyVoice via the official ``dashscope`` SDK.
  Imported lazily so it's only required when actually selected.

Output formats: ``mp3`` (default), ``wav``, ``pcm`` (raw 16-bit little-endian
PCM at the model's native sample rate, written with extension ``.raw``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..config import get_settings
from .ffmpeg import probe_duration

_VALID_FORMATS = {"mp3", "wav", "pcm", "raw"}


def _normalize_format(fmt: str) -> tuple[str, str]:
    """Return (api_format, file_extension)."""
    fmt = fmt.lower().strip()
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"unsupported audio format '{fmt}'. Use one of {sorted(_VALID_FORMATS)}.")
    if fmt == "raw":
        return "pcm", "raw"
    if fmt == "pcm":
        return "pcm", "raw"
    return fmt, fmt


def _resolve_output(path: str | Path | None, default_stem: str, ext: str) -> Path:
    if path is None:
        out_dir = get_settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{default_stem}.{ext}"
    p = Path(path)
    # If user passed a path without the right extension, replace the suffix.
    if p.suffix.lower().lstrip(".") not in {ext, "mp3", "wav", "raw", "pcm"}:
        p = p.with_suffix(f".{ext}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------- OpenAI-compatible ----------


def _synthesize_openai(
    text: str,
    output_path: Path,
    *,
    api_format: str,
    voice: str | None,
    model: str | None,
    speed: float | None,
    timeout: float,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for OpenAI-compatible TTS. "
            "Set it in .env or switch TTS_PROVIDER."
        )
    payload: dict[str, Any] = {
        "model": model or settings.openai_tts_model,
        "voice": voice or settings.openai_tts_voice,
        "input": text,
        "response_format": api_format,
    }
    if speed is not None:
        payload["speed"] = float(speed)
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.openai_base_url.rstrip('/')}/audio/speech"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenAI TTS failed ({resp.status_code}): {resp.text[:500]}"
            )
        output_path.write_bytes(resp.content)
    return {"provider": "openai", "model": payload["model"], "voice": payload["voice"]}


# ---------- DashScope CosyVoice ----------


def _synthesize_dashscope(
    text: str,
    output_path: Path,
    *,
    api_format: str,
    voice: str | None,
    model: str | None,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is required for DashScope TTS. "
            "Set it in .env or switch TTS_PROVIDER=openai."
        )
    try:
        import dashscope  # type: ignore
        from dashscope.audio.tts_v2 import SpeechSynthesizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "dashscope SDK is not installed. Run `pip install dashscope` to use TTS_PROVIDER=dashscope."
        ) from exc

    dashscope.api_key = settings.dashscope_api_key
    synthesizer = SpeechSynthesizer(
        model=model or settings.dashscope_tts_model,
        voice=voice or settings.dashscope_tts_voice,
    )
    audio = synthesizer.call(text)
    if audio is None:
        raise RuntimeError("DashScope synthesizer returned no audio.")
    # DashScope CosyVoice returns mp3 bytes by default. If the user asked for
    # wav/pcm we transcode through ffmpeg.
    if api_format == "mp3":
        output_path.write_bytes(audio)
    else:
        # write temp mp3, transcode
        tmp_mp3 = output_path.with_suffix(".tmp.mp3")
        tmp_mp3.write_bytes(audio)
        try:
            from .ffmpeg import run_ffmpeg

            args = ["-i", str(tmp_mp3)]
            if api_format == "wav":
                args += ["-acodec", "pcm_s16le", str(output_path)]
            else:  # pcm / raw
                args += [
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "22050",
                    "-ac",
                    "1",
                    str(output_path),
                ]
            run_ffmpeg(args)
        finally:
            if tmp_mp3.exists():
                tmp_mp3.unlink()
    return {
        "provider": "dashscope",
        "model": model or settings.dashscope_tts_model,
        "voice": voice or settings.dashscope_tts_voice,
    }


# ---------- public ----------


def synthesize_dubbing(
    text: str,
    output_path: str | Path | None = None,
    *,
    audio_format: str = "mp3",
    voice: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    speed: float | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Synthesize dubbing audio from text.

    Args:
        text: Script to read aloud.
        output_path: Destination file. Defaults to ``OUTPUT_DIR/dubbing.<ext>``.
        audio_format: ``mp3`` | ``wav`` | ``pcm``/``raw``.
        voice: Voice id / preset (provider-specific).
        model: TTS model id.
        provider: Override TTS_PROVIDER (``openai`` | ``dashscope``).
        speed: Optional speech rate multiplier (OpenAI only).
        timeout: HTTP timeout for OpenAI provider.

    Returns:
        ``{"audio_path": str, "duration": float, "format": str, "provider": str, ...}``.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")

    api_format, ext = _normalize_format(audio_format)
    out_path = _resolve_output(output_path, "dubbing", ext)

    chosen = (provider or get_settings().tts_provider or "openai").lower()
    if chosen == "openai":
        meta = _synthesize_openai(
            text,
            out_path,
            api_format=api_format,
            voice=voice,
            model=model,
            speed=speed,
            timeout=timeout,
        )
    elif chosen == "dashscope":
        meta = _synthesize_dashscope(
            text, out_path, api_format=api_format, voice=voice, model=model
        )
    else:
        raise ValueError(f"unknown TTS provider '{chosen}'. Use 'openai' or 'dashscope'.")

    # Probing duration for raw PCM is unreliable, so skip there.
    duration = 0.0
    if ext != "raw":
        try:
            duration = probe_duration(out_path)
        except Exception:
            duration = 0.0

    return {
        "audio_path": str(out_path),
        "format": ext,
        "duration": duration,
        "byte_size": out_path.stat().st_size if out_path.exists() else 0,
        **meta,
    }
