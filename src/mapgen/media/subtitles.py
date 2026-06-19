"""Script-to-SRT subtitle generation, parsing, and audio alignment."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import get_settings
from .ffmpeg import probe_duration

# Punctuation that ends a sentence/clause (Chinese + English).
_HARD_BREAKS = "。！？.!?\n"
_SOFT_BREAKS = "，、；：,;:"

# Defaults for "natural" segmentation.
_DEFAULT_MAX_CHARS = 40
_DEFAULT_MIN_CHARS = 6
# Approximate read speed when no audio is available, used for synthetic timing.
# Chinese-friendly default: ~5 characters/second.
_DEFAULT_CHARS_PER_SECOND = 5.0


@dataclass
class Cue:
    """One subtitle cue."""

    index: int
    start: float  # seconds
    end: float    # seconds
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


# ---------- segmentation ----------


def _split_sentences(script: str, max_chars: int, min_chars: int) -> list[str]:
    """Split script into subtitle-sized chunks.

    Strategy:
      1. Split on hard sentence terminators.
      2. If a chunk is longer than max_chars, sub-split on soft punctuation.
      3. Greedily merge tiny adjacent chunks until each is >= min_chars (when possible).
    """
    text = script.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 1) hard splits
    pieces: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _HARD_BREAKS:
            piece = "".join(buf).strip()
            if piece:
                pieces.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        pieces.append(tail)

    # 2) soft splits for over-long pieces
    refined: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            refined.append(piece)
            continue
        sub_buf: list[str] = []
        for ch in piece:
            sub_buf.append(ch)
            if ch in _SOFT_BREAKS and len("".join(sub_buf).strip()) >= min_chars:
                chunk = "".join(sub_buf).strip()
                if chunk:
                    refined.append(chunk)
                sub_buf = []
            elif len(sub_buf) >= max_chars:
                chunk = "".join(sub_buf).strip()
                if chunk:
                    refined.append(chunk)
                sub_buf = []
        tail = "".join(sub_buf).strip()
        if tail:
            refined.append(tail)

    # 3) merge dangling small fragments
    merged: list[str] = []
    for piece in refined:
        if merged and len(piece) < min_chars and len(merged[-1]) + len(piece) <= max_chars:
            merged[-1] = (merged[-1] + " " + piece).strip()
        else:
            merged.append(piece)
    return merged


def script_to_cues(
    script: str,
    *,
    total_duration: float | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
    chars_per_second: float = _DEFAULT_CHARS_PER_SECOND,
    gap: float = 0.05,
) -> list[Cue]:
    """Convert a raw script into timed cues.

    If ``total_duration`` is provided, cues are distributed across that span
    proportionally to chunk length. Otherwise timings come from ``chars_per_second``.
    """
    chunks = _split_sentences(script, max_chars=max_chars, min_chars=min_chars)
    if not chunks:
        return []

    if total_duration and total_duration > 0:
        weights = [max(len(c), 1) for c in chunks]
        total_weight = sum(weights)
        # Reserve gap*N for inter-cue spacing.
        speakable = max(total_duration - gap * (len(chunks) - 1), 0.1)
        cues: list[Cue] = []
        cursor = 0.0
        for idx, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
            dur = speakable * (weight / total_weight)
            start = cursor
            end = start + dur
            cues.append(Cue(index=idx, start=start, end=end, text=chunk))
            cursor = end + gap
        # snap last cue's end to total_duration
        if cues and total_duration > 0:
            cues[-1] = Cue(cues[-1].index, cues[-1].start, total_duration, cues[-1].text)
        return cues

    # Synthetic timing
    cues = []
    cursor = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        dur = max(len(chunk) / max(chars_per_second, 0.1), 0.8)
        start = cursor
        end = start + dur
        cues.append(Cue(index=idx, start=start, end=end, text=chunk))
        cursor = end + gap
    return cues


# ---------- SRT serialization ----------


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def cues_to_srt(cues: Iterable[Cue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        blocks.append(
            f"{cue.index}\n"
            f"{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n"
            f"{cue.text}\n"
        )
    return "\n".join(blocks).strip() + "\n"


def write_srt(cues: Iterable[Cue], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cues_to_srt(cues), encoding="utf-8")
    return path


# ---------- SRT parsing ----------

_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(path: str | Path) -> list[Cue]:
    """Parse an SRT file into Cue objects (tolerant of CRLF / .srt with `.` separator)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # Optional index line
        idx_line = lines[0].strip()
        if idx_line.isdigit() and len(lines) > 1:
            time_line = lines[1]
            text_lines = lines[2:]
            index = int(idx_line)
        else:
            time_line = lines[0]
            text_lines = lines[1:]
            index = len(cues) + 1
        m = _SRT_TIME_RE.search(time_line)
        if not m:
            continue
        start = _parse_timestamp(*m.group(1, 2, 3, 4))
        end = _parse_timestamp(*m.group(5, 6, 7, 8))
        cues.append(Cue(index=index, start=start, end=end, text="\n".join(text_lines).strip()))
    return cues


# ---------- High-level helpers used by the MCP server ----------


def _resolve_output(path: str | Path | None, default_name: str) -> Path:
    if path is None:
        out_dir = get_settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / default_name
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def generate_subtitle(
    script: str,
    output_path: str | Path | None = None,
    *,
    audio_path: str | Path | None = None,
    total_duration: float | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
    chars_per_second: float = _DEFAULT_CHARS_PER_SECOND,
    gap: float = 0.05,
) -> dict[str, Any]:
    """Generate a `.srt` subtitle file from a text script.

    Args:
        script: Source text. Any language; punctuation drives segmentation.
        output_path: Destination .srt file (default: OUTPUT_DIR/subtitle.srt).
        audio_path: If given, total_duration is read from this audio file via ffprobe.
        total_duration: Explicit total duration in seconds (overrides synthesised timing).
        max_chars / min_chars: Per-cue length bounds.
        chars_per_second: Read-speed for synthetic timing when no duration is known.
        gap: Inter-cue silence in seconds.
    """
    if not script or not script.strip():
        raise ValueError("script is empty")

    duration = total_duration
    if duration is None and audio_path is not None:
        duration = probe_duration(audio_path)
        if duration <= 0:
            duration = None

    cues = script_to_cues(
        script,
        total_duration=duration,
        max_chars=max_chars,
        min_chars=min_chars,
        chars_per_second=chars_per_second,
        gap=gap,
    )
    out = _resolve_output(output_path, "subtitle.srt")
    write_srt(cues, out)
    return {
        "subtitle_path": str(out),
        "cues": [c.to_dict() for c in cues],
        "duration": duration,
        "cue_count": len(cues),
    }


def align_subtitle_to_audio(
    subtitle_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path | None = None,
    *,
    gap: float = 0.05,
) -> dict[str, Any]:
    """Re-time existing SRT cues so the last cue ends at the audio duration.

    Useful when the dubbing audio was generated independently and the original
    subtitle timings drift.
    """
    cues = parse_srt(subtitle_path)
    if not cues:
        raise ValueError(f"no cues parsed from {subtitle_path}")
    duration = probe_duration(audio_path)
    if duration <= 0:
        raise ValueError(f"could not probe duration for audio {audio_path}")

    weights = [max(len(c.text), 1) for c in cues]
    total_weight = sum(weights)
    speakable = max(duration - gap * (len(cues) - 1), 0.1)

    new_cues: list[Cue] = []
    cursor = 0.0
    for cue, weight in zip(cues, weights):
        dur = speakable * (weight / total_weight)
        start = cursor
        end = start + dur
        new_cues.append(Cue(index=cue.index, start=start, end=end, text=cue.text))
        cursor = end + gap
    if new_cues:
        last = new_cues[-1]
        new_cues[-1] = Cue(last.index, last.start, duration, last.text)

    out = _resolve_output(output_path, Path(subtitle_path).stem + ".aligned.srt")
    write_srt(new_cues, out)
    return {
        "subtitle_path": str(out),
        "cues": [c.to_dict() for c in new_cues],
        "duration": duration,
        "cue_count": len(new_cues),
    }
