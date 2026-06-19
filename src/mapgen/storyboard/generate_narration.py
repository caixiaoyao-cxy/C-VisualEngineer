from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import NarrationSegment, Storyboard

DEFAULT_STORYBOARD = "outputs/b/storyboard.json"
DEFAULT_OUTPUT_MD = "outputs/b/narration.md"
DEFAULT_OUTPUT_JSON = "outputs/b/narration.json"


def generate_narration(
    storyboard_path: str | Path = DEFAULT_STORYBOARD,
    output_md: str | Path = DEFAULT_OUTPUT_MD,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
) -> dict[str, str]:
    storyboard = load_storyboard(storyboard_path)
    segments = build_segments(storyboard)
    md_content = to_markdown(storyboard, segments)
    json_content = {
        "project_title": storyboard.project_title,
        "place_name": storyboard.place_name,
        "segments": [segment.to_dict() for segment in segments],
    }

    md_path = Path(output_md)
    json_path = Path(output_json)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown_path": str(md_path), "json_path": str(json_path)}


def load_storyboard(path: str | Path) -> Storyboard:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Storyboard.from_dict(data)


def build_segments(storyboard: Storyboard) -> list[NarrationSegment]:
    segments: list[NarrationSegment] = []
    current = 0.0
    for scene in storyboard.scenes:
        segment = NarrationSegment.from_scene(scene, start_seconds=current)
        segments.append(segment)
        current = segment.end_seconds
    return segments


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    minutes = total // 60
    sec = total % 60
    return f"{minutes:02d}:{sec:02d}"


def to_markdown(storyboard: Storyboard, segments: list[NarrationSegment]) -> str:
    lines = [
        f"# {storyboard.project_title} 解说词",
        "",
        f"- 地点：{storyboard.place_name}",
        f"- 风格：{storyboard.style}",
        f"- 总时长：约 {storyboard.total_duration_seconds} 秒",
        "",
    ]
    scene_by_id = {scene.scene_id: scene for scene in storyboard.scenes}
    for segment in segments:
        scene = scene_by_id.get(segment.scene_id)
        title = scene.title if scene else f"Scene {segment.scene_id}"
        lines.extend(
            [
                f"## Scene {segment.scene_id}: {title}",
                f"时间：{format_time(segment.start_seconds)} - {format_time(segment.end_seconds)}",
                "",
                segment.text.strip(),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate narration.md and narration.json for D.")
    parser.add_argument("--storyboard", default=DEFAULT_STORYBOARD, help="B 生成的 storyboard.json。")
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD, help="输出 narration.md 路径。")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="输出 narration.json 路径。")
    args = parser.parse_args()

    result = generate_narration(args.storyboard, args.output_md, args.output_json)
    print(f"narration markdown saved to: {result['markdown_path']}")
    print(f"narration json saved to: {result['json_path']}")


if __name__ == "__main__":
    main()
