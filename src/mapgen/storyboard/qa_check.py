from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import QualityIssue, QualityReport, Storyboard

DEFAULT_STORYBOARD = "outputs/b/storyboard.json"
DEFAULT_PROMPTS = "outputs/b/prompts.json"
DEFAULT_NARRATION = "outputs/b/narration.md"
DEFAULT_OUTPUT = "outputs/b/qa_report.md"

REQUIRED_STYLE_KEYWORDS = ["2D", "hand-drawn", "map"]
CHINESE_STYLE_KEYWORDS = ["二维", "手绘", "地图"]


def qa_check(
    storyboard_path: str | Path = DEFAULT_STORYBOARD,
    prompts_path: str | Path = DEFAULT_PROMPTS,
    narration_path: str | Path = DEFAULT_NARRATION,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    issues: list[QualityIssue] = []

    storyboard = load_storyboard(storyboard_path, issues)
    prompts_data = load_json_file(prompts_path, issues, file_label="prompts")
    narration_text = load_text_file(narration_path, issues, file_label="narration")

    if storyboard is not None:
        check_storyboard(storyboard, issues)
        check_prompt_alignment(storyboard, prompts_data, issues)
        check_narration_alignment(storyboard, narration_text, issues)

    passed = not any(issue.level == "error" for issue in issues)
    report = QualityReport(passed=passed, issues=issues)
    content = to_markdown(report)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return {"path": str(output), "passed": passed, "issues": [issue.to_dict() for issue in issues]}


def load_storyboard(path: str | Path, issues: list[QualityIssue]) -> Storyboard | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Storyboard.from_dict(data)
    except FileNotFoundError:
        issues.append(QualityIssue("error", None, f"storyboard file not found: {path}"))
    except Exception as exc:
        issues.append(QualityIssue("error", None, f"cannot load storyboard: {exc}"))
    return None


def load_json_file(path: str | Path, issues: list[QualityIssue], file_label: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        issues.append(QualityIssue("error", None, f"{file_label} file not found: {path}"))
    except Exception as exc:
        issues.append(QualityIssue("error", None, f"cannot load {file_label}: {exc}"))
    return {}


def load_text_file(path: str | Path, issues: list[QualityIssue], file_label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(QualityIssue("error", None, f"{file_label} file not found: {path}"))
    except Exception as exc:
        issues.append(QualityIssue("error", None, f"cannot load {file_label}: {exc}"))
    return ""


def check_storyboard(storyboard: Storyboard, issues: list[QualityIssue]) -> None:
    for error in storyboard.validate():
        issues.append(QualityIssue("error", None, error))

    scene_count = len(storyboard.scenes)
    if scene_count < 3:
        issues.append(QualityIssue("warning", None, f"only {scene_count} scenes; recommended 3-5 scenes."))
    if scene_count > 8:
        issues.append(QualityIssue("warning", None, f"{scene_count} scenes may be too many for a 30-60 second demo."))

    for scene in storyboard.scenes:
        text = " ".join(
            [
                scene.title,
                scene.visual_description,
                scene.character_action,
                scene.map_reference,
                scene.narration,
                " ".join(scene.style_notes),
            ]
        )
        if not contains_any(text, ["地图", "轮廓", "路线", "点位", "map"]):
            issues.append(QualityIssue("warning", scene.scene_id, "scene does not clearly mention map outline / route / point."))
        if not contains_any(scene.character_action, ["走", "移动", "停", "指", "看", "抬", "挥", "旋转", "出现"]):
            issues.append(QualityIssue("warning", scene.scene_id, "character_action may be too static for animation."))
        if len(scene.narration) > 90:
            issues.append(QualityIssue("warning", scene.scene_id, "narration is long; TTS may exceed scene duration."))
        if scene.duration_seconds < 3:
            issues.append(QualityIssue("warning", scene.scene_id, "duration is very short."))
        if not scene.culture_element.sources:
            issues.append(QualityIssue("info", scene.scene_id, "culture element has no source; acceptable for demo fallback, but A should provide sources."))


def check_prompt_alignment(storyboard: Storyboard, prompts_data: dict[str, Any], issues: list[QualityIssue]) -> None:
    prompts = prompts_data.get("prompts", [])
    if not isinstance(prompts, list):
        issues.append(QualityIssue("error", None, "prompts.json must contain a prompts list."))
        return

    prompt_by_scene = {}
    for item in prompts:
        if not isinstance(item, dict):
            continue
        scene_id = int(item.get("scene_id", 0) or 0)
        prompt_by_scene[scene_id] = item

    scene_ids = {scene.scene_id for scene in storyboard.scenes}
    prompt_ids = set(prompt_by_scene.keys())
    missing = scene_ids - prompt_ids
    extra = prompt_ids - scene_ids
    for scene_id in sorted(missing):
        issues.append(QualityIssue("error", scene_id, "missing prompt for this scene."))
    for scene_id in sorted(extra):
        issues.append(QualityIssue("warning", scene_id, "prompt exists but storyboard has no matching scene."))

    for scene in storyboard.scenes:
        item = prompt_by_scene.get(scene.scene_id)
        if not item:
            continue
        positive = str(item.get("positive_prompt", ""))
        negative = str(item.get("negative_prompt", ""))
        control_hint = str(item.get("control_hint", ""))
        joined = " ".join([positive, control_hint])

        if not positive:
            issues.append(QualityIssue("error", scene.scene_id, "positive_prompt is empty."))
        if not negative:
            issues.append(QualityIssue("warning", scene.scene_id, "negative_prompt is empty."))
        if not contains_any(joined, REQUIRED_STYLE_KEYWORDS + CHINESE_STYLE_KEYWORDS):
            issues.append(QualityIssue("warning", scene.scene_id, "prompt lacks required style/map keywords."))
        element_name = scene.culture_element.element_name
        if element_name and element_name not in positive and element_name.lower() not in positive.lower():
            issues.append(QualityIssue("warning", scene.scene_id, f"prompt may not mention culture element: {element_name}"))
        if not contains_any(control_hint, ["ControlNet", "mask", "outline", "map", "地图", "轮廓"]):
            issues.append(QualityIssue("warning", scene.scene_id, "control_hint does not clearly mention map/mask/ControlNet."))


def check_narration_alignment(storyboard: Storyboard, narration_text: str, issues: list[QualityIssue]) -> None:
    if not narration_text.strip():
        issues.append(QualityIssue("error", None, "narration.md is empty."))
        return

    for scene in storyboard.scenes:
        if f"Scene {scene.scene_id}" not in narration_text:
            issues.append(QualityIssue("warning", scene.scene_id, "narration.md does not contain this scene heading."))
        text = scene.narration.strip()
        if text and text not in narration_text:
            issues.append(QualityIssue("warning", scene.scene_id, "scene narration is not found exactly in narration.md."))
        element_name = scene.culture_element.element_name
        if element_name and element_name not in narration_text:
            issues.append(QualityIssue("info", scene.scene_id, f"narration may not explicitly mention culture element: {element_name}"))


def contains_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def to_markdown(report: QualityReport) -> str:
    lines = ["# B 部分质检报告", "", f"总体结果：{'通过' if report.passed else '未通过'}", ""]
    if not report.issues:
        lines.append("未发现问题。")
        return "\n".join(lines) + "\n"

    grouped = {"error": [], "warning": [], "info": []}
    for issue in report.issues:
        grouped[issue.level].append(issue)

    titles = {"error": "必须修改", "warning": "建议修改", "info": "提示信息"}
    for level in ["error", "warning", "info"]:
        items = grouped[level]
        if not items:
            continue
        lines.extend([f"## {titles[level]}", ""])
        for issue in items:
            prefix = f"Scene {issue.scene_id}" if issue.scene_id is not None else "Global"
            lines.append(f"- [{prefix}] {issue.message}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check B outputs and generate qa_report.md.")
    parser.add_argument("--storyboard", default=DEFAULT_STORYBOARD, help="storyboard.json 路径。")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="prompts.json 路径。")
    parser.add_argument("--narration", default=DEFAULT_NARRATION, help="narration.md 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 qa_report.md 路径。")
    args = parser.parse_args()

    result = qa_check(args.storyboard, args.prompts, args.narration, args.output)
    print(f"qa report saved to: {result['path']}")
    print(f"passed: {result['passed']}")


if __name__ == "__main__":
    main()
