from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mapgen.config import get_settings
from mapgen.llm import LLMConfigurationError, OpenAICompatibleClient
from mapgen.llm.openai_compatible import parse_json_object

from .schema import PromptItem, Storyboard, normalize_prompt_type

DEFAULT_STORYBOARD = "outputs/b/storyboard.json"
DEFAULT_OUTPUT = "outputs/b/prompts.json"


def generate_prompts(
    storyboard_path: str | Path = DEFAULT_STORYBOARD,
    output_path: str | Path = DEFAULT_OUTPUT,
    map_mask: str | None = None,
    map_outline: str | None = None,
    style_ref: str | None = None,
    prompt_type: str = "image",
    use_llm: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    storyboard = load_storyboard(storyboard_path)
    reference_files = [path for path in [map_mask, map_outline, style_ref] if path]
    ptype = normalize_prompt_type(prompt_type)

    if use_llm:
        try:
            prompts = generate_prompts_with_llm(storyboard, reference_files=reference_files, prompt_type=ptype, model=model)
        except (LLMConfigurationError, KeyError, ValueError, json.JSONDecodeError, RuntimeError, Exception):
            prompts = generate_prompts_fallback(storyboard, reference_files=reference_files, prompt_type=ptype)
    else:
        prompts = generate_prompts_fallback(storyboard, reference_files=reference_files, prompt_type=ptype)

    errors = []
    for item in prompts:
        errors.extend(item.validate())
    if errors:
        raise ValueError("Invalid prompts: " + "; ".join(errors))

    result = {
        "project_title": storyboard.project_title,
        "place_name": storyboard.place_name,
        "style": storyboard.style,
        "reference_files": reference_files,
        "prompts": [item.to_dict() for item in prompts],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(output), "prompts": result}


def load_storyboard(path: str | Path) -> Storyboard:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Storyboard.from_dict(data)


def generate_prompts_fallback(
    storyboard: Storyboard,
    reference_files: list[str],
    prompt_type: str = "image",
) -> list[PromptItem]:
    prompts: list[PromptItem] = []
    for scene in storyboard.scenes:
        element = scene.culture_element
        keywords = ", ".join(element.visual_keywords + scene.style_notes)
        animation_phrase = "short subtle animation, character interacting with cultural element, gentle camera push, " if prompt_type == "animation" else ""
        char_note = ", one consistent character across all scenes: a young Chinese woman, black shoulder-length hair with bangs, round face, big black eyes, wearing a red traditional Chinese dress (qipao) with gold trim, white sneakers"
        positive = (
            f"flat 2D hand-drawn map illustration{char_note}, {animation_phrase}"
            f"{storyboard.place_name} local culture map scene, "
            f"scene {scene.scene_id}: {scene.title}, {scene.visual_description}, "
            f"character action inside map: {scene.character_action}, "
            f"map reference: {scene.map_reference}, "
            f"cultural element: {element.element_name}, "
            f"visual keywords: {keywords}, "
            f"everything inside the map outline, same character face across all scenes, "
            f"small character (less than 8% of canvas), "
            f"clean flat composition, warm colors, storybook feeling, no readable text"
        )
        negative = (
            "photorealistic, 3d render, low quality, blurry, messy map, distorted map outline, "
            "content outside map boundary, character outside map, wrong landmark, "
            "unrelated culture, extra limbs, bad anatomy, unreadable text, watermark, logo, oil painting, frame border"
        )
        control_hint = (
            "The map outline must always be clearly visible as the frame boundary. "
            "ALL content must remain INSIDE the map outline. "
            "The character, buildings, and cultural objects must not extend beyond the map edge. "
            "Character should be actively engaging with the scene's cultural element inside the map area, "
            "doing natural actions like pointing, reading, walking, drinking tea, or admiring the view. "
            "Character must be drawn very small (less than 8% of canvas) and away from the map edges. "
            "Keep at least 5% margin from the map edge so nothing gets clipped."
        )
        prompts.append(
            PromptItem(
                scene_id=scene.scene_id,
                prompt_type=prompt_type,  # type: ignore[arg-type]
                positive_prompt=positive,
                negative_prompt=negative,
                control_hint=control_hint,
                style_keywords=["2D", "hand-drawn", "fairy-tale", "cultural map", "map outline"],
                reference_files=reference_files,
            )
        )
    return prompts


def generate_prompts_with_llm(
    storyboard: Storyboard,
    reference_files: list[str],
    prompt_type: str,
    model: str | None = None,
) -> list[PromptItem]:
    settings = get_settings()
    client = OpenAICompatibleClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    selected_model = model or settings.openai_text_model
    prompt = build_prompt_generation_prompt(storyboard, reference_files=reference_files, prompt_type=prompt_type)
    raw = client.chat(model=selected_model, messages=[{"role": "user", "content": prompt}], temperature=0.25)
    data = parse_json_object(raw)
    items = data.get("prompts", [])
    if not isinstance(items, list):
        raise ValueError("LLM output must contain prompts list.")
    return [prompt_item_from_dict(item, reference_files=reference_files) for item in items if isinstance(item, dict)]


def prompt_item_from_dict(data: dict[str, Any], reference_files: list[str]) -> PromptItem:
    return PromptItem(
        scene_id=int(data.get("scene_id", 0)),
        prompt_type=normalize_prompt_type(data.get("prompt_type", "image")),
        positive_prompt=str(data.get("positive_prompt", "")).strip(),
        negative_prompt=str(data.get("negative_prompt", "photorealistic, 3d render, low quality, blurry, watermark")).strip(),
        control_hint=str(data.get("control_hint", "Use map outline as ControlNet constraint.")).strip(),
        style_keywords=[str(x).strip() for x in data.get("style_keywords", []) if str(x).strip()],
        reference_files=[str(x).strip() for x in data.get("reference_files", reference_files) if str(x).strip()],
    )


def build_prompt_generation_prompt(storyboard: Storyboard, reference_files: list[str], prompt_type: str) -> str:
    return (
        "你是 ComfyUI/Flux/ControlNet 图像提示词工程师。请把分镜转换为英文图像生成 Prompt。\n"
        "只返回 JSON 对象，不要 Markdown。格式：\n"
        "{\"prompts\":[{\"scene_id\":1,\"prompt_type\":\"image/animation\","
        "\"positive_prompt\":\"英文正向提示词\",\"negative_prompt\":\"英文反向提示词\","
        "\"control_hint\":\"角色活动范围说明\",\"style_keywords\":[\"2D\",\"hand-drawn\"],"
        "\"reference_files\":[\"...\"]}]}\n"
         "要求：平面手绘地图风格、所有内容（人物/建筑/文化元素）全部在地图轮廓边界内活动；"
         "角色融入地图场景、与当地文化元素互动；"
         "同一角色在所有场景中出现：年轻中国女孩，黑色齐肩发+刘海，圆脸，大眼睛，穿红色旗袍+金色镶边+白色运动鞋；"
         "角色体形小（占画面不超过8%），不要靠近地图边缘；"
         "不要写实摄影风、角色不要跑出地图边界、"
         "不要超出地图范围。\n"
        f"prompt_type={prompt_type}\n"
        f"reference_files={json.dumps(reference_files, ensure_ascii=False)}\n"
        f"storyboard={json.dumps(storyboard.to_dict(), ensure_ascii=False)[:16000]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts.json for C.")
    parser.add_argument("--storyboard", default=DEFAULT_STORYBOARD, help="B 生成的 storyboard.json。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 prompts.json 路径。")
    parser.add_argument("--map-mask", default=None, help="A 生成的地图 mask 图路径。")
    parser.add_argument("--map-outline", default=None, help="A 生成的地图轮廓图路径。")
    parser.add_argument("--style-ref", default=None, help="风格参考图路径。")
    parser.add_argument("--prompt-type", default="image", choices=["image", "animation"], help="image 或 animation。")
    parser.add_argument("--use-llm", action="store_true", help="调用大模型优化 Prompt。")
    parser.add_argument("--model", default=None, help="文本模型名称；默认读取 .env。")
    args = parser.parse_args()

    result = generate_prompts(
        storyboard_path=args.storyboard,
        output_path=args.output,
        map_mask=args.map_mask,
        map_outline=args.map_outline,
        style_ref=args.style_ref,
        prompt_type=args.prompt_type,
        use_llm=args.use_llm,
        model=args.model,
    )
    print(f"prompts saved to: {result['path']}")


if __name__ == "__main__":
    main()
