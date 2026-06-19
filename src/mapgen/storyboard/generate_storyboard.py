from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mapgen.config import get_settings
from mapgen.llm import LLMConfigurationError, OpenAICompatibleClient
from mapgen.llm.openai_compatible import parse_json_object

from .schema import CultureElementRef, Storyboard, StoryboardScene

DEFAULT_STYLE = "二维手绘童话风"
DEFAULT_OUTPUT = "outputs/b/storyboard.json"


def generate_storyboard(
    place_name: str,
    project_title: str | None = None,
    inventory_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    num_scenes: int = 5,
    target_duration_seconds: float = 45.0,
    style: str = DEFAULT_STYLE,
    use_llm: bool = True,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate storyboard.json for B.

    Input priority:
    1. Read A's culture inventory JSON if inventory_path exists.
    2. Otherwise use a small local fallback inventory so the script can still run.
    3. If OPENAI_API_KEY is configured and use_llm=True, ask the LLM to write scenes.
       If the LLM is unavailable, fall back to deterministic template generation.
    """

    place_name = place_name.strip()
    if not place_name:
        raise ValueError("place_name cannot be empty.")

    project_title = project_title or f"{place_name}文化地图宣传动画"
    inventory_items = load_inventory_items(inventory_path, place_name=place_name)
    selected_items = select_inventory_items(inventory_items, num_scenes=num_scenes, place_name=place_name)

    if use_llm:
        try:
            storyboard = generate_storyboard_with_llm(
                place_name=place_name,
                project_title=project_title,
                inventory_items=selected_items,
                num_scenes=num_scenes,
                target_duration_seconds=target_duration_seconds,
                style=style,
                model=model,
            )
            if storyboard.validate():
                raise ValueError("LLM returned an incomplete storyboard.")
        except (LLMConfigurationError, KeyError, ValueError, json.JSONDecodeError, RuntimeError, Exception):
            storyboard = generate_storyboard_fallback(
                place_name=place_name,
                project_title=project_title,
                inventory_items=selected_items,
                num_scenes=num_scenes,
                target_duration_seconds=target_duration_seconds,
                style=style,
            )
    else:
        storyboard = generate_storyboard_fallback(
            place_name=place_name,
            project_title=project_title,
            inventory_items=selected_items,
            num_scenes=num_scenes,
            target_duration_seconds=target_duration_seconds,
            style=style,
        )

    errors = storyboard.validate()
    if errors:
        raise ValueError("Invalid storyboard: " + "; ".join(errors))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = storyboard.to_dict()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(output), "storyboard": result}


def load_inventory_items(inventory_path: str | Path | None, place_name: str) -> list[dict[str, Any]]:
    if inventory_path:
        path = Path(inventory_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                items = data.get("inventory", [])
            elif isinstance(data, list):
                items = data
            else:
                items = []
            return [item for item in items if isinstance(item, dict)]

    return fallback_inventory(place_name)


def fallback_inventory(place_name: str) -> list[dict[str, Any]]:
    """Minimal local data used only when A has not produced inventory yet."""

    return [
        {
            "place_name": place_name,
            "element_name": f"{place_name}地图轮廓",
            "category": "建筑地标",
            "summary": "以地图轮廓作为画面构图骨架，人物沿地图路线移动。",
            "visual_keywords": ["地图轮廓", "路线", "地标点位"],
            "usage_suggestions": ["用于开场建立空间", "作为人物移动路径"],
            "confidence": 0.3,
            "sources": [],
        },
        {
            "place_name": place_name,
            "element_name": "地方美食",
            "category": "饮食",
            "summary": "用代表性食物表现当地生活气息。",
            "visual_keywords": ["食物", "摊位", "烟火气"],
            "usage_suggestions": ["作为中段文化场景"],
            "confidence": 0.3,
            "sources": [],
        },
        {
            "place_name": place_name,
            "element_name": "民俗活动",
            "category": "民俗节庆",
            "summary": "通过节庆、人群、道具表现地方文化活动。",
            "visual_keywords": ["灯彩", "人群", "节庆道具"],
            "usage_suggestions": ["作为高潮场景"],
            "confidence": 0.3,
            "sources": [],
        },
        {
            "place_name": place_name,
            "element_name": "自然景观",
            "category": "自然景观",
            "summary": "用山水、河流、植被表现地方自然环境。",
            "visual_keywords": ["山水", "河流", "绿地"],
            "usage_suggestions": ["作为转场背景"],
            "confidence": 0.3,
            "sources": [],
        },
        {
            "place_name": place_name,
            "element_name": "城市记忆",
            "category": "产业符号",
            "summary": "用符号化元素收束地方印象。",
            "visual_keywords": ["城市符号", "纪念章", "文化图标"],
            "usage_suggestions": ["作为结尾总结"],
            "confidence": 0.3,
            "sources": [],
        },
    ]


def select_inventory_items(items: list[dict[str, Any]], num_scenes: int, place_name: str) -> list[dict[str, Any]]:
    if not items:
        return fallback_inventory(place_name)[:num_scenes]

    normalized = []
    for item in items:
        copy = dict(item)
        copy.setdefault("place_name", place_name)
        copy.setdefault("element_name", "文化元素")
        copy.setdefault("category", "产业符号")
        copy.setdefault("summary", "")
        copy.setdefault("visual_keywords", [])
        copy.setdefault("usage_suggestions", [])
        copy.setdefault("confidence", 0.5)
        copy.setdefault("sources", [])
        normalized.append(copy)

    normalized.sort(key=lambda x: float(x.get("confidence", 0.5) or 0.5), reverse=True)
    while len(normalized) < num_scenes:
        normalized.append(fallback_inventory(place_name)[len(normalized) % 5])
    return normalized[:num_scenes]


def generate_storyboard_with_llm(
    place_name: str,
    project_title: str,
    inventory_items: list[dict[str, Any]],
    num_scenes: int,
    target_duration_seconds: float,
    style: str,
    model: str | None = None,
) -> Storyboard:
    settings = get_settings()
    client = OpenAICompatibleClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    selected_model = model or settings.openai_text_model
    prompt = build_storyboard_prompt(
        place_name=place_name,
        project_title=project_title,
        inventory_items=inventory_items,
        num_scenes=num_scenes,
        target_duration_seconds=target_duration_seconds,
        style=style,
    )
    raw = client.chat(model=selected_model, messages=[{"role": "user", "content": prompt}], temperature=0.4)
    data = parse_json_object(raw)
    return Storyboard.from_dict(data)


def build_storyboard_prompt(
    place_name: str,
    project_title: str,
    inventory_items: list[dict[str, Any]],
    num_scenes: int,
    target_duration_seconds: float,
    style: str,
) -> str:
    return (
        "你是二维文化地图动画的分镜导演。请根据文化元素清单生成分镜。\n"
        "必须只返回 JSON 对象，不要输出 Markdown，不要解释。\n"
        "JSON 格式必须为：\n"
        "{\"project_title\":\"...\",\"place_name\":\"...\",\"tone\":\"文化宣传\","
        "\"style\":\"二维手绘童话风\",\"total_duration_seconds\":45,\"scenes\":["
        "{\"scene_id\":1,\"title\":\"...\",\"shot_type\":\"开场/过渡/文化展示/人物行动/结尾\","
        "\"duration_seconds\":8,\"map_reference\":\"地图上的位置或路线\","
        "\"visual_description\":\"画面描述，必须体现地图轮廓\","
        "\"character_action\":\"人物动作，必须能做成小动画\","
        "\"camera_movement\":\"镜头运动\",\"transition\":\"转场方式\","
        "\"narration\":\"中文解说词，1到2句\",\"style_notes\":[\"二维\",\"手绘\",\"童话风\"],"
        "\"culture_element\":{\"place_name\":\"...\",\"element_name\":\"...\",\"category\":\"...\","
        "\"summary\":\"...\",\"visual_keywords\":[\"...\"],\"usage_suggestions\":[\"...\"],"
        "\"confidence\":0.5,\"sources\":[{\"title\":\"...\",\"url\":\"...\"}]}}]}\n"
        f"项目标题：{project_title}\n"
        f"地名：{place_name}\n"
        f"分镜数量：{num_scenes}\n"
        f"总时长：{target_duration_seconds} 秒\n"
        f"统一风格：{style}\n"
         "要求：地图轮廓是构图框架；人物要在地图中移动；每个镜头体现一个文化元素；"
         "动作简单，例如走两步、停下、抬头、指向地标、镜头推进。\n"
         "每个场景的解说词必须有不同的句式开头，不能重复使用'在{place_name}，...'的固定句式。\n"
         "每条解说词限制在1句，12个中文字以内，确保3秒内能念完。\n"
        f"文化元素清单：{json.dumps(inventory_items, ensure_ascii=False)[:12000]}"
    )


def _clean_title(raw: str) -> str:
    raw = raw.strip().rstrip("。，.，")
    for sep in [" - ", " – ", " — ", " | ", " :: ", "：", "·"]:
        parts = raw.split(sep, 1)
        if len(parts) == 2 and len(parts[0]) <= 18:
            return parts[0].strip()
    import re
    m = re.match(r"^(.{2,18}?)[（(—\-]", raw)
    if m:
        return m.group(1).strip()
    return raw[:20].rstrip("，,。")


def generate_storyboard_fallback(
    place_name: str,
    project_title: str,
    inventory_items: list[dict[str, Any]],
    num_scenes: int,
    target_duration_seconds: float,
    style: str,
) -> Storyboard:
    duration = round(target_duration_seconds / max(num_scenes, 1), 2)
    scenes: list[StoryboardScene] = []
    shot_types = ["开场", "文化展示", "人物行动", "过渡", "结尾"]
    transitions = ["地图线条淡入", "镜头轻推", "沿路线滑动", "柔和叠化", "地图点位汇聚"]

    # 按类别分配多样化解说模板，确保不重复
    _narration_pool: dict[str, list[str]] = {
        "饮食": [
            f"来{place_name}，尝一口{{clean}}。",
            f"舌尖上的{place_name}，少不了{{clean}}。",
            f"闻香识{place_name}，先来一份{{clean}}。",
            f"老味道：{place_name}的{{clean}}。",
        ],
        "建筑地标": [
            f"抬头望去，{{clean}}已伫立百年。",
            f"走近{{clean}}，触摸历史的温度。",
            f"这座{{clean}}，见证了{place_name}的变迁。",
            f"穿过街巷，{{clean}}就在眼前。",
        ],
        "民俗节庆": [
            f"热闹的{{clean}}，人人都在欢笑。",
            f"锣鼓喧天，{place_name}的{{clean}}开始了。",
            f"跟着人群走进{{clean}}，感受最地道的{place_name}。",
            f"灯火璀璨，{{clean}}正热闹。",
        ],
        "自然景观": [
            f"深呼吸，{place_name}的{{clean}}让人心静。",
            f"站在{{clean}}前，满眼都是{place_name}的山水。",
            f"微风拂面，{{clean}}风光正好。",
            f"远离喧嚣，{place_name}的{{clean}}美得像画。",
        ],
    }
    _cat_defaults = [
        f"这是{place_name}独有的{{clean}}。",
            f"来到{place_name}，一定要看{{clean}}。",
        f"听当地人讲讲{{clean}}的故事。",
        f"用心感受，{place_name}的{{clean}}。",
        f"每一处{{clean}}，都是{place_name}的印记。",
    ]
    _used_templates: set[int] = set()

    def _pick_narration_tmpl(cat: str) -> str:
        pool = _narration_pool.get(cat, _cat_defaults)
        available = [i for i in range(len(pool)) if i not in _used_templates]
        if not available:
            available = list(range(len(pool)))
        idx = available[0]
        _used_templates.add(idx)
        return pool[idx]

    for index, item in enumerate(inventory_items[:num_scenes], start=1):
        element = CultureElementRef.from_inventory_item(item)
        keywords = "、".join(element.visual_keywords[:3]) or element.element_name
        clean = _clean_title(element.element_name)
        if not clean or len(clean) < 2:
            clean = element.element_name[:16]
        title = f"{place_name}·{clean}"
        shot_type = shot_types[(index - 1) % len(shot_types)]
        if index == 1:
            shot_type = "开场"
        elif index == num_scenes:
            shot_type = "结尾"
        cat = element.category
        # 角色动作融入场景，依据类别变化
        if index == 1:
            action = f"主角从地图边缘走入画面，沿着地图轮廓线走到{clean}点位前，停下，抬头看向点位标志，伸出手轻轻触碰。"
            visual = f"一张{style}的{place_name}地图在眼前展开，{clean}是第一个文化点位，在地图轮廓内发光，周围有{keywords}元素点缀，角色走入画面。"
            narration = f"第一站：{clean}，出发！"
        elif index == num_scenes:
            action = f"主角站在地图中央，回顾走过的所有点位，{clean}在身旁闪烁，轻轻挥手告别，所有文化元素缓缓旋转汇聚。"
            visual = f"一张{style}的{place_name}地图上，所有文化点位依次亮起，{clean}与{keywords}汇聚成完整的地方文化图景，角色站在地图中心。"
            narration = f"走到终点，{clean}在眼前闪耀。"
        else:
            tmpl = _pick_narration_tmpl(cat)
            narration = tmpl.replace("{clean}", clean)
            if cat == "饮食":
                action = f"主角走到{clean}摊位前，好奇地俯身观察桌上的{keywords}，伸手拿起一件仔细端详，露出惊喜的表情。"
                visual = f"一张{style}的{place_name}地图中，{clean}场景：{keywords}，摊位冒着热气，角色正在摊位前与食物互动。"
            elif cat == "建筑地标":
                action = f"主角站在{clean}前，仰头欣赏建筑细节，沿着建筑边缘缓步走动，手指轻轻划过墙面纹理。"
                visual = f"一张{style}的{place_name}地图中，{clean}场景：{keywords}，古建筑与街巷肌理清晰可见，角色在建筑前漫步。"
            elif cat == "民俗节庆":
                action = f"主角融入{clean}的人群中，跟着节拍轻轻摆动身体，好奇地看着周围的{keywords}，脸上带着开心的笑容。"
                visual = f"一张{style}的{place_name}地图中，{clean}场景：{keywords}，灯彩与人群热闹非凡，角色融入节日氛围。"
            elif cat == "自然景观":
                action = f"主角站在{clean}的观景位置，双手扶栏远眺，深呼吸感受自然，微风吹动发梢和衣角。"
                visual = f"一张{style}的{place_name}地图中，{clean}场景：{keywords}，山水与植被环绕，角色在自然中放松身心。"
            else:
                action = f"主角走到{clean}点位前，停下观察，伸手触摸文化元素，与之产生简单互动。"
                visual = f"一张{style}的{place_name}地图中，{clean}场景：{keywords}，文化元素在地图轮廓内展现，角色正在互动。"

        scenes.append(
            StoryboardScene(
                scene_id=index,
                title=title,
                culture_element=element,
                shot_type=shot_type,  # type: ignore[arg-type]
                duration_seconds=duration,
                map_reference=f"{place_name}地图第{index}个文化点位",
                visual_description=visual,
                character_action=action,
                camera_movement="轻微推进，保持地图轮廓清晰可见",
                transition=transitions[(index - 1) % len(transitions)],
                narration=narration,
                style_notes=["二维", "手绘", "童话风", "地图轮廓约束"],
            )
        )

    return Storyboard(
        project_title=project_title,
        place_name=place_name,
        tone="文化宣传",
        style=style,
        total_duration_seconds=target_duration_seconds,
        scenes=scenes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate storyboard.json for B.")
    parser.add_argument("--place-name", required=True, help="地方名称，如 杭州、上虞。")
    parser.add_argument("--project-title", default=None, help="项目标题。")
    parser.add_argument("--inventory", default=None, help="A 生成的文化元素清单 JSON 文件路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 storyboard.json 路径。")
    parser.add_argument("--num-scenes", type=int, default=5, help="分镜数量，默认 5。")
    parser.add_argument("--target-duration", type=float, default=45.0, help="总时长秒数，默认 45。")
    parser.add_argument("--style", default=DEFAULT_STYLE, help="统一画面风格。")
    parser.add_argument("--model", default=None, help="文本模型名称；默认读取 .env。")
    parser.add_argument("--no-llm", action="store_true", help="不调用大模型，直接用模板生成。")
    args = parser.parse_args()

    result = generate_storyboard(
        place_name=args.place_name,
        project_title=args.project_title,
        inventory_path=args.inventory,
        output_path=args.output,
        num_scenes=args.num_scenes,
        target_duration_seconds=args.target_duration,
        style=args.style,
        use_llm=not args.no_llm,
        model=args.model,
    )
    print(f"storyboard saved to: {result['path']}")


if __name__ == "__main__":
    main()
