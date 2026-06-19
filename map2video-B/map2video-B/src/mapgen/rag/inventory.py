from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from mapgen.config import get_settings
from mapgen.llm import LLMConfigurationError, OpenAICompatibleClient

CATEGORIES = ["非遗", "历史人物", "建筑地标", "民俗节庆", "饮食", "自然景观", "产业符号"]


def build_culture_inventory(
    places: list[dict[str, Any]],
    search_results: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts = options or {}
    if search_results is None:
        from mapgen.rag.search import search_culture_elements

        search_results = search_culture_elements(places, opts.get("search_options"))

    settings = get_settings()
    model = opts.get("model") or settings.openai_text_model
    prompt = opts.get("prompt") or _inventory_prompt(places, search_results)

    try:
        client = OpenAICompatibleClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        response = client.summarize_culture_inventory(model=model, prompt=prompt)
        inventory = response.get("inventory", [])
        if not isinstance(inventory, list):
            inventory = []
    except LLMConfigurationError:
        inventory = _fallback_inventory(search_results)
        response = {"inventory": inventory, "fallback": "OPENAI_API_KEY missing; used search snippets directly."}

    normalized = [_normalize_inventory_item(item) for item in inventory if isinstance(item, dict)]
    return {"inventory": normalized, "raw_response": response}


def generate_report(inventory: dict[str, Any] | list[dict[str, Any]], format: str = "markdown") -> dict[str, Any]:
    items = inventory.get("inventory", []) if isinstance(inventory, dict) else inventory
    if format.lower() == "json":
        content = json.dumps({"inventory": items}, ensure_ascii=False, indent=2)
        extension = "json"
    elif format.lower() in {"markdown", "md"}:
        content = _to_markdown(items)
        extension = "md"
    else:
        raise ValueError("format must be 'markdown' or 'json'.")

    output_dir = get_settings().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"culture_inventory_{uuid4().hex[:10]}.{extension}"
    path.write_text(content, encoding="utf-8")
    return {"format": format, "content": content, "path": str(path.resolve())}


def _inventory_prompt(places: list[dict[str, Any]], search_results: dict[str, Any]) -> str:
    return (
        "你是文化地图设计研究助手。请基于联网搜索片段，为每个地名提取当地文化元素。"
        "只返回 JSON 对象，格式为 {\"inventory\":[{\"place_name\":\"...\","
        "\"element_name\":\"...\",\"category\":\"非遗/历史人物/建筑地标/民俗节庆/饮食/自然景观/产业符号\","
        "\"summary\":\"...\",\"visual_keywords\":[\"...\"],\"usage_suggestions\":[\"...\"],"
        "\"confidence\":0.0,\"sources\":[{\"title\":\"...\",\"url\":\"...\"}]}]}。"
        "不要编造来源；没有来源的元素不要输出。输入如下：\n"
        f"places={json.dumps(places, ensure_ascii=False)}\n"
        f"search_results={json.dumps(search_results, ensure_ascii=False)[:16000]}"
    )


def _fallback_inventory(search_results: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    seen: set[tuple[str, str]] = set()
    for result in search_results.get("results", []):
        place_name = str(result.get("place_name", "")).strip()
        source_by_url = {source.get("url", ""): source for source in result.get("sources", []) if isinstance(source, dict)}
        for snippet in result.get("snippets", []):
            title = str(snippet.get("title", "")).strip() or f"{place_name}文化元素"
            url = str(snippet.get("url", "")).strip()
            key = (place_name, title)
            if key in seen:
                continue
            seen.add(key)
            category = _guess_category(title, str(snippet.get("content", "")))
            items.append(
                {
                    "place_name": place_name,
                    "element_name": title,
                    "category": category,
                    "summary": str(snippet.get("content", "")).strip(),
                    "visual_keywords": _guess_visual_keywords(title, category),
                    "usage_suggestions": _guess_usage_suggestions(category),
                    "confidence": 0.35 if url else 0.25,
                    "sources": [source_by_url.get(url, {"title": title, "url": url})] if url else [],
                }
            )
    return items


def _guess_category(title: str, content: str) -> str:
    rules = [
        ("建筑地标", ["曹娥庙", "孝德园", "街区", "古镇", "老街", "遗址", "庙", "园"]),
        ("民俗节庆", ["庙会", "节", "民俗", "嘉年华", "文化和自然遗产日"]),
        ("非遗", ["非遗", "非物质文化遗产", "青瓷", "黄酒", "五香干", "翠茗", "蓝鳊"]),
        ("饮食", ["美食", "杨梅", "家宴", "茶", "干", "酒"]),
        ("自然景观", ["曹娥江", "石浪", "梯田", "生态园", "景观", "山", "江"]),
        ("历史人物", ["曹娥", "虞舜", "谢安", "祝英台", "名人", "东山再起"]),
    ]
    best_category = "产业符号"
    best_score = 0
    for category, keywords in rules:
        score = 0
        for keyword in keywords:
            if keyword in title:
                score += 3
            if keyword in content:
                score += 1
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def _guess_visual_keywords(title: str, category: str) -> list[str]:
    defaults = {
        "非遗": ["手作纹样", "传统器物", "地方色彩"],
        "历史人物": ["人物剪影", "故事场景", "碑刻文字"],
        "建筑地标": ["屋檐", "牌坊", "街巷肌理"],
        "民俗节庆": ["灯彩", "人群", "仪式道具"],
        "饮食": ["食材", "餐桌", "地方包装"],
        "自然景观": ["水系", "山形", "生态色块"],
        "产业符号": ["品牌符号", "产业图标", "地域标识"],
    }
    keywords = defaults.get(category, defaults["产业符号"])
    return [title[:12], *keywords]


def _guess_usage_suggestions(category: str) -> list[str]:
    defaults = {
        "非遗": ["作为地图图例纹样", "转化为边框或点位图标"],
        "历史人物": ["用于人物故事标注", "作为文化路线节点"],
        "建筑地标": ["用于地标插画", "作为重点 POI 图标"],
        "民俗节庆": ["用于活动时间轴", "作为节庆主题装饰"],
        "饮食": ["用于特产图标", "作为文创包装视觉元素"],
        "自然景观": ["用于底纹色块", "作为地貌水系图案"],
        "产业符号": ["用于产业标签", "作为区域品牌元素"],
    }
    return defaults.get(category, defaults["产业符号"])


def _normalize_inventory_item(item: dict[str, Any]) -> dict[str, Any]:
    confidence = item.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    category = str(item.get("category", "产业符号")).strip()
    if category not in CATEGORIES:
        category = "产业符号"
    return {
        "place_name": str(item.get("place_name", "")).strip(),
        "element_name": str(item.get("element_name", "")).strip(),
        "category": category,
        "summary": str(item.get("summary", "")).strip(),
        "visual_keywords": _string_list(item.get("visual_keywords", [])),
        "usage_suggestions": _string_list(item.get("usage_suggestions", [])),
        "confidence": max(0.0, min(1.0, confidence)),
        "sources": [source for source in item.get("sources", []) if isinstance(source, dict)],
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _to_markdown(items: list[dict[str, Any]]) -> str:
    lines = ["# 文化元素清单", ""]
    for item in items:
        lines.extend(
            [
                f"## {item.get('place_name', '')} - {item.get('element_name', '')}",
                f"- 类别：{item.get('category', '')}",
                f"- 摘要：{item.get('summary', '')}",
                f"- 视觉关键词：{', '.join(item.get('visual_keywords', [])) or '无'}",
                f"- 使用建议：{'; '.join(item.get('usage_suggestions', [])) or '无'}",
                f"- 置信度：{item.get('confidence', 0)}",
                "- 来源：",
            ]
        )
        sources = item.get("sources", [])
        if sources:
            for source in sources:
                lines.append(f"  - [{source.get('title', source.get('url', 'source'))}]({source.get('url', '')})")
        else:
            lines.append("  - 无")
        lines.append("")
    return "\n".join(lines)
