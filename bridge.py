"""
bridge.py — map2video ↔ C-VisualEngineer 全链路桥接

完整流程:
  地图图片 → map2video 提取轮廓 + 识别地名 → 补充文化元素
  → 自动生成 Storyboard → 生成 PromptItem(C的输入)
  → 场景图 → 动画 → 视频
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ==========================================================
# 路径配置：把三个分支都加入 import 路径
# ==========================================================
BASE = Path(__file__).resolve().parents[1] / "map2video-main" / "map2video-main"
PARTA = Path(__file__).resolve().parents[1] / "map2video-Parta" / "map2video-Parta"
BRANCH_B = Path(__file__).resolve().parents[1] / "map2video-B" / "map2video-B"

for p in [BASE / "src", PARTA / "src", BRANCH_B / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# C-VisualEngineer 自己
CVE = Path(__file__).parent
if str(CVE) not in sys.path:
    sys.path.insert(0, str(CVE))

import config as cve_config
from config import OUTPUT_DIR, SCENES_DIR, ANIMATIONS_DIR, VIDEO_DIR

# ==========================================================
# API Key 配置
# ==========================================================
def load_env(env_path: str = ".env"):
    p = Path(env_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
                k, v = line.split("=", 1)
                import os
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ==========================================================
# Step 1: map2video 提取轮廓 + 识别地名
# ==========================================================
def step1_extract_map(map_image: str) -> dict[str, Any]:
    from mapgen.vision import analyze_map
    print("\n[Step 1/4] map2video 轮廓提取 + 地名识别...")
    result = analyze_map(map_image, {
        "contour_options": {"output_dir": str(OUTPUT_DIR)},
    })
    print(f"  轮廓: {len(result.get('contours', []))} 个")
    print(f"  地名: {[p['name'] for p in result.get('places', [])]}")
    return result

# ==========================================================
# Step 2: map2video 搜索文化元素
# ==========================================================
def step2_culture_search(map_result: dict[str, Any]) -> dict[str, Any]:
    from mapgen.rag import build_culture_inventory
    print("\n[Step 2/4] 搜索文化元素...")
    places = map_result.get("places", [])
    if not places:
        print("  ⚠️ 没有识别到地名，使用占位数据")
        places = [{"name": "杭州", "type_guess": "市", "confidence": 0.5}]
    culture = build_culture_inventory(places, options={
        "search_options": {"max_results": 3},
    })
    print(f"  文化元素: {len(culture.get('inventory', []))} 条")
    return culture

# ==========================================================
# Step 3: 生成 Storyboard + PromptItem（B角色输出格式）
# ==========================================================
def step3_generate_storyboard(
    map_result: dict[str, Any],
    culture_result: dict[str, Any],
) -> list[dict[str, Any]]:
    from mapgen.storyboard.schema import (
        Storyboard, StoryboardScene, CultureElementRef,
        PromptItem, build_prompts_from_storyboard,
    )
    print("\n[Step 3/4] 生成分镜...")

    inventory = culture_result.get("inventory", [])
    places = map_result.get("places", [])
    artifacts = map_result.get("artifacts", {})
    mask_path = artifacts.get("mask_path", "")

    scenes = []
    for i, item in enumerate(inventory[:5]):
        element = CultureElementRef.from_inventory_item(item)
        title = f"{element.place_name} - {element.element_name}"
        desc = element.summary[:120] if element.summary else title
        keywords = "、".join(element.visual_keywords[:3])
        scene = StoryboardScene(
            scene_id=i + 1,
            title=title,
            culture_element=element,
            shot_type="文化展示",
            duration_seconds=8.0,
            map_reference=mask_path,
            visual_description=f"{desc}。视觉元素：{keywords}。地图轮廓内构图，白底无背景。",
            character_action="人物在画面中自然行走、展示周围文化元素",
            camera_movement="缓缓推进",
            transition="淡入淡出",
            narration=element.summary[:150] or f"欢迎来到{element.place_name}，体验{element.element_name}。",
            style_notes=["二次元", "手绘风", "童话色彩", "白底", "无背景"],
        )
        scenes.append(scene)

    # 如果 inventory 不够 5 个，用地名补充
    if len(scenes) < 3:
        for place in places:
            if len(scenes) >= 5:
                break
            name = place.get("name", f"场景{len(scenes)+1}")
            element = CultureElementRef(
                place_name=name, element_name=f"{name}风貌",
                category="自然景观", summary=f"{name}的美丽风光",
            )
            scene = StoryboardScene(
                scene_id=len(scenes) + 1, title=f"{name}全景",
                culture_element=element, visual_description=f"{name}整体风貌，地标建筑与自然景观",
                character_action="镜头缓缓扫过全景", narration=f"这里就是{name}。",
                map_reference=mask_path,
            )
            scenes.append(scene)

    storyboard = Storyboard(
        project_title=f"{places[0].get('name', '未知')}文化宣传片" if places else "文化宣传片",
        place_name=places[0].get("name", "") if places else "",
        tone="文化宣传",
        style="二维手绘童话风",
        scenes=scenes,
    )

    # 生成 PromptItem（C的输入）
    prompt_items = build_prompts_from_storyboard(storyboard, prompt_type="image")

    # 保存文件
    storyboard_path = OUTPUT_DIR / "storyboard.json"
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard.to_dict(), f, ensure_ascii=False, indent=2)

    prompts_path = OUTPUT_DIR / "prompt_items.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in prompt_items], f, ensure_ascii=False, indent=2)

    print(f"  分镜: {storyboard_path} ({len(scenes)} 个场景)")
    print(f"  Prompt: {prompts_path}")

    return [p.to_dict() for p in prompt_items]

# ==========================================================
# Step 4: C管线 — 场景图 → 动画 → 视频
# ==========================================================
def step4_run_c_pipeline(prompt_items: list[dict[str, Any]], mask_path: str):
    print("\n[Step 4/4] C-VisualEngineer...")

    # 把 PromptItem 转成 storyboard 格式喂给管线
    scenes = []
    for i, item in enumerate(prompt_items):
        scene = {
            "scene_id": item["scene_id"],
            "description": item["positive_prompt"],
            "action": "人物在地图轮廓内活动",
            "narration": "",
            "style": ", ".join(item.get("style_keywords", ["二次元", "手绘风"])),
            "contour_map": mask_path,
            "seed": 42 + i * 100,
            "variants": 1,
        }
        scenes.append(scene)

    sb_path = OUTPUT_DIR / "_c_pipeline_input.json"
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    # 调用各模块
    import scene_generator
    import animation_generator
    import video_composer

    scene_generator.main([str(sb_path)])
    animation_generator.main([str(sb_path)])
    video_composer.main([str(sb_path)])

    print("\n✅ 全链路完成！")
    print(f"   最终视频: {(VIDEO_DIR / 'final_video.mp4').resolve()}")

# ==========================================================
# 主入口
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="map2video → C-VisualEngineer 全链路")
    parser.add_argument("map_image", help="地图图片路径")
    parser.add_argument("--skip-culture", action="store_true", help="跳过文化搜索（无 API key 时使用）")
    args = parser.parse_args()

    if not Path(args.map_image).exists():
        print(f"❌ 图片不存在: {args.map_image}")
        sys.exit(1)

    load_env()

    # Step 1
    map_result = step1_extract_map(args.map_image)

    # Step 2
    if args.skip_culture:
        culture_result = {"inventory": []}
        print("\n[Step 2/4] 跳过文化搜索")
    else:
        culture_result = step2_culture_search(map_result)

    # Step 3
    prompt_items = step3_generate_storyboard(map_result, culture_result)

    # Step 4
    mask_path = map_result.get("artifacts", {}).get("mask_path", "")
    step4_run_c_pipeline(prompt_items, mask_path)


if __name__ == "__main__":
    main()
