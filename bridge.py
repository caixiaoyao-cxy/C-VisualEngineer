"""bridge.py — 输入地名，全自动出视频"""

import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1] / "map2video-main" / "map2video-main"
PARTA = Path(__file__).resolve().parents[1] / "map2video-Parta" / "map2video-Parta"
BRANCH_B = Path(__file__).resolve().parents[1] / "map2video-B" / "map2video-B"

for p in [BASE / "src", PARTA / "src", BRANCH_B / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CVE = Path(__file__).parent
if str(CVE) not in sys.path:
    sys.path.insert(0, str(CVE))

from config import OUTPUT_DIR, SCENES_DIR, ANIMATIONS_DIR, VIDEO_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def search_map_image(place: str) -> str | None:
    """多引擎搜地图图片"""
    engines = [
        ("Bing", f"https://www.bing.com/images/search?q={urllib.parse.quote(place + ' 地图')}&FORM=HDRSC2"),
        ("DuckDuckGo", f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(place + ' 地图')}"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for name, url in engines:
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            html = resp.read().decode("utf-8", errors="ignore")
            # 多种匹配模式
            patterns = [
                r'src="([^"]+\.(?:jpg|jpeg|png|webp))"',
                r'img[^>]+src="([^"]+)"',
                r'data-src="([^"]+)"',
            ]
            for pat in patterns:
                matches = re.findall(pat, html, re.IGNORECASE)
                for m in matches:
                    m = m.strip()
                    if m.startswith("http") and not m.endswith((".svg", ".gif")):
                        return m
        except:
            continue
    return None


def download_image(url: str, path: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        conn = urllib.request.urlopen(req, timeout=30)
        with open(path, "wb") as f:
            f.write(conn.read())
        return True
    except:
        return False


def run_full_pipeline(map_image_path: str):
    from mapgen.vision import analyze_map
    from mapgen.rag import build_culture_inventory
    from mapgen.storyboard.schema import (
        Storyboard, StoryboardScene, CultureElementRef,
        PromptItem, build_prompts_from_storyboard,
    )

    print("🗺️ 提取轮廓...", end=" ", flush=True)
    map_result = analyze_map(map_image_path, {
        "contour_options": {"output_dir": str(OUTPUT_DIR)},
    })
    places = map_result.get("places", [])
    print(f"✅ {[p['name'] for p in places]}")

    if not places:
        places = [{"name": Path(map_image_path).stem, "type_guess": "未知"}]

    print("📚 搜索文化元素...", end=" ", flush=True)
    try:
        culture = build_culture_inventory(places, options={
            "search_options": {"max_results": 3},
        })
        print(f"✅ {len(culture.get('inventory', []))} 条")
    except Exception as e:
        print(f"⚠️ {e}")
        culture = {"inventory": []}

    print("📝 生成分镜...", end=" ", flush=True)
    inventory = culture.get("inventory", [])
    artifacts = map_result.get("artifacts", {})
    mask_path = artifacts.get("mask_path", "")

    scenes = []
    for i, item in enumerate(inventory[:5]):
        element = CultureElementRef.from_inventory_item(item)
        desc = element.summary[:120] if element.summary else f"{element.place_name}的{element.element_name}"
        keywords = "、".join(element.visual_keywords[:3])
        scene = StoryboardScene(
            scene_id=i + 1,
            title=f"{element.place_name} - {element.element_name}",
            culture_element=element,
            shot_type="文化展示",
            duration_seconds=8.0,
            map_reference=mask_path,
            visual_description=f"{desc}。视觉元素：{keywords}。地图轮廓内构图，白底无背景，只显示人物和物件。",
            character_action="人物在画面中行走、展示文化元素",
            camera_movement="缓缓推进",
            transition="淡入淡出",
            narration=element.summary[:150] or f"欢迎体验{element.place_name}的{element.element_name}。",
            style_notes=["二次元", "手绘风", "童话色彩", "白底", "无背景"],
        )
        scenes.append(scene)

    if len(scenes) < 3:
        for place in places:
            if len(scenes) >= 5:
                break
            name = place.get("name", f"场景{len(scenes)+1}")
            element = CultureElementRef(place_name=name, element_name=f"{name}风貌",
                                        category="自然景观", summary=f"{name}的美丽风光")
            scenes.append(StoryboardScene(
                scene_id=len(scenes) + 1, title=f"{name}全景",
                culture_element=element,
                visual_description=f"{name}整体风貌，地标建筑与自然景观",
                character_action="镜头缓缓扫过全景",
                narration=f"这里就是{name}。",
                map_reference=mask_path,
            ))

    storyboard = Storyboard(
        project_title=f"{places[0].get('name', '')}文化宣传片",
        place_name=places[0].get("name", "") if places else "",
        tone="文化宣传", style="二维手绘童话风",
        scenes=scenes,
    )
    prompt_items = build_prompts_from_storyboard(storyboard, prompt_type="image")

    sb_path = OUTPUT_DIR / "storyboard.json"
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(storyboard.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"✅ {len(scenes)} 个场景")

    print("🎬 场景图...", end=" ", flush=True)
    c_scenes = [{
        "scene_id": item["scene_id"],
        "description": item["positive_prompt"],
        "action": "人物在地图轮廓内活动",
        "narration": "",
        "style": ", ".join(item.get("style_keywords", ["二次元", "手绘风"])),
        "contour_map": mask_path,
        "seed": 42 + i * 100,
        "variants": 1,
    } for i, item in enumerate(prompt_items)]
    c_sb_path = OUTPUT_DIR / "_c_input.json"
    with open(c_sb_path, "w", encoding="utf-8") as f:
        json.dump(c_scenes, f, ensure_ascii=False, indent=2)

    import scene_generator
    scene_generator.main([str(c_sb_path)])
    print("🎬 动画...", end=" ", flush=True)
    import animation_generator
    animation_generator.main([str(c_sb_path)])
    print("🎬 合成视频...")
    import video_composer
    video_composer.main([str(c_sb_path)])
    print(f"\n🎉 完成！视频: {(VIDEO_DIR / 'final_video.mp4').resolve()}")


def main():
    print("\n" + "=" * 50)
    print("  🗺️  AI 地图宣传视频生成器")
    print("=" * 50)
    load_env()

    place = input("\n📍 输入地名（如 杭州）: ").strip()
    if not place:
        print("❌ 地名不能为空"); sys.exit(1)

    print(f"🔍 搜索「{place}」地图...", end=" ", flush=True)
    map_url = search_map_image(place)
    if map_url:
        print("✅")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(OUTPUT_DIR))
        if download_image(map_url, tmp.name):
            print("⚡ 全自动生成中...")
            run_full_pipeline(tmp.name)
            return

    # 搜不到 → 本机已有测试图则自动用
    test_map = Path("input/contours/hangzhou_map.png")
    if test_map.exists():
        print(f"⚠️ 未搜到，使用本地测试图")
        run_full_pipeline(str(test_map.resolve()))
        return

    print(f"❌ 未找到「{place}」地图")
    image = input("📂 拖入地图图片: ").strip().strip('"').strip("'")
    if not Path(image).exists():
        print(f"❌ 文件不存在"); sys.exit(1)
    run_full_pipeline(image)


if __name__ == "__main__":
    main()
