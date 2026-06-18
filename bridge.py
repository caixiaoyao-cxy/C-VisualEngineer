import argparse
import io
import json
import math
import random
import shutil
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from config import *

PLACEHOLDER_PLACE = "杭州"

FALLBACK_CULTURE_ITEMS = [
    {"element_name": "地图轮廓", "category": "建筑地标", "summary": "以地图轮廓作为画面构图骨架", "visual_keywords": ["地图轮廓", "路线", "地标点位"]},
    {"element_name": "地方美食", "category": "饮食", "summary": "用代表性食物表现当地生活气息", "visual_keywords": ["食物", "摊位", "烟火气"]},
    {"element_name": "民俗活动", "category": "民俗节庆", "summary": "通过节庆表现地方文化活动", "visual_keywords": ["灯彩", "人群", "节庆道具"]},
    {"element_name": "自然景观", "category": "自然景观", "summary": "山水河流表现地方自然环境", "visual_keywords": ["山水", "河流", "绿地"]},
    {"element_name": "城市记忆", "category": "产业符号", "summary": "符号化元素收束地方印象", "visual_keywords": ["城市符号", "纪念章", "文化图标"]},
]

FALLBACK_STORYBOARD_TEMPLATE = [
    {"shot_type": "开场", "transition": "地图线条淡入", "camera": "缓慢展开，建立全貌"},
    {"shot_type": "文化展示", "transition": "镜头轻推", "camera": "轻微推进"},
    {"shot_type": "人物行动", "transition": "沿路线滑动", "camera": "跟拍人物移动"},
    {"shot_type": "过渡", "transition": "柔和叠化", "camera": "缓缓平移"},
    {"shot_type": "结尾", "transition": "地图点位汇聚", "camera": "缓慢拉起，俯瞰全景"},
]

def log(msg: str):
    print(f"[bridge] {msg}")

def fetch_osm_map(place_name: str, output_path: str) -> bool:
    try:
        import httpx

        headers = {"User-Agent": "C-VisualEngineer/1.0"}
        query = f"{place_name} 行政区划图"
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&limit=1"
        resp = httpx.get(url, headers=headers, timeout=10)
        data = resp.json()
        if not data:
            log(f"OSM 未找到: {place_name}")
            return False

        lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
        zoom = 12
        size = 1024
        tile_size = 256
        n_tiles = math.ceil(size / tile_size)

        n = 2.0 ** zoom
        cx = int((lon + 180.0) / 360.0 * n)
        cy = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)

        img = Image.new("RGB", (size, size), (255, 255, 255))
        tile_servers = ["a.tile.openstreetmap.org", "b.tile.openstreetmap.org", "c.tile.openstreetmap.org"]
        loaded = 0
        for dx in range(-n_tiles // 2, n_tiles // 2 + 1):
            for dy in range(-n_tiles // 2, n_tiles // 2 + 1):
                tx, ty = cx + dx, cy + dy
                for ts in tile_servers:
                    try:
                        tile_url = f"https://{ts}/{zoom}/{tx}/{ty}.png"
                        tile_resp = httpx.get(tile_url, headers=headers, timeout=10)
                        if tile_resp.status_code == 200:
                            tile = Image.open(io.BytesIO(tile_resp.content))
                            px = (dx + n_tiles // 2) * tile_size
                            py = (dy + n_tiles // 2) * tile_size
                            img.paste(tile, (px, py))
                            loaded += 1
                            break
                    except Exception:
                        continue

        log(f"已加载 {loaded} 个地图瓦片")
        if loaded == 0:
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        log(f"OSM 地图已保存: {output_path}")
        return True
    except Exception as e:
        log(f"OSM 获取失败: {e}")
        return False

def extract_contour_from_map(map_path: str, output_mask: str) -> dict[str, Any]:
    img = cv2.imread(map_path)
    if img is None:
        raise ValueError(f"无法读取地图: {map_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_found = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_found[0] if len(contours_found) == 2 else contours_found[1]

    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [biggest], -1, 255, thickness=cv2.FILLED)

    # 居中缩放：把轮廓缩小到屏幕中央，不占满
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cx2, cy2 = int(xs.mean()), int(ys.mean())
        bw2, bh2 = int(xs.max() - xs.min()) + 40, int(ys.max() - ys.min()) + 40
        scale = min(w * CONTOUR_SCALE_RATIO / bw2, h * CONTOUR_SCALE_RATIO / bh2)
        new_w, new_h = int(bw2 * scale), int(bh2 * scale)
        ox2, oy2 = (w - new_w) // 2, (h - new_h) // 2
        centered = np.zeros_like(mask)
        for y, x in zip(ys, xs):
            nx = int((x - (cx2 - bw2//2)) * scale) + ox2
            ny = int((y - (cy2 - bh2//2)) * scale) + oy2
            if 0 <= nx < w and 0 <= ny < h:
                centered[ny, nx] = 255
        mask = centered

    Path(output_mask).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_mask, mask)
    log(f"轮廓掩码已保存: {output_mask} ({w}x{h})，已居中缩放至 {CONTOUR_SCALE_RATIO*100:.0f}%")

    return {"width": w, "height": h, "mask_path": output_mask, "contour_count": len(contours)}

def _ddgs_search(queries: list[str], max_results: int = 15) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        for query in queries:
            try:
                results = []
                with DDGS(timeout=20) as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        title = r.get("title", "")
                        snippet = r.get("body", "")
                        if title:
                            results.append({"title": title, "snippet": snippet})
                if results:
                    log(f"  DDGS 找到 {len(results)} 条: {query[:30]}")
                    return results
            except Exception as e:
                log(f"  DDGS 失败 ({query[:20]}): {e}")
                continue
    except ImportError:
        pass
    return []

def _wikipedia_search(query: str, max_results: int = 15) -> list[dict]:
    try:
        import requests as req
    except ImportError:
        try:
            import httpx as req
            req.get = req.get
        except ImportError:
            return []

    urls = [
        f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit={max_results}&utf8=1",
        f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit={max_results}&utf8=1",
    ]
    for url in urls:
        try:
            resp = req.get(url, headers={"User-Agent": "C-VisualEngineer/1.0"}, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            results = []
            for item in data.get("query", {}).get("search", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                })
            if results:
                return results
        except Exception as e:
            log(f"  Wikipedia 请求失败: {e}")
    return []

def search_culture(place_name: str) -> list[dict]:
    results = _ddgs_search([
        f"{place_name} tourism landmark cuisine",
        f"{place_name} 文化 地标",
    ])
    if not results:
        log("  DDGS 无结果，尝试 Wikipedia...")
        results = _wikipedia_search(f"{place_name}旅游景点美食")
        if not results:
            results = _wikipedia_search(f"{place_name}")
    if results:
        log(f"搜索到 {len(results)} 条文化结果")
    return results

def wiki_to_culture_items(place_name: str, wiki_results: list[dict]) -> list[dict]:
    items = []
    category_keywords = {
        "建筑地标": ["塔", "桥", "寺", "庙", "楼", "阁", "园", "故居", "遗址", "古镇", "城墙", "宫殿", "祠"],
        "饮食": ["美食", "小吃", "菜", "酒", "茶", "宴", "糕", "饼", "食"],
        "非遗": ["非遗", "传承", "戏曲", "工艺", "刺绣", "陶瓷", "剪纸", "雕刻", "民歌", "舞蹈"],
        "历史人物": ["故居", "墓", "纪念馆", "人物", "名人", "将军", "诗人", "文学家"],
        "民俗节庆": ["节", "庙会", "灯会", "龙舟", "民俗", "祭祀", "庆典"],
        "自然景观": ["山", "湖", "河", "江", "海", "峰", "瀑布", "森林", "公园", "湿地"],
    }
    for r in wiki_results:
        title = r["title"]
        snippet = r["snippet"]
        category = "产业符号"
        for cat, keywords in category_keywords.items():
            if any(k in title or k in snippet for k in keywords):
                category = cat
                break
        visual_keywords = {
            "建筑地标": ["地标建筑", "屋顶", "石阶", "门楼"],
            "饮食": ["食材", "餐桌", "热气", "摊位"],
            "非遗": ["手工艺", "纹样", "传统器物"],
            "历史人物": ["人物剪影", "故事场景", "古迹"],
            "民俗节庆": ["灯彩", "人群", "仪式"],
            "自然景观": ["山水", "植物", "天空"],
            "产业符号": ["符号", "图标", "标识"],
        }.get(category, ["文化元素"])
        items.append({
            "place_name": place_name,
            "element_name": title,
            "category": category,
            "summary": snippet[:200],
            "visual_keywords": visual_keywords,
            "usage_suggestions": [f"{title}作为地标展示"],
            "confidence": 0.6,
            "sources": [],
        })
    return items

def rank_culture_items(items: list[dict]) -> list[dict]:
    """按标志性程度排序：匹配多类别关键词的排前面，泛泛的排后面"""
    scored = []
    for item in items:
        score = 0
        title = item.get("element_name", "")
        summary = item.get("summary", "")
        category = item.get("category", "")
        # 有明确分类 +1
        if category != "产业符号":
            score += 1
        # 标题短且具体加分（不含百科、维基等通用词）
        generic_words = ["百科", "维基", "门户", "攻略", "官网", "首页", "旅游"]
        if not any(g in title for g in generic_words):
            score += 2
        # 标题长度适中（5-15 字）
        if 5 <= len(title) <= 15:
            score += 1
        # 摘要中有具体描述
        if len(summary) > 50:
            score += 1
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]

def interactive_select_items(all_items: list[dict], needed: int = 5) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  共搜索到 {len(all_items)} 个文化元素")
    print(f"  需要从中选择 {needed} 个用于分镜")
    print(f"{'='*60}")

    for i, item in enumerate(all_items):
        cat = item.get("category", "其他")
        name = item.get("element_name", "?")
        summary = item.get("summary", "")[:80]
        print(f"  [{i+1}] [{cat}] {name}")
        if summary:
            print(f"      {summary}")

    print(f"\n  选择方式：")
    print(f"    1) 输入序号用逗号分隔（如 1,3,5,7,9）")
    print(f"    2) 输入 random 随机抽取 {needed} 个")
    print(f"    3) 输入 theme:分类名 按主题筛选（如 theme:建筑地标）")
    print(f"    4) 直接回车 = 用排序后的前 {needed} 个")

    choice = input("\n  请选择: ").strip()

    if not choice:
        return rank_culture_items(all_items)[:needed]

    if choice.lower() == "random":
        selected = random.sample(all_items, min(needed, len(all_items)))
        return selected

    if choice.lower().startswith("theme:"):
        theme = choice[6:].strip()
        filtered = [item for item in all_items if theme.lower() in item.get("category", "").lower()]
        if not filtered:
            print(f"  未找到主题「{theme}」，使用全部结果")
            filtered = all_items
        if len(filtered) > needed:
            filtered = rank_culture_items(filtered)[:needed]
        return filtered

    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        selected = []
        seen = set()
        for idx in indices:
            if 1 <= idx <= len(all_items) and idx not in seen:
                selected.append(all_items[idx - 1])
                seen.add(idx)
        if not selected:
            print("  无效输入，使用前 5 个")
            return rank_culture_items(all_items)[:needed]
        return selected[:needed]
    except ValueError:
        print("  输入无法识别，使用前 5 个")
        return rank_culture_items(all_items)[:needed]

CATEGORY_VISUALS = {
    "建筑地标": ["traditional architecture", "landmark", "stone bridge", "pagoda", "temple", "historical building"],
    "饮食": ["street food", "local cuisine", "fresh ingredients", "market stall", "cooking steam", "fruit basket", "bread"],
    "非遗": ["traditional craft", "handmade art", "artisan workshop", "weaving", "pottery", "embroidery"],
    "历史人物": ["historical figure", "ancient story", "traditional costume", "calligraphy scroll", "ink brush"],
    "民俗节庆": ["festival celebration", "folk performance", "lanterns", "dragon dance", "paper cut", "decorations"],
    "自然景观": ["natural scenery", "mountains", "river", "green landscape", "wildflowers", "animals", "birds"],
    "产业符号": ["cultural symbol", "local character", "community life", "iconic pattern", "farm tools", "livestock"],
}

def generate_scene_prompt(place_name: str, item: dict) -> str:
    category = item.get("category", "产业符号")
    element = item.get("element_name", "")
    visuals = CATEGORY_VISUALS.get(category, CATEGORY_VISUALS["产业符号"])
    scenic = ", ".join(visuals[:4])
    return f"{place_name} scenery, {scenic}, {element[:20]}, character placeholder at center, surrounded by local objects, food, tools, animals radiating outward"

def generate_storyboard_scenes(place_name: str, contour_map_path: str, culture_items: list[dict] | None = None, num_scenes: int = 5) -> list[dict]:
    if culture_items:
        items = culture_items[:num_scenes]
        while len(items) < num_scenes:
            items.append(FALLBACK_CULTURE_ITEMS[len(items) % len(FALLBACK_CULTURE_ITEMS)])
    else:
        items = [FALLBACK_CULTURE_ITEMS[i % len(FALLBACK_CULTURE_ITEMS)] for i in range(num_scenes)]

    scenes = []
    for i in range(num_scenes):
        item = items[i]
        template = FALLBACK_STORYBOARD_TEMPLATE[i % len(FALLBACK_STORYBOARD_TEMPLATE)]
        keywords = "、".join(item["visual_keywords"][:3])

        if i == 0:
            desc = f"{place_name}地图缓缓展开，{item['element_name']}在轮廓内作为第一个文化点位亮起，{keywords}"
            action = "主角从地图边缘走入，沿着轮廓线走两步后停下，抬头看向发光的文化点位"
            narration = f"沿着地图轮廓出发，我们走进{place_name}的地方文化故事。"
        elif i == num_scenes - 1:
            desc = f"地图上的文化点位依次亮起，{item['element_name']}与{keywords}汇聚成完整的地方文化图案"
            action = "主角回到地图中央，轻轻挥手，所有文化元素围绕人物缓慢旋转"
            narration = f"这些地标、风物与记忆，共同组成了{place_name}独特的文化名片。"
        else:
            desc = f"地图轮廓内出现{item['element_name']}相关元素：{keywords}，画面像地方文化绘本"
            action = "主角沿地图路线走到该点位，停下观察，并与文化物件产生简单互动"
            narration = f"在{place_name}，{item['element_name']}承载着当地人的生活记忆与文化想象。"

        scenes.append({
            "scene_id": i + 1,
            "description": desc,
            "action": action,
            "narration": narration,
            "style": "二次元, 手绘风, 童话风, 清新温暖",
            "culture_element": item.get("element_name", ""),
            "contour_map": contour_map_path,
            "seed": 42 + i * 100,
            "variants": 1,
            "shot_type": template["shot_type"],
            "transition": template["transition"],
            "camera_movement": template["camera"],
            "duration_seconds": 9.0,
            "scene_prompt": generate_scene_prompt(place_name, item),
            "place_name": place_name,
        })
    return scenes

def run_pipeline_interactive():
    print("=" * 60)
    print("  C-VisualEngineer — 完整流水线")
    print("  输入地名 → 自动获取地图 → 生成文化动画视频")
    print("=" * 60)

    place_name = input("\n请输入地名（如 杭州、上虞、北京）: ").strip()
    if not place_name:
        place_name = PLACEHOLDER_PLACE
        print(f"  使用默认地名: {place_name}")

    maps_dir = ROOT / "input" / "maps"
    contours_dir = ROOT / "input" / "contours"
    maps_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)

    map_path = str(maps_dir / f"{place_name}.png")
    mask_path = str(contours_dir / f"{place_name}_map.png")

    log(f"尝试从 OSM 获取地图: {place_name}")
    osm_ok = fetch_osm_map(place_name, map_path)

    if not osm_ok:
        print(f"\n⚠️  无法自动获取「{place_name}」的地图。")
        print("请手动上传该地区的地图图片（.png/.jpg），或提供描述信息。")
        upload_choice = input("是否已有地图图片？(y/n, 默认 n): ").strip().lower()
        if upload_choice == "y":
            uploaded = input("请将图片路径拖入终端: ").strip().strip('"').strip("'")
            if Path(uploaded).exists():
                shutil.copy(uploaded, map_path)
                log(f"已复制地图: {uploaded} → {map_path}")
            else:
                log(f"文件不存在，使用占位地图: {uploaded}")
                generate_placeholder_map(place_name, map_path)
        else:
            desc = input(f"请简单描述「{place_name}」的特色（如：江南水乡、历史名城）: ").strip()
            generate_placeholder_map(place_name, map_path)
    else:
        log("OSM 地图获取成功，继续处理")

    log("提取地图轮廓掩码...")
    extract_contour_from_map(map_path, mask_path)

    culture_items = None
    log("搜索当地文化元素...")
    raw = search_culture(place_name)
    if raw:
        culture_items = wiki_to_culture_items(place_name, raw)
        culture_items = rank_culture_items(culture_items)
        log(f"找到 {len(culture_items)} 个文化元素（已排序）")
        culture_items = interactive_select_items(culture_items, needed=5)

    log("生成分镜脚本...")
    scenes = generate_storyboard_scenes(place_name, mask_path, culture_items)

    sb_path = ROOT / "storyboard_generated.json"
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    log(f"分镜脚本已保存: {sb_path} ({len(scenes)} 个场景)")

    log("\n" + "=" * 60)
    log("步骤 1/3: 生成场景图...")
    from scene_generator import main as scene_main
    scene_main([str(sb_path)])

    log("\n" + "=" * 60)
    log("步骤 2/3: 生成动画片段...")
    from animation_generator import main as anim_main
    try:
        anim_main([str(sb_path)])
    except Exception as e:
        log(f"动画生成跳过: {e}")

    log("\n" + "=" * 60)
    log("步骤 3/3: 合成视频...")
    from video_composer import main as video_main
    video_main([str(sb_path)])

    video_path = VIDEO_DIR / "final_video.mp4"
    if video_path.exists():
        log(f"\n✅ 流水线完成！视频位置: {video_path}")
    else:
        log(f"\n⚠️  视频未生成，请检查上面日志。")

def generate_placeholder_map(place_name: str, output_path: str):
    w, h = 512, 768
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx, ry = w // 4, h // 3
    axes = (rx, ry)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, thickness=cv2.FILLED)
    cv2.putText(mask, place_name[:2], (cx - 40, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    log(f"已生成占位地图掩码: {output_path} ({w}x{h})")

def main():
    parser = argparse.ArgumentParser(description="C-VisualEngineer 完整流水线")
    parser.add_argument("--place", default=None, help="地名（可选，不传则交互式输入）")
    parser.add_argument("--map", default=None, help="地图图片路径（可选，跳过OSM）")
    parser.add_argument("--storyboard", default=None, help="已有分镜JSON（可选，跳过生成）")
    parser.add_argument("--theme", default=None, help="按主题筛选（如 建筑地标、饮食、自然景观）")
    args = parser.parse_args()

    if args.storyboard:
        log(f"使用已有分镜: {args.storyboard}")
        from scene_generator import main as scene_main
        scene_main([args.storyboard])
        from animation_generator import main as anim_main
        try:
            anim_main([args.storyboard])
        except Exception as e:
            log(f"动画跳过: {e}")
        from video_composer import main as video_main
        video_main([args.storyboard])
        return

    if args.place:
        run_pipeline_cli(args.place, args.map, theme=args.theme)
    else:
        run_pipeline_interactive()

def run_pipeline_cli(place_name: str, map_path_arg: str | None = None, theme: str | None = None):
    maps_dir = ROOT / "input" / "maps"
    contours_dir = ROOT / "input" / "contours"
    maps_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)

    map_path = map_path_arg or str(maps_dir / f"{place_name}.png")
    mask_path = str(contours_dir / f"{place_name}_map.png")

    if not map_path_arg:
        log(f"获取 OSM 地图: {place_name}")
        if not fetch_osm_map(place_name, map_path):
            log("OSM 获取失败，生成占位地图")
            generate_placeholder_map(place_name, map_path)
    else:
        log(f"使用本地地图: {map_path}")

    log("提取轮廓...")
    extract_contour_from_map(map_path, mask_path)

    culture_items = None
    log("搜索文化元素...")
    raw = search_culture(place_name)
    if raw:
        culture_items = wiki_to_culture_items(place_name, raw)
        culture_items = rank_culture_items(culture_items)
        if theme:
            filtered = [item for item in culture_items if theme.lower() in item.get("category", "").lower()]
            if filtered:
                culture_items = filtered
                log(f"按主题「{theme}」筛选: {len(culture_items)} 个")
        culture_items = culture_items[:5]
        log(f"选中 {len(culture_items)} 个元素")

    log("生成分镜...")
    scenes = generate_storyboard_scenes(place_name, mask_path, culture_items)
    sb_path = ROOT / "storyboard_generated.json"
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    log("运行场景生成...")
    from scene_generator import main as scene_main
    scene_main([str(sb_path)])

    log("运行动画生成...")
    from animation_generator import main as anim_main
    try:
        anim_main([str(sb_path)])
    except Exception as e:
        log(f"动画跳过: {e}")

    log("合成视频...")
    from video_composer import main as video_main
    video_main([str(sb_path)])

if __name__ == "__main__":
    main()
