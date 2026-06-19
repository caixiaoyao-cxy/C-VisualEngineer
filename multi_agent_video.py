#!/usr/bin/env python3
"""
多智能体自动化：地名 → 故事板 → 4 张场景全图 → 12 秒淡入淡出微动视频

流程:
  Agent 1 (RAG) : 联网搜索 → 文化元素清单 → 4 场分镜 + 绘图提示词
  Agent 2 (Draw) : 4 张场景全图 (提示词驱动，含人物+文化元素+地图构图)
  Agent 3 (Layout): 组装场景列表 json
  Agent 4 (Video) : 12s MP4, Ken Burns 慢放 + 淡入淡出
"""
import argparse
import json
import sys
from pathlib import Path

REF_IMAGE = Path(__file__).resolve().parent / "style_ref.jpg"

import cv2
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapgen.config import load_dotenv, get_settings
load_dotenv()

import os as _os
_os.environ.setdefault("SEARCH_PROVIDER", "tavily")
_os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
_os.environ.setdefault("OPENAI_TEXT_MODEL", "qwen-turbo")

from mapgen.rag.search import search_culture_elements, SearchConfigurationError
from mapgen.rag.inventory import build_culture_inventory
from mapgen.storyboard.generate_storyboard import generate_storyboard
from mapgen.storyboard.generate_prompts import generate_prompts
from mapgen.drawing.api import DrawingAgent
from mapgen.place.osm import get_osm_contour
from mapgen.video.motion import MotionVideoAgent
from mapgen.media.tts import synthesize_dubbing
from mapgen.media.video import mux_audio, burn_subtitle


def main():
    parser = argparse.ArgumentParser(description="多智能体文化手账视频生成器")
    parser.add_argument("place", nargs="?", help="地名（如 Hong Kong / Tokyo / Paris / Kyoto），不填则交互式输入")
    parser.add_argument("--draw-provider", default="", help="绘图 API: baidu 或 alibaba")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--skip-draw", action="store_true", help="跳过绘图步骤（使用已有贴纸）")
    parser.add_argument("--layout", help="已有 layout.json 路径（跳过 1-3）")
    args = parser.parse_args()
    if not args.place:
        args.place = input("请输入地名 (如 Hong Kong / Tokyo / Paris / Kyoto): ").strip()
        while not args.place:
            args.place = input("地名不能为空，请重新输入: ").strip()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  文化手账视频 (B 分支完整场景)")

    scenes = []

    if args.layout:
        print(f"\n[Agent 1-3] 跳过，使用已有 layout: {args.layout}")
        with open(args.layout, encoding="utf-8") as f:
            layout_data = json.load(f)
        # Try to load storyboard for narration
        sb_path = Path(args.layout).parent / f"{args.place.lower()}_storyboard.json"
        if sb_path.exists():
            sb_data = json.loads(sb_path.read_text(encoding="utf-8"))
            scenes = sb_data.get("storyboard", {}).get("scenes", [])
    else:
        # ── Agent 1: RAG + 分镜 + 提示词 ────────────────────────────
        print(f"\n[Agent 1] 搜索文化元素 → {args.place}")
        settings = get_settings()
        places = [{"name": args.place}]

        inventory_path = None
        _search_ok = False
        _manual_map_path = None

        # ── 第一步：搜地图 ──────────────────────────────────────────────
        from mapgen.place.osm import get_osm_contour
        print(f"\n[地图] 搜索「{args.place}」的地图轮廓...")
        _osm_raw = get_osm_contour(args.place, {"output_width": 1024, "output_height": 1024, "zoom": 13})
        if _osm_raw.get("fallback"):
            print(f"  ⚠ 未找到「{args.place}」的地图轮廓。")
            print("  请提供当地地图图片（轮廓图，白底黑轮廓）。")
            try:
                from google.colab import files
                print("  正在打开文件上传对话框...")
                _uploaded = files.upload()
                if _uploaded:
                    _fname = next(iter(_uploaded))
                    _p = Path(_fname)
                    _manual_map_path = _p
                    print(f"  已上传: {_fname}")
                else:
                    raise RuntimeError("未上传文件")
            except ImportError:
                _user_map = input("  输入文件路径: ").strip()
                while not _user_map:
                    _user_map = input("  路径不能为空: ").strip()
                _p = Path(_user_map)
                while not _p.exists():
                    _user_map = input("  文件不存在，重新输入: ").strip()
                    _p = Path(_user_map)
                _manual_map_path = _p
                print(f"  已使用地图: {_p.resolve()}")
            # 地图手动提供，直接手动输文化元素
            print(f"\n[文化] 请输入「{args.place}」的当地特色（景点/食物/活动等，逗号分隔，至少一项）：")
            _manual_input = input("  > ").strip()
            while not _manual_input:
                print("  至少输入一项特色才能继续：")
                _manual_input = input("  > ").strip()
            _features = [f.strip() for f in _manual_input.split(",") if f.strip()]
            _cat_map = {"塔":"建筑地标","寺":"建筑地标","庙":"建筑地标","楼":"建筑地标","街":"建筑地标","桥":"建筑地标",
                        "湖":"自然景观","山":"自然景观","河":"自然景观","江":"自然景观","海":"自然景观","岛":"自然景观",
                        "吃":"饮食","食":"饮食","菜":"饮食","茶":"饮食","酒":"饮食","小吃":"饮食",
                        "节":"民俗节庆","庆":"民俗节庆","会":"民俗节庆"}
            def _guess_cat(n):
                for kw, cat in _cat_map.items():
                    if kw in n: return cat
                return "建筑地标"
            _manual_items = []
            for feat in _features[:4]:
                _manual_items.append({
                    "place_name": args.place,
                    "element_name": feat,
                    "category": _guess_cat(feat),
                    "summary": f"{args.place}的{feat}",
                    "visual_keywords": [feat],
                    "usage_suggestions": [f"参观{feat}"],
                    "confidence": 1.0,
                    "sources": [],
                })
            inv_path = out_dir / f"{args.place.lower()}_inventory.json"
            inv_path.write_text(json.dumps({"inventory": _manual_items}, ensure_ascii=False, indent=2), encoding="utf-8")
            inventory_path = inv_path
            print(f"  已录入 {len(_manual_items)} 项文化特色")
        else:
            _manual_map_path = None
            # 地图自动命中 → 再搜文化元素
            if settings.search_api_key:
                print(f"\n[文化] 搜索「{args.place}」的文化元素...")
                try:
                    raw = search_culture_elements(places, {
                        "api_key": settings.search_api_key,
                        "query_template": "{place} 标志性景点 代表性文化 打卡地标 必去",
                    })
                    inv = build_culture_inventory(places, raw)
                    items = inv.get("inventory", [])
                    _place_lower = args.place.lower()
                    _relevant = [i for i in items if _place_lower in str(i.get("place_name", "")).lower() or _place_lower in str(i.get("element_name", "")).lower()]
                    if _relevant:
                        print(f"  文化元素: {len(_relevant)} 项")
                        inv_path = out_dir / f"{args.place.lower()}_inventory.json"
                        inv_path.write_text(
                            json.dumps({"inventory": _relevant}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        inventory_path = inv_path
                        _search_ok = True
                    else:
                        print("  搜索到的内容与地名无关")
                except (SearchConfigurationError, Exception) as e:
                    print(f"  搜索失败 ({e})")

            if not _search_ok:
                print(f"\n  ⚠ 搜索不到「{args.place}」的文化信息，请手动输入当地特色。")
                print("  请输入景点/食物/活动等，逗号分隔，至少一项：")
                _manual_input = input("  > ").strip()
                while not _manual_input:
                    print("  至少输入一项特色才能继续：")
                    _manual_input = input("  > ").strip()
                _features = [f.strip() for f in _manual_input.split(",") if f.strip()]
                _cat_map = {"塔":"建筑地标","寺":"建筑地标","庙":"建筑地标","楼":"建筑地标","街":"建筑地标","桥":"建筑地标",
                            "湖":"自然景观","山":"自然景观","河":"自然景观","江":"自然景观","海":"自然景观","岛":"自然景观",
                            "吃":"饮食","食":"饮食","菜":"饮食","茶":"饮食","酒":"饮食","小吃":"饮食",
                            "节":"民俗节庆","庆":"民俗节庆","会":"民俗节庆"}
                def _guess_cat(n):
                    for kw, cat in _cat_map.items():
                        if kw in n: return cat
                    return "建筑地标"
                _manual_items = []
                for feat in _features[:4]:
                    _manual_items.append({
                        "place_name": args.place,
                        "element_name": feat,
                        "category": _guess_cat(feat),
                        "summary": f"{args.place}的{feat}",
                        "visual_keywords": [feat],
                        "usage_suggestions": [f"参观{feat}"],
                        "confidence": 1.0,
                        "sources": [],
                    })
                inv_path = out_dir / f"{args.place.lower()}_inventory.json"
                inv_path.write_text(json.dumps({"inventory": _manual_items}, ensure_ascii=False, indent=2), encoding="utf-8")
                inventory_path = inv_path
                print(f"  已录入 {len(_manual_items)} 项文化特色")

        print("  生成分镜 (4 场景)...")
        sb = generate_storyboard(
            place_name=args.place,
            inventory_path=inventory_path,
            num_scenes=4,
            target_duration_seconds=12.0,
            use_llm=True,
        )
        storyboard = sb["storyboard"]
        scenes = storyboard.get("scenes", [])

        print("  生成绘图提示词...")
        prompts_result = generate_prompts(
            storyboard_path=sb["path"],
            prompt_type="image",
            use_llm=True,
        )
        scene_prompts = prompts_result["prompts"]["prompts"]
        prompt_by_scene = {p["scene_id"]: p for p in scene_prompts}

        print(f"  分镜: {len(scenes)} 场")
        for s in scenes:
            sp = prompt_by_scene.get(s["scene_id"], {})
            pp = sp.get("positive_prompt", "")
            print(f"    场 {s['scene_id']}: {s['title']}")

        # ── Agent 2: 绘图 ────────────────────────────────────────────
        print(f"\n[Agent 2] 绘图 → 4 张场景全图")
        draw_agent = DrawingAgent(provider=args.draw_provider or "alibaba")

        scene_images = []
        import random
        fixed_seed = random.randint(1, 2147483646)
        ref = str(REF_IMAGE) if REF_IMAGE.exists() else None
        if ref:
            print(f"  参考图: {ref}")
        for i, scene in enumerate(scenes):
            theme = scene["title"]
            sp = prompt_by_scene.get(scene["scene_id"], {})
            prompt = sp.get("positive_prompt", "")
            if not prompt or len(prompt) < 20:
                prompt = f"水彩手绘, {args.place}当地文化, {theme}, 地图轮廓构图, 一个二次元女孩旅游, 可爱风格, 温暖色调"
            path = out_dir / f"scene_{i+1:02d}_{__import__('uuid').uuid4().hex[:8]}.png"
            draw_agent._draw_one_large(prompt, str(path), seed=fixed_seed, ref_image=ref)
            scene_images.append({"theme": theme, "prompt": prompt, "path": str(path.resolve())})
            print(f"  [场景] {i+1}/{len(scenes)}: {theme}")

        # ── Agent 2.5: 地图边框 —— 画面裁切到地图轮廓内 ────────────────
        print(f"\n[Agent 2.5] 地图轮廓构图...")
        if _manual_map_path:
            print(f"  手动地图: {_manual_map_path.resolve()}")
            _map_img = cv2.imread(str(_manual_map_path))
            _gray = cv2.cvtColor(_map_img, cv2.COLOR_BGR2GRAY)
            _, _thresh = cv2.threshold(_gray, 240, 255, cv2.THRESH_BINARY_INV)
            _contours_found, _ = cv2.findContours(_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if _contours_found:
                _largest = max(_contours_found, key=cv2.contourArea)
                _raw_mask = np.zeros_like(_gray)
                cv2.drawContours(_raw_mask, [_largest], -1, 255, thickness=cv2.FILLED)
                mask_raw = Image.fromarray(cv2.dilate(_raw_mask, np.ones((5,5),np.uint8), iterations=2)).convert("L")
            else:
                mask_raw = Image.new("L", (1024, 1024), 255)
        else:
            mask_raw = Image.open(_osm_raw["mask_path"]).convert("L")
            source = _osm_raw.get("source", "unknown")
            print(f"  地图来源: {source}")
            if _osm_raw.get("fallback"):
                print(f"  [WARN] 地图轮廓降级: {osm.get('fallback_reason', 'unknown')}")

        # 把 mask 缩放到铺满全画布，必要时横向/纵向拉长
        arr = np.array(mask_raw)
        ys, xs = np.where(arr > 80)
        if len(ys) > 0:
            min_y, max_y = int(ys.min()), int(ys.max())
            min_x, max_x = int(xs.min()), int(xs.max())
            bw, bh = max_x - min_x, max_y - min_y
            sx = 1024 / bw if bw > 0 else 1.0
            sy = 1024 / bh if bh > 0 else 1.0
            nw, nh = int(bw * sx), int(bh * sy)
            mask_scaled = mask_raw.resize((nw, nh), Image.LANCZOS)
            mask_final = mask_scaled
        else:
            mask_final = Image.new("L", (1024, 1024), 255)

        mask_final = mask_final.point(lambda x: 255 if x > 80 else 0)

        # 膨胀 mask，给人物更多活动空间
        arr_final = np.array(mask_final)
        kernel7 = np.ones((7, 7), np.uint8)
        arr_dilated = cv2.dilate(arr_final, kernel7, iterations=7)
        mask_roomy = Image.fromarray(arr_dilated).convert("L")

        # 生成粗描边：用多层膨胀得到明显的地图轮廓
        edges = cv2.Canny(arr_final, 30, 100)
        kernel3 = np.ones((3, 3), np.uint8)
        border_thick = cv2.dilate(edges, kernel3, iterations=4)
        # 内侧装饰线
        border_inner = cv2.dilate(edges, kernel3, iterations=2)
        border_inner = cv2.erode(border_inner, kernel3, iterations=1)
        stroke_outer = Image.fromarray(border_thick).convert("L")
        stroke_inner = Image.fromarray(border_inner).convert("L")

        bg_color = (245, 240, 230)  # 暖白纸色
        border_color = (50, 55, 70)  # 深蓝灰

        for i, si in enumerate(scene_images):
            img = Image.open(si["path"]).convert("RGBA")
            inside = Image.composite(img, Image.new("RGBA", (1024, 1024), (0, 0, 0, 0)), mask_roomy)
            canvas = Image.new("RGBA", (1024, 1024), (*bg_color, 255))
            canvas = Image.alpha_composite(canvas, inside)
            canvas.save(si["path"])

        # 单独保存地图边框 overlay（视频最后合成时叠在每一帧上，保证不动）
        border_overlay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        outer = Image.new("RGBA", (1024, 1024), (*border_color, 255))
        outer.putalpha(stroke_outer)
        border_overlay = Image.alpha_composite(border_overlay, outer)
        inner = Image.new("RGBA", (1024, 1024), (*border_color, 180))
        inner.putalpha(stroke_inner)
        border_overlay = Image.alpha_composite(border_overlay, inner)
        border_overlay_path = out_dir / f"{args.place.lower()}_border.png"
        border_overlay.save(str(border_overlay_path))

        # 保存 mask，用于视频每一帧重新裁切，保证边界不动
        mask_path = out_dir / f"{args.place.lower()}_mask.png"
        mask_roomy.save(str(mask_path))
        print(f"  地图轮廓构图完成: {len(scene_images)} 张 + 边框overlay + mask")

        # ── 生成开场总览地图：地图轮廓 + 倒水滴定位点（可震动）───────────
        print("  生成开场总览地图（倒水滴定位点 + 上下震动）...")
        # 先收集每个景点的名称和像素坐标
        import requests as _req
        import urllib.parse
        _marker_infos = []  # [{px, py, label, is_geo}, ...]
        _place_bounds = {}
        try:
            _geo_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(args.place)}&format=json&limit=1"
            _geo_r = _req.get(_geo_url, headers={"User-Agent": "Map2Video/0.1"}, timeout=10)
            if _geo_r.status_code == 200 and _geo_r.json():
                _geo_d = _geo_r.json()[0]
                _bb = _geo_d.get("boundingbox")
                if _bb and len(_bb) == 4:
                    _place_bounds = {
                        "min_lat": float(_bb[0]), "max_lat": float(_bb[1]),
                        "min_lon": float(_bb[2]), "max_lon": float(_bb[3]),
                    }
        except Exception:
            pass
        _marker_color = (220, 60, 60)
        _label_color = (50, 55, 70)
        for _si, _scene in enumerate(scene_images, start=1):
            _elem_name = _scene.get("theme", "").replace(f"{args.place}·", "").strip()
            if not _elem_name:
                continue
            _coords = {}
            try:
                _eq = urllib.parse.quote(f"{_elem_name} {args.place}")
                _er = _req.get(
                    f"https://nominatim.openstreetmap.org/search?q={_eq}&format=json&limit=1",
                    headers={"User-Agent": "Map2Video/0.1"}, timeout=10
                )
                if _er.status_code == 200 and _er.json():
                    _ed = _er.json()[0]
                    _coords = {"lat": float(_ed["lat"]), "lon": float(_ed["lon"])}
            except Exception:
                pass
            if _coords and _place_bounds:
                _blon = _place_bounds["max_lon"] - _place_bounds["min_lon"]
                _blat = _place_bounds["max_lat"] - _place_bounds["min_lat"]
                _px = int((_coords["lon"] - _place_bounds["min_lon"]) / _blon * 1024) if _blon else 512
                _py = int((_place_bounds["max_lat"] - _coords["lat"]) / _blat * 1024) if _blat else 512
                _px = max(40, min(984, _px))
                _py = max(40, min(984, _py))
                _marker_infos.append({"px": _px, "py": _py, "label": _elem_name[:10], "num": _si, "geo": True})
            else:
                _ring_x = 50 + (_si - 1) * 220
                _marker_infos.append({"px": _ring_x, "py": 980, "label": _elem_name[:10], "num": _si, "geo": False})

        def _draw_teardrop(draw, cx, cy, size, fill, outline, num):
            """画倒水滴定位点"""
            hr = size // 2
            # 圆头
            draw.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=fill, outline=outline, width=2)
            # 尖底（三角形）
            tip = cy + size
            draw.polygon([(cx - hr // 3, cy + hr // 2), (cx + hr // 3, cy + hr // 2), (cx, tip)], fill=fill, outline=outline, width=2)
            # 编号
            draw.text((cx - 5, cy - 7), str(num), fill=(255, 255, 255), font=None)

        # 生成带震动的逐帧动画
        _fps = 10
        _frame_count = _fps * 3  # 3 秒
        _frames_dir = out_dir / "_intro_frames"
        _frames_dir.mkdir(parents=True, exist_ok=True)
        for _fi in range(_frame_count):
            _frame = Image.new("RGBA", (1024, 1024), (*bg_color, 255))
            _frame = Image.alpha_composite(_frame, border_overlay)
            _draw_f = ImageDraw.Draw(_frame)
            # 标题
            _title_text = f"探索 {args.place}"
            try:
                _font = ImageFont.truetype("NotoSansSC-Regular.ttf", 56) if _fi == 0 else font_large
            except Exception:
                _font = ImageFont.load_default()
            if _fi == 0:
                try:
                    font_large = ImageFont.truetype("NotoSansSC-Regular.ttf", 56)
                except Exception:
                    font_large = ImageFont.load_default()
                _font = font_large
            _bbox = _draw_f.textbbox((0, 0), _title_text, font=_font)
            _draw_f.text(((1024 - (_bbox[2] - _bbox[0])) / 2, 30), _title_text, fill=(50, 55, 70), font=_font)
            # 标点（震动）
            _bounce = math.sin(_fi / _fps * math.pi * 2) * 6 if _fi > 0 else 0
            for _mi in _marker_infos:
                if _mi["geo"]:
                    _dy = int(_bounce)
                    _draw_teardrop(_draw_f, _mi["px"], _mi["py"] + _dy, 24, _marker_color, (255, 255, 255), _mi["num"])
                    _draw_f.text((_mi["px"] + 18, _mi["py"] + _dy - 6), _mi["label"], fill=_label_color, font=None)
                else:
                    _draw_f.ellipse([_mi["px"] - 14, _mi["py"] - 14, _mi["px"] + 14, _mi["py"] + 14], fill=None, outline=_marker_color, width=3)
                    _draw_f.text((_mi["px"] - 6, _mi["py"] - 10), str(_mi["num"]), fill=_marker_color, font=None)
                    _draw_f.text((_mi["px"] - 20, _mi["py"] + 20), _mi["label"], fill=_label_color, font=None)
            _frame.convert("RGB").save(str(_frames_dir / f"frame_{_fi:03d}.png"))
        # 合成为视频 clip
        _intro_clip = out_dir / "intro_clip.mp4"
        _sp.run([
            "ffmpeg", "-y", "-framerate", str(_fps),
            "-i", str(_frames_dir / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(_intro_clip)
        ], check=True, capture_output=True)
        # 清理帧
        import shutil
        shutil.rmtree(str(_frames_dir), ignore_errors=True)
        print(f"  开场视频已生成: {_intro_clip}")

        # ── Agent 3: 排版 ────────────────────────────────────────────
        print(f"\n[Agent 3] 组装场景列表")
        layout_data = {
            "place": args.place,
            "canvas_width": 1024,
            "canvas_height": 1024,
            "border_overlay": str(border_overlay_path),
            "mask_path": str(mask_path),
            "bg_color": list(bg_color),
            "scenes": [
                {
                    "theme": si["theme"],
                    "image_path": si["path"],
                    "prompt": si["prompt"],
                    "zoom_start": 1.0,
                    "zoom_end": 1.06,
                }
                for si in scene_images
            ],
        }
        layout_path = out_dir / f"{args.place.lower()}_layout.json"
        layout_path.write_text(json.dumps(layout_data, ensure_ascii=False, indent=2), encoding="utf-8")
        layout_data["layout_path"] = str(layout_path.resolve())
        print(f"  layout: {layout_data['layout_path']}")

    # ── Agent 4: 视频 ──────────────────────────────────────────────
    print(f"\n[Agent 4] 视频 → 12 秒 MP4")
    motion = MotionVideoAgent(layout_data, out_dir)
    video_path = motion.render()
    print(f"  视频完成: {video_path}")

    # ── 开场 clip 拼接 ─────────────────────────────────────────────
    _intro_clip_path = out_dir / "intro_clip.mp4"
    if _intro_clip_path.exists():
        print("  拼接开场视频到片头...")
        _concat_txt = out_dir / "video_concat.txt"
        _concat_txt.write_text(
            f"file '{_intro_clip_path.as_posix()}'\nfile '{video_path.as_posix()}'\n",
            encoding="utf-8"
        )
        _concat_video = out_dir / "combined_preview.mp4"
        _sp.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(_concat_txt),
            "-c", "copy",
            str(_concat_video)
        ], check=True)
        video_path = _concat_video
        print(f"  拼接完成: {video_path}")

    # ── Agent 5: TTS 配音 + 字幕 (逐场景 3s) ─────────────────────
    print(f"\n[Agent 5] TTS 配音 + 字幕 (逐场景 3s)")
    narration_list = [s.get("narration", "") for s in scenes]
    # 去重：重复的解说用场景标题替代
    _seen_narr: set[str] = set()
    for i, n in enumerate(narration_list):
        if not n.strip() or n in _seen_narr:
            narration_list[i] = f"这是{scenes[i]['title']}。"
        _seen_narr.add(narration_list[i])
    # 开场解说
    _intro_narration = f"今天我们去{args.place}，探索这里的文化印记。"
    narration_list = [_intro_narration] + narration_list
    if any(narration_list):
        import subprocess as _sp
        per_scene_sec = 3.0
        wave_parts = []

        for i, text in enumerate(narration_list):
            if not text.strip():
                text = f"{scenes[i]['title']}。"
            raw_mp3 = out_dir / f"scene_tts_{i+1:02d}.mp3"
            print(f"  [TTS] 场景 {i+1}: {text[:30]}...")
            synthesize_dubbing(
                text, raw_mp3,
                audio_format="mp3", voice="longxiaochun",
                model="cosyvoice-v1", provider="dashscope",
            )
            wav = out_dir / f"scene_pad_{i+1:02d}.wav"
            # 转 WAV + 垫到 3s（WAV 无 stream 限制，concat 可靠）
            _sp.run([
                "ffmpeg", "-y",
                "-i", str(raw_mp3),
                "-af", f"apad=pad_dur={per_scene_sec}",
                "-t", str(per_scene_sec),
                "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                str(wav)
            ], check=True)
            wave_parts.append(str(wav))

        # concat demuxer 拼接 WAV（可靠）
        from pathlib import Path as _Path
        concat_list = out_dir / "audio_concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{_Path(p).resolve().as_posix()}'" for p in wave_parts),
            encoding="utf-8"
        )
        final_audio = out_dir / "dubbing.m4a"
        _sp.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:a", "aac", "-b:a", "192k", "-y",
            str(final_audio)
        ], check=True)
        print(f"  音频拼接完成: {final_audio}")

        # 生成逐场景 SRT (每个场景固定 3s)
        srt_path = out_dir / "dubbing.srt"
        srt_lines = []
        for i, text in enumerate(narration_list):
            if not text.strip():
                text = f"{scenes[i]['title']}。"
            start = i * per_scene_sec
            end = (i + 1) * per_scene_sec
            def _fmt(t):
                h = int(t // 3600); m = int(t % 3600 // 60)
                s = t % 60; return f"{h:02d}:{m:02d}:{s:06.3f}"
            srt_lines.append(str(i+1))
            srt_lines.append(f"{_fmt(start)} --> {_fmt(end)}")
            srt_lines.append(text)
            srt_lines.append("")
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        print(f"  字幕生成: {srt_path}")

        # 混音
        print(f"\n[Agent 5] 混音 → 合成最终视频")
        dubbed = mux_audio(video_path, final_audio, out_dir / "final_dubbed.mp4", mode="replace", shortest=False)
        dubbed_path = dubbed["output_path"]

        # 烧录字幕
        print(f"\n[Agent 5] 烧录字幕")
        final = burn_subtitle(dubbed_path, srt_path, out_dir / "final_with_sub.mp4")
        final_path = final["output_path"]
    else:
        print("  无旁白文本，跳过配音")
        final_path = video_path

    print(f"\n  完成! {final_path}")


if __name__ == "__main__":
    main()
