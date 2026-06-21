#!/usr/bin/env python3
import argparse, json, sys, os, math, subprocess, random, shutil, requests
from pathlib import Path

REF_IMAGE = Path(__file__).resolve().parent / "style_ref.jpg"

import cv2, numpy as np
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from mapgen.config import load_dotenv, get_settings
load_dotenv()

os.environ.setdefault("SEARCH_PROVIDER", "tavily")
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ.setdefault("OPENAI_TEXT_MODEL", "qwen-turbo")

from mapgen.rag.search import search_culture_elements, SearchConfigurationError
from mapgen.rag.inventory import build_culture_inventory
from mapgen.storyboard.generate_storyboard import generate_storyboard
from mapgen.storyboard.generate_prompts import generate_prompts
from mapgen.drawing.api import DrawingAgent
from mapgen.place.osm import get_osm_contour
from mapgen.video.motion import MotionVideoAgent
from mapgen.media.tts import synthesize_dubbing
from mapgen.media.video import mux_audio, burn_subtitle


def _style_transfer(img_path: str, prompt: str, output_path: str, ref: str | None = None):
    """Apply watercolor style transfer to an image via Alibaba img2img."""
    draw = DrawingAgent(provider="alibaba")
    draw._draw_alibaba_with_ref(prompt, output_path, ref or img_path)


def main():
    parser = argparse.ArgumentParser(description="文化手账视频生成器")
    parser.add_argument("place", nargs="?", help="地名，不填则交互输入")
    parser.add_argument("--draw-provider", default="", help="绘图 API: baidu / alibaba")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--skip-draw", action="store_true", help="跳过绘图")
    parser.add_argument("--layout", help="已有 layout.json 路径")
    args = parser.parse_args()
    if not args.place:
        args.place = input("> ").strip()
        while not args.place: args.place = input("> ").strip()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  文化手账视频")

    scenes = []

    if args.layout:
        print(f"[Layout] 使用已有: {args.layout}")
        with open(args.layout, encoding="utf-8") as f: layout_data = json.load(f)
        sb_path = Path(args.layout).parent / f"{args.place.lower()}_storyboard.json"
        if sb_path.exists():
            scenes = json.loads(sb_path.read_text(encoding="utf-8")).get("storyboard", {}).get("scenes", [])
    else:
        print(f"\n[Agent 1] 文化元素 -> {args.place}")
        settings = get_settings()
        inventory_path = None; _search_ok = False; _manual_map_path = None

        # -- Step 1: Map --
        from mapgen.place.osm import get_osm_contour
        print(f"\n[地图] 搜索 {args.place} 地图轮廓...")
        _osm_raw = get_osm_contour(args.place, {"output_width": 1024, "output_height": 1024, "zoom": 13})

        style_transfer_images = []  # paths of style-transferred reference images

        if _osm_raw.get("fallback"):
            print(f"  未找到 {args.place} 的地图轮廓。")
            print("  [1] 文字描述当地特色  [2] 上传图片做风格转换")
            _choice = input("  选择 (1/2): ").strip()
            while _choice not in ("1", "2"):
                _choice = input("  请输入 1 或 2: ").strip()

            if _choice == "2":
                # Upload image for style transfer
                try:
                    from google.colab import files as _fup
                    _uploaded = _fup.upload()
                    if _uploaded:
                        _fname = next(iter(_uploaded))
                        _upload_path = Path(_fname)
                    else: raise RuntimeError("no file")
                except Exception:
                    _p_str = input("  图片路径: ").strip()
                    while not _p_str or not Path(_p_str).exists():
                        _p_str = input("  文件不存在, 重新输入: ").strip()
                    _upload_path = Path(_p_str)
                print(f"  已获取图片: {_upload_path}")
                # Style transfer
                _st_out = out_dir / "style_transfer_map.png"
                _st_prompt = f"水彩手绘风格, {args.place} 地图, 温暖色调, 旅游手账"
                print("  正在做风格转换 (img2img)...")
                _style_transfer(str(_upload_path), _st_prompt, str(_st_out), ref=str(REF_IMAGE) if REF_IMAGE.exists() else None)
                _manual_map_path = _st_out
                style_transfer_images.append(str(_st_out))
                print(f"  风格转换完成: {_st_out}")
                # Derive mask from style-transferred image
                _map_img = cv2.imread(str(_st_out))
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

            # _choice == "1" or after upload: manual culture features
            print(f"\n[文化] 输入 {args.place} 的特色（逗号分隔）:")
            _manual_input = input("  > ").strip()
            while not _manual_input:
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
                _manual_items.append({"place_name": args.place, "element_name": feat, "category": _guess_cat(feat),
                    "summary": f"{args.place}的{feat}", "visual_keywords": [feat],
                    "usage_suggestions": [f"参观{feat}"], "confidence": 1.0, "sources": []})
            inv_path = out_dir / f"{args.place.lower()}_inventory.json"
            inv_path.write_text(json.dumps({"inventory": _manual_items}, ensure_ascii=False, indent=2), encoding="utf-8")
            inventory_path = inv_path
            print(f"  已录入 {len(_manual_items)} 项")
        else:
            _manual_map_path = None
            # Map found — search culture elements
            if settings.search_api_key:
                print(f"\n[文化] 搜索 {args.place} 文化元素...")
                try:
                    raw = search_culture_elements([{"name": args.place}], {
                        "api_key": settings.search_api_key,
                        "query_template": "{place} 标志性景点 代表性文化 打卡地标 必去",
                    })
                    inv = build_culture_inventory([{"name": args.place}], raw)
                    items = inv.get("inventory", [])
                    _place_lower = args.place.lower()
                    _relevant = [i for i in items if _place_lower in str(i.get("place_name","")).lower() or _place_lower in str(i.get("element_name","")).lower()]
                    if _relevant:
                        print(f"  文化元素: {len(_relevant)} 项")
                        inv_path = out_dir / f"{args.place.lower()}_inventory.json"
                        inv_path.write_text(json.dumps({"inventory": _relevant}, ensure_ascii=False, indent=2), encoding="utf-8")
                        inventory_path = inv_path; _search_ok = True
                    else: print("  搜索结果与地名无关")
                except (SearchConfigurationError, Exception) as e: print(f"  搜索失败 ({e})")

            if not _search_ok:
                print(f"\n  搜索不到 {args.place} 的文化信息，手动输入:")
                _manual_input = input("  > ").strip()
                while not _manual_input: _manual_input = input("  > ").strip()
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
                    _manual_items.append({"place_name": args.place, "element_name": feat, "category": _guess_cat(feat),
                        "summary": f"{args.place}的{feat}", "visual_keywords": [feat],
                        "usage_suggestions": [f"参观{feat}"], "confidence": 1.0, "sources": []})
                inv_path = out_dir / f"{args.place.lower()}_inventory.json"
                inv_path.write_text(json.dumps({"inventory": _manual_items}, ensure_ascii=False, indent=2), encoding="utf-8")
                inventory_path = inv_path
                print(f"  已录入 {len(_manual_items)} 项")

            # -- Map found: also search web images for style transfer --
            if settings.search_api_key:
                print(f"\n[图片素材] 搜索 {args.place} 相关图片...")
                try:
                    _img_raw = search_culture_elements([{"name": args.place}], {
                        "api_key": settings.search_api_key,
                        "query_template": "{place} 风景 地标 照片",
                    })
                    _inv2 = build_culture_inventory([{"name": args.place}], _img_raw)
                    _img_items = _inv2.get("inventory", [])[:3]
                    for _ii in _img_items:
                        _elem = _ii.get("element_name", "")
                        if not _elem: continue
                        print(f"  生成风格转换: {_elem}")
                        _st_out = out_dir / f"ref_{__import__('uuid').uuid4().hex[:8]}.png"
                        _st_prompt = f"水彩手绘风格, {args.place} {_elem}, 温暖色调"
                        try:
                            _style_transfer(str(REF_IMAGE) if REF_IMAGE.exists() else "",
                                            _st_prompt, str(_st_out))
                            if _st_out.exists(): style_transfer_images.append(str(_st_out))
                        except Exception as _e:
                            print(f"    风格转换跳过: {_e}")
                except Exception as _e:
                    print(f"  图片搜索跳过: {_e}")

        # -- Storyboard --
        print("  生成分镜...")
        sb = generate_storyboard(place_name=args.place, inventory_path=inventory_path, num_scenes=4, target_duration_seconds=12.0, use_llm=True)
        storyboard = sb["storyboard"]; scenes = storyboard.get("scenes", [])

        print("  生成绘图提示词...")
        prompts_result = generate_prompts(storyboard_path=sb["path"], prompt_type="image", use_llm=True)
        scene_prompts = prompts_result["prompts"]["prompts"]
        prompt_by_scene = {p["scene_id"]: p for p in scene_prompts}

        print(f"  分镜: {len(scenes)} 场")

        # -- Agent 2: Draw --
        print(f"\n[Agent 2] 绘图 -> 4 张全图")
        draw_agent = DrawingAgent(provider=args.draw_provider or "alibaba")
        scene_images = []
        fixed_seed = random.randint(1, 2147483646)
        ref = str(REF_IMAGE) if REF_IMAGE.exists() else None
        if ref: print(f"  参考图: {ref}")
        for i, scene in enumerate(scenes):
            theme = scene["title"]; sp = prompt_by_scene.get(scene["scene_id"], {})
            prompt = sp.get("positive_prompt", "")
            if not prompt or len(prompt) < 20:
                prompt = f"水彩手绘, {args.place}当地文化, {theme}, 地图轮廓构图, 一个二次元女孩旅游, 可爱风格, 温暖色调"
            path = out_dir / f"scene_{i+1:02d}_{__import__('uuid').uuid4().hex[:8]}.png"
            draw_agent._draw_one_large(prompt, str(path), seed=fixed_seed, ref_image=ref)
            scene_images.append({"theme": theme, "prompt": prompt, "path": str(path.resolve())})
            print(f"  [场景] {i+1}/{len(scenes)}: {theme}")

        # -- Map mask & blur transition --
        print(f"\n[构图] 地图轮廓 + 模糊过渡...")
        if _manual_map_path and _manual_map_path.exists():
            print(f"  手动地图: {_manual_map_path}")
            _map_img = cv2.imread(str(_manual_map_path))
            _gray = cv2.cvtColor(_map_img, cv2.COLOR_BGR2GRAY)
            _, _thresh = cv2.threshold(_gray, 240, 255, cv2.THRESH_BINARY_INV)
            _contours_found, _ = cv2.findContours(_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if _contours_found:
                _largest = max(_contours_found, key=cv2.contourArea)
                _raw_mask = np.zeros_like(_gray)
                cv2.drawContours(_raw_mask, [_largest], -1, 255, thickness=cv2.FILLED)
                mask_raw = Image.fromarray(cv2.dilate(_raw_mask, np.ones((5,5),np.uint8), iterations=2)).convert("L")
            else: mask_raw = Image.new("L", (1024, 1024), 255)
        else:
            mask_raw = Image.open(_osm_raw["mask_path"]).convert("L")
            print(f"  地图来源: {_osm_raw.get('source', 'unknown')}")

        # Scale mask to fill canvas
        arr = np.array(mask_raw)
        ys, xs = np.where(arr > 80)
        if len(ys) > 0:
            min_y, max_y = int(ys.min()), int(ys.max())
            min_x, max_x = int(xs.min()), int(xs.max())
            bw, bh = max_x - min_x, max_y - min_y
            sx = 1024 / bw if bw > 0 else 1.0; sy = 1024 / bh if bh > 0 else 1.0
            mask_final = mask_raw.resize((int(bw*sx), int(bh*sy)), Image.LANCZOS)
        else: mask_final = Image.new("L", (1024, 1024), 255)

        mask_final = mask_final.point(lambda x: 255 if x > 80 else 0)

        # Dilate for breathing room
        arr_final = np.array(mask_final)
        kernel7 = np.ones((7,7), np.uint8)
        arr_dilated = cv2.dilate(arr_final, kernel7, iterations=7)
        mask_roomy = Image.fromarray(arr_dilated).convert("L")

        # Blur transition border (instead of hard black outline)
        dist = cv2.distanceTransform(arr_final, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur_radius = 25
        blur_edge = cv2.GaussianBlur(dist, (blur_radius, blur_radius), 0)
        # Feather: edge zone gets semi-transparent
        blur_alpha = np.clip(blur_edge.astype(float) * 1.5, 0, 255).astype(np.uint8)
        border_transition = Image.fromarray(blur_alpha).convert("L")

        bg_color = (245, 240, 230)
        for si in scene_images:
            img = Image.open(si["path"]).convert("RGBA")
            inside = Image.composite(img, Image.new("RGBA", (1024,1024), (0,0,0,0)), mask_roomy)
            canvas = Image.new("RGBA", (1024,1024), (*bg_color, 255))
            canvas = Image.alpha_composite(canvas, inside)
            canvas.save(si["path"])

        # Transition overlay (blur edge instead of hard border)
        transition_overlay = Image.new("RGBA", (1024,1024), (0,0,0,0))
        # Semi-transparent color along the edge for a soft vignette
        edge_color = (50, 55, 70, 60)  # very faint gray-blue
        _edge_layer = Image.new("RGBA", (1024,1024), edge_color)
        _edge_layer.putalpha(border_transition)
        transition_overlay = Image.alpha_composite(transition_overlay, _edge_layer)
        border_overlay_path = out_dir / f"{args.place.lower()}_border.png"
        transition_overlay.save(str(border_overlay_path))

        mask_path = out_dir / f"{args.place.lower()}_mask.png"
        mask_roomy.save(str(mask_path))
        print(f"  构图完成: {len(scene_images)} 张 + 模糊过渡overlay + mask")

        # -- Layout --
        print(f"\n[Agent 3] 组装场景列表")
        layout_data = {
            "place": args.place, "canvas_width": 1024, "canvas_height": 1024,
            "border_overlay": str(border_overlay_path), "mask_path": str(mask_path), "bg_color": list(bg_color),
            "scenes": [{"theme": si["theme"], "image_path": si["path"], "prompt": si["prompt"],
                        "zoom_start": 1.0, "zoom_end": 1.06} for si in scene_images],
            "style_transfer_images": style_transfer_images,
        }
        layout_path = out_dir / f"{args.place.lower()}_layout.json"
        layout_path.write_text(json.dumps(layout_data, ensure_ascii=False, indent=2), encoding="utf-8")
        layout_data["layout_path"] = str(layout_path.resolve())
        print(f"  layout: {layout_data['layout_path']}")

    # -- Agent 4: Video --
    print(f"\n[Agent 4] 视频 -> 12s MP4")
    motion = MotionVideoAgent(layout_data, out_dir)
    video_path = motion.render()
    print(f"  视频: {video_path}")

    # -- Agent 5: TTS + Subtitles --
    print(f"\n[Agent 5] TTS 配音 + 字幕")
    narration_list = [s.get("narration", "") for s in scenes]
    _seen_narr = set()
    for i, n in enumerate(narration_list):
        if not n.strip() or n in _seen_narr: narration_list[i] = f"这是{scenes[i]['title']}。"
        _seen_narr.add(narration_list[i])
    if any(narration_list):
        per_scene_sec = 3.0; wave_parts = []
        for i, text in enumerate(narration_list):
            if not text.strip(): text = f"{scenes[i]['title']}。"
            raw_mp3 = out_dir / f"scene_tts_{i+1:02d}.mp3"
            print(f"  [TTS] 场景 {i+1}: {text[:30]}...")
            synthesize_dubbing(text, raw_mp3, audio_format="mp3", voice="longxiaochun", model="cosyvoice-v1", provider="dashscope")
            wav = out_dir / f"scene_pad_{i+1:02d}.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp3), "-af", f"apad=pad_dur={per_scene_sec}",
                "-t", str(per_scene_sec), "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", str(wav)], check=True)
            wave_parts.append(str(wav))
        concat_list = out_dir / "audio_concat.txt"
        concat_list.write_text("\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in wave_parts), encoding="utf-8")
        final_audio = out_dir / "dubbing.m4a"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:a", "aac", "-b:a", "192k", "-y", str(final_audio)], check=True)
        print(f"  音频: {final_audio}")

        # SRT
        srt_path = out_dir / "dubbing.srt"
        srt_lines = []
        for i, text in enumerate(narration_list):
            if not text.strip(): text = f"{scenes[i]['title']}。"
            start = i * per_scene_sec; end = (i + 1) * per_scene_sec
            def _fmt(t): h=int(t//3600); m=int(t%3600//60); s=t%60; return f"{h:02d}:{m:02d}:{s:06.3f}"
            srt_lines.extend([str(i+1), f"{_fmt(start)} --> {_fmt(end)}", text, ""])
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        print(f"  字幕: {srt_path}")

        # Mux audio
        print(f"\n  混音")
        dubbed = mux_audio(video_path, final_audio, out_dir / "final_dubbed.mp4", mode="replace", shortest=False)
        dubbed_path = dubbed["output_path"]

        # Burn subtitles (font auto-detect in video.py)
        print(f"\n  烧录字幕")
        final = burn_subtitle(dubbed_path, srt_path, out_dir / "final_with_sub.mp4")
        final_path = final["output_path"]
    else:
        print("  无旁白，跳过配音")
        final_path = video_path

    print(f"\n  完成! {final_path}")


if __name__ == "__main__":
    main()
