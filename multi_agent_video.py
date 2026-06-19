#!/usr/bin/env python3
"""
多智能体自动化：地名 → 4×6 文化贴纸 × 地图轮廓 → 12 秒微动视频

Agent 1: 文化主题策划    从硬编码数据中取 4 个主题×6 个物品
Agent 2: 水彩插画绘制    调用百度文心一格或阿里通义万相 API
Agent 3: 地图散点排版    地图剪影内散落 24 张贴纸
Agent 4: 微动视频制作    12s MP4, 淡入淡出 + sin 浮动

用法:
  python multi_agent_video.py "Hong Kong"
  python multi_agent_video.py "Kyoto" --draw-provider alibaba
"""
import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 加载 .env 中的密钥
from mapgen.config import load_dotenv
load_dotenv()

from mapgen.culture.planner import CulturePlanner
from mapgen.drawing.api import DrawingAgent
from mapgen.layout.scatter import ScatterLayoutAgent
from mapgen.video.motion import MotionVideoAgent


def main():
    parser = argparse.ArgumentParser(description="多智能体文化手账视频生成器")
    parser.add_argument("place", help="地名（如 Hong Kong / Tokyo / Paris / Kyoto）")
    parser.add_argument("--draw-provider", default="", help="绘图 API: baidu 或 alibaba")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--skip-draw", action="store_true", help="跳过绘图步骤（使用已有贴纸）")
    parser.add_argument("--layout", help="已有 layout.json 路径（跳过 1-3）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  多智能体文化手账视频生成器")
    print("=" * 60)

    # ── Agent 1: 文化主题策划 ──────────────────────────────────────────
    if args.layout:
        print(f"\n[Agent 1–3] 跳过，使用已有 layout: {args.layout}")
        with open(args.layout, encoding="utf-8") as f:
            layout_data = json.load(f)
    else:
        print(f"\n[Agent 1] 文化主题策划 → {args.place}")
        planner = CulturePlanner(args.place)
        themes = planner.plan()
        for t in themes:
            print(f"  主题: {t['theme']}")
            for item in t["items"]:
                print(f"    · {item}")

        # ── Agent 2: 水彩插画绘制 ──────────────────────────────────────
        if not args.skip_draw:
            print(f"\n[Agent 2] 水彩插画绘制 → {len(planner.all_items())} 张贴纸")
            draw_agent = DrawingAgent(provider=args.draw_provider)
            flat_results = draw_agent.generate_all(planner.all_items(), out_dir)
            # 按主题分组回 sticker_layouts
            sticker_layouts = []
            idx = 0
            for t in themes:
                items_block = []
                for _ in range(6):
                    items_block.append({"item": flat_results[idx]["item"], "path": flat_results[idx]["path"]})
                    idx += 1
                sticker_layouts.append({"theme": t["theme"], "items": items_block})
        else:
            print(f"\n[Agent 2] 跳过（使用 --skip-draw）")
            # 创建假路径占位
            sticker_layouts = []
            for t in themes:
                items_block = []
                for item in t["items"]:
                    items_block.append({"item": item, "path": ""})
                sticker_layouts.append({"theme": t["theme"], "items": items_block})

        # ── Agent 3: 地图散点排版 ──────────────────────────────────────
        print(f"\n[Agent 3] 地图散点排版 → 地图剪影 + 24 张贴纸")
        scatter = ScatterLayoutAgent(args.place, sticker_layouts, out_dir)
        layout_data = scatter.compose()
        print(f"  layout: {layout_data['layout_path']}")
        print(f"  preview: {layout_data.get('preview_path', 'N/A')}")

    # ── Agent 4: 微动视频制作 ──────────────────────────────────────────
    print(f"\n[Agent 4] 微动视频制作 → 12 秒 MP4")
    motion = MotionVideoAgent(layout_data, out_dir)
    video_path = motion.render()

    print(f"\n{'=' * 60}")
    print(f"  完成! {video_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
