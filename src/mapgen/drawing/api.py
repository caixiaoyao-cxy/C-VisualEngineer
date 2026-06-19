from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from mapgen.config import get_settings


class DrawingAgent:
    """Agent 2: 水彩插画绘制。

    调用百度文心一格或阿里通义万相 AI 绘图 API。
    为每个词汇生成水彩手绘风格贴纸 PNG。
    """

    STYLE_PROMPT = "水彩手绘风格, 低饱和马卡龙色, 白底, 孤立物品, {item}, 贴纸风格, 扁平化, 干净边缘, 无阴影, 无背景杂物"
    CHARACTER_PROMPT = "水彩手绘, 低饱和马卡龙色, 一个穿红色旗袍的二次元黑发女孩正在{action}, 全身, 可爱风格, 干净背景"

    def __init__(self, provider: str = ""):
        self.provider = (provider or os.getenv("AI_DRAW_PROVIDER", "")).lower()
        if not self.provider:
            raise RuntimeError(
                "未设置 AI_DRAW_PROVIDER。请设为 'baidu' 或 'alibaba'，"
                "并设置对应 API 密钥。"
            )
        if self.provider == "baidu":
            self.api_key = os.getenv("BAIDU_API_KEY", "")
            self.secret_key = os.getenv("BAIDU_SECRET_KEY", "")
            if not self.api_key or not self.secret_key:
                raise RuntimeError(
                    "百度文心一格需要 BAIDU_API_KEY 和 BAIDU_SECRET_KEY"
                )
            self._access_token: str | None = None
        elif self.provider == "alibaba":
            self.api_key = os.getenv("ALIBABA_API_KEY", "")
            if not self.api_key:
                raise RuntimeError("阿里通义万相需要 ALIBABA_API_KEY")
        else:
            raise RuntimeError(f"不支持的绘图提供商: '{provider}'，仅支持 'baidu' 或 'alibaba'")

    def generate_all(
        self, items: list[str], output_dir: str | Path = ""
    ) -> list[dict[str, Any]]:
        """为每个词汇生成一张贴纸，返回每张的结果路径。"""
        out_dir = Path(output_dir) if output_dir else get_settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, item in enumerate(items):
            prompt = self.STYLE_PROMPT.format(item=item)
            path = out_dir / f"sticker_{i+1:02d}_{uuid4().hex[:8]}.png"
            self._draw_one(prompt, str(path))
            results.append({"item": item, "prompt": prompt, "path": str(path.resolve())})
            print(f"  [Drawing] {i+1}/{len(items)}: {item} -> {path.name}")
        return results

    def generate_characters(
        self, themes: list[dict], output_dir: str | Path = ""
    ) -> list[dict[str, Any]]:
        out_dir = Path(output_dir) if output_dir else get_settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, t in enumerate(themes):
            theme = t["theme"]
            action = self._theme_to_action(theme)
            prompt = self.CHARACTER_PROMPT.format(action=action)
            path = out_dir / f"character_{i+1:02d}_{uuid4().hex[:8]}.png"
            self._draw_one_large(prompt, str(path))
            results.append({"theme": theme, "prompt": prompt, "path": str(path.resolve())})
            print(f"  [Drawing] character {i+1}/{len(themes)}: {theme} -> {path.name}")
        return results

    @staticmethod
    def _theme_to_action(theme: str) -> str:
        mapping = {
            "茶餐厅文化": "喝奶茶",
            "天星小轮": "坐渡轮",
            "庙街夜市": "逛夜市",
            "叮叮车": "乘电车",
            "浅草下町": "散步",
            "秋叶原": "逛动漫店",
            "筑地市场": "吃寿司",
            "涩谷": "逛街",
            "左岸咖啡": "喝咖啡",
            "卢浮宫": "参观美术馆",
            "埃菲尔铁塔": "看风景",
            "蒙马特": "画画",
            "伏见稻荷": "参拜神社",
            "岚山": "赏红叶",
            "祇园": "穿和服",
            "锦市场": "买食材",
            "外滩": "看夜景",
            "弄堂": "骑自行车",
            "小笼包": "吃小笼包",
            "豫园": "逛园林",
            "故宫": "游览故宫",
            "胡同": "逛胡同",
            "烤鸭": "吃烤鸭",
            "长城": "爬长城",
            "饮食": "品尝美食",
            "建筑地标": "参观地标",
            "民俗节庆": "参加节日",
            "非遗": "体验非遗",
            "自然景观": "欣赏风景",
            "历史人物": "了解历史",
            "产业符号": "体验文化",
        }
        return mapping.get(theme, "旅游")

    def _draw_one_large(self, prompt: str, output_path: str, seed: int | None = None, ref_image: str | None = None) -> None:
        if self.provider == "baidu":
            self._draw_baidu(prompt, output_path)
        elif self.provider == "alibaba":
            if ref_image:
                self._draw_alibaba_with_ref(prompt, output_path, ref_image, seed=seed)
            else:
                self._draw_alibaba(prompt, output_path, size="1024*1024", seed=seed)

    def _draw_one(self, prompt: str, output_path: str) -> None:
        if self.provider == "baidu":
            self._draw_baidu(prompt, output_path)
        elif self.provider == "alibaba":
            self._draw_alibaba(prompt, output_path)

    # ── 百度文心一格 ──────────────────────────────────────────────────────

    def _baidu_token(self) -> str:
        if self._access_token:
            return self._access_token
        r = requests.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={"grant_type": "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if "access_token" not in data:
            raise RuntimeError(f"百度 token 获取失败: {data}")
        self._access_token = data["access_token"]
        return self._access_token

    def _draw_baidu(self, prompt: str, output_path: str) -> None:
        token = self._baidu_token()
        r = requests.post(
            "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2img",
            params={"access_token": token},
            json={"prompt": prompt, "width": 512, "height": 512, "image_num": 1},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"百度绘图提交失败: {data}")

        # 轮询结果
        for _ in range(30):
            time.sleep(5)
            q = requests.post(
                "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImg",
                params={"access_token": token},
                json={"task_id": task_id},
                timeout=30,
            )
            q.raise_for_status()
            img_data = q.json().get("data", {})
            status = img_data.get("status")
            if status == 1:
                img_url = img_data.get("img")
                if not img_url:
                    raise RuntimeError(f"百度结果无图片 URL: {img_data}")
                self._download_image(img_url, output_path)
                return
            elif status == 4:
                raise RuntimeError(f"百度绘图失败: {img_data.get('fail_reason', '未知')}")
        raise RuntimeError(f"百度绘图超时: {prompt[:40]}")

    # ── 阿里通义万相 ──────────────────────────────────────────────────────

    def _draw_alibaba(self, prompt: str, output_path: str, size: str = "512*512", seed: int | None = None) -> None:
        params: dict[str, Any] = {"size": size, "n": 1}
        if seed is not None:
            params["seed"] = seed
        r = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json={
                "model": "wanx2.1-t2i-turbo",
                "input": {"prompt": prompt},
                "parameters": params,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        output = data.get("output")
        if not output:
            raise RuntimeError(f"阿里绘图失败: {data}")

        task_id = output.get("task_id")
        if task_id:
            for _ in range(30):
                time.sleep(3)
                q = requests.get(
                    f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30,
                )
                q.raise_for_status()
                task_data = q.json().get("output", {})
                status = task_data.get("task_status")
                if status == "SUCCEEDED":
                    results = task_data.get("results", [])
                    if results:
                        self._download_image(results[0]["url"], output_path)
                        return
                    raise RuntimeError(f"阿里结果为空: {task_data}")
                elif status == "FAILED":
                    raise RuntimeError(f"阿里绘图失败: {task_data.get('message', '未知')}")
            raise RuntimeError(f"阿里绘图超时: {prompt[:40]}")
        else:
            results = output.get("results", [])
            if results:
                self._download_image(results[0]["url"], output_path)
                return
            raise RuntimeError(f"阿里同步结果为空: {output}")

    def _draw_alibaba_with_ref(self, prompt: str, output_path: str, ref_image: str, seed: int | None = None) -> None:
        import base64
        ref_path = Path(ref_image)
        if not ref_path.exists():
            raise RuntimeError(f"参考图不存在: {ref_image}")
        img_b64 = base64.b64encode(ref_path.read_bytes()).decode()
        data_uri = f"data:image/{ref_path.suffix.lstrip('.')};base64,{img_b64}"

        params: dict[str, Any] = {"size": "1024*1024", "n": 1}
        if seed is not None:
            params["seed"] = seed

        r = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "wan2.7-image",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"image": data_uri},
                            {"text": prompt},
                        ],
                    }],
                },
                "parameters": params,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("output", {}).get("choices", [])
        if not choices:
            raise RuntimeError(f"阿里带参考绘图失败: {data}")
        img_url = None
        for c in choices:
            content = c.get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "image":
                    img_url = item["image"]
                    break
        if not img_url:
            raise RuntimeError(f"阿里结果无图片URL: {data}")
        self._download_image(img_url, output_path)

    @staticmethod
    def _download_image(url: str, output_path: str) -> None:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = __import__("PIL").Image.open(io.BytesIO(r.content))
        img = img.convert("RGBA")
        # 白底转透明
        arr = img.load()
        w, h = img.size
        bg = (255, 255, 255)
        threshold = 230
        for y in range(h):
            for x in range(w):
                rv, gv, bv, av = arr[x, y][:4]
                if rv > threshold and gv > threshold and bv > threshold:
                    arr[x, y] = (rv, gv, bv, 0)
        img.save(output_path, "PNG")
