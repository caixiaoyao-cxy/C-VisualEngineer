from __future__ import annotations

import os
from typing import Any

from mapgen.culture.data import CULTURE_THEMES


class CulturePlanner:
    """Agent 1: 文化主题策划。

    优先从硬编码数据中取 4 个主题×6 个物品。
    若地名不在硬编码列表中，自动联网搜索 + AI 生成（需 SEARCH_API_KEY）。
    """

    def __init__(self, place: str, search_api_key: str = "", openai_api_key: str = ""):
        key = place.lower().strip()
        self.themes = CULTURE_THEMES.get(key)
        if self.themes is not None:
            self._validate()
        else:
            self.themes = self._generate_dynamic(place, search_api_key, openai_api_key)

    def _validate(self) -> None:
        if len(self.themes) != 4:
            raise RuntimeError(f"数据异常: 应有 4 个主题, 实际 {len(self.themes)}")
        for t in self.themes:
            if len(t["items"]) != 6:
                raise RuntimeError(
                    f"数据异常: 主题 '{t['theme']}' 应有 6 项, 实际 {len(t['items'])}"
                )

    def _generate_dynamic(self, place: str, search_api_key: str, openai_api_key: str) -> list[dict]:
        if not search_api_key:
            raise RuntimeError(
                f"不支持的地名: '{place}'，且未配置 SEARCH_API_KEY。\n"
                f"请前往 https://tavily.com/ 免费注册获取 API Key，"
                f"然后设置到 .env 的 SEARCH_API_KEY=。\n"
                f"或从硬编码地点中选择: {', '.join(CULTURE_THEMES.keys())}"
            )

        from mapgen.config import load_dotenv
        load_dotenv()
        from mapgen.rag.search import search_culture_elements
        from mapgen.rag.inventory import build_culture_inventory

        places = [{"name": place}]
        opts = {}
        if openai_api_key:
            from mapgen.llm import OpenAICompatibleClient
            opts["model"] = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

        raw = search_culture_elements(places, {"api_key": search_api_key})
        inv = build_culture_inventory(places, raw, opts)
        items = inv.get("inventory", [])

        KNOWN = self._known_items_for(place)
        if not items and not KNOWN:
            raise RuntimeError(f"未能通过联网搜索获取到 '{place}' 的文化元素，请检查地名或网络连接。")

        grouped: dict[str, list[str]] = {}
        for item in items:
            cat = item.get("category", "其他")
            name = self._clean_name(item.get("element_name", ""))
            if name:
                grouped.setdefault(cat, []).append(name)

        for cat, names in grouped.items():
            seen: set[str] = set()
            unique = []
            for n in names:
                short = n[:12]
                if short not in seen:
                    seen.add(short)
                    unique.append(short)
            grouped[cat] = unique

        priority = ["饮食", "建筑地标", "民俗节庆", "非遗", "自然景观", "历史人物", "产业符号"]
        used_cats = [c for c in priority if c in grouped]
        other_cats = [c for c in grouped if c not in priority]
        sorted_cats = (used_cats + other_cats)[:4]

        themes = []
        known_idx = 0
        for cat in sorted_cats:
            names: list[str] = []
            for n in grouped[cat]:
                if len(names) >= 6:
                    break
                names.append(n)
            while len(names) < 6 and known_idx < len(KNOWN):
                names.append(KNOWN[known_idx])
                known_idx += 1
            while len(names) < 6:
                names.append(f"{place}{cat[:2]}")
            themes.append({"theme": cat, "items": names})

        while len(themes) < 4:
            rest: list[str] = []
            while len(rest) < 6 and known_idx < len(KNOWN):
                rest.append(KNOWN[known_idx])
                known_idx += 1
            while len(rest) < 6:
                rest.append(f"{place}风光")
            themes.append({"theme": f"{place}印象", "items": rest})

        return themes

    @staticmethod
    def _clean_name(raw: str) -> str:
        import re
        s = raw.strip()
        s = re.sub(r'^\[.*?\]\s*', '', s)
        s = re.sub(r'\s*[—\-|]\s*(byFood|Trip\.com|携程|马蜂窝|小红书|大众点评).*$', '', s)
        s = re.sub(r'\s*[—\-|]\s*\S+\s*(美食|景点|攻略|推荐|攻略)$', '', s)
        s = re.sub(r'^感受一下.*?(?=\s|$)', '', s)
        s = re.sub(r'\s*发现\d+个.*$', '', s)
        s = s.strip().rstrip('，。！？,.:;!?')
        if not s:
            return ""
        if len(s) > 30:
            s = s[:30]
        return s

    @staticmethod
    def _known_items_for(place: str) -> list[str]:
        common = {
            "hangzhou": ["西湖", "龙井茶", "断桥", "雷峰塔", "灵隐寺", "东坡肉",
                         "西湖醋鱼", "宋城", "河坊街", "西溪湿地", "钱塘江", "丝绸"],
            "shenzhen": ["锦绣中华", "世界之窗", "华强北", "欢乐谷", "大梅沙",
                         "深圳湾", "莲花山", "地王大厦", "东门老街", "华侨城",
                         "盐田港", "红树林"],
            "guangzhou": ["广州塔", "珠江夜游", "沙面", "陈家祠", "越秀公园",
                          "白云山", "北京路", "石室圣心", "长隆", "上下九",
                          "广式早茶", "粤剧"],
            "chongqing": ["洪崖洞", "解放碑", "长江索道", "磁器口", "武隆",
                          "大足石刻", "朝天门", "李子坝", "山城步道", "火锅",
                          "小面", "南山"],
            "chengdu": ["大熊猫", "宽窄巷子", "锦里", "武侯祠", "杜甫草堂",
                        "都江堰", "青城山", "九眼桥", "春熙路", "火锅",
                        "串串", "盖碗茶"],
        }
        return common.get(place.lower().strip(), [])

    def plan(self) -> list[dict]:
        return self.themes

    def all_items(self) -> list[str]:
        items = []
        for t in self.themes:
            items.extend(t["items"])
        return items
