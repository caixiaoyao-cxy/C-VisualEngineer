from __future__ import annotations

from typing import Any

import httpx

from mapgen.config import get_settings

DEFAULT_CATEGORIES = ["非遗", "历史人物", "建筑地标", "民俗节庆", "饮食", "自然景观", "产业符号"]


class SearchConfigurationError(RuntimeError):
    pass


def search_culture_elements(places: list[dict[str, Any]], options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    settings = get_settings()
    provider = (opts.get("provider") or settings.search_provider).lower()
    api_key = opts.get("api_key") or settings.search_api_key
    max_results = int(opts.get("max_results", 5))
    categories = opts.get("categories") or DEFAULT_CATEGORIES
    timeout = float(opts.get("timeout", 30.0))

    if not api_key:
        raise SearchConfigurationError("SEARCH_API_KEY is required for web search.")

    items = []
    for place in places:
        name = str(place.get("name", "")).strip()
        if not name:
            continue
        query = opts.get("query_template", "{place} 文化 元素 非遗 历史 民俗 地标 美食").format(place=name)
        snippets = _run_search(provider, api_key, query, max_results=max_results, timeout=timeout)
        items.append({"place_name": name, "query": query, "snippets": snippets, "sources": _sources_from_snippets(snippets)})
    return {"results": items}


def _run_search(provider: str, api_key: str, query: str, max_results: int, timeout: float) -> list[dict[str, str]]:
    if provider == "tavily":
        return _search_tavily(api_key, query, max_results, timeout)
    if provider == "serpapi":
        return _search_serpapi(api_key, query, max_results, timeout)
    raise SearchConfigurationError(f"Unsupported SEARCH_PROVIDER: {provider}. Use tavily or serpapi.")


def _search_tavily(api_key: str, query: str, max_results: int, timeout: float) -> list[dict[str, str]]:
    payload = {"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
    data = response.json()
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", "")),
        }
        for item in data.get("results", [])[:max_results]
    ]


def _search_serpapi(api_key: str, query: str, max_results: int, timeout: float) -> list[dict[str, str]]:
    params = {"api_key": api_key, "engine": "google", "q": query, "num": max_results, "hl": "zh-cn"}
    with httpx.Client(timeout=timeout) as client:
        response = client.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()
    data = response.json()
    organic = data.get("organic_results", [])
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("link", "")),
            "content": str(item.get("snippet", "")),
        }
        for item in organic[:max_results]
    ]


def _sources_from_snippets(snippets: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources = []
    for snippet in snippets:
        url = snippet.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"title": snippet.get("title", ""), "url": url})
    return sources
