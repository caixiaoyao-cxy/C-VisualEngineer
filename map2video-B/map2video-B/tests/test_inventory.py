from mapgen.rag.inventory import build_culture_inventory, generate_report


def test_build_inventory_falls_back_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    search_results = {
        "results": [
            {
                "place_name": "苏州",
                "snippets": [{"title": "苏州园林", "url": "https://example.com", "content": "古典园林文化。"}],
                "sources": [{"title": "苏州园林", "url": "https://example.com"}],
            }
        ]
    }
    result = build_culture_inventory([{"name": "苏州"}], search_results)
    assert result["inventory"][0]["place_name"] == "苏州"
    assert result["inventory"][0]["sources"][0]["url"] == "https://example.com"


def test_generate_report_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    result = generate_report(
        {
            "inventory": [
                {
                    "place_name": "苏州",
                    "element_name": "苏州园林",
                    "category": "建筑地标",
                    "summary": "古典园林文化。",
                    "visual_keywords": ["粉墙黛瓦"],
                    "usage_suggestions": ["地图纹样"],
                    "confidence": 0.8,
                    "sources": [{"title": "来源", "url": "https://example.com"}],
                }
            ]
        },
        "markdown",
    )
    assert "苏州园林" in result["content"]
    assert result["path"].endswith(".md")
