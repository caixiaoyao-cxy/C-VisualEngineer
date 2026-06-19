from mapgen.rag.search import search_culture_elements


class DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"results": [{"title": "杭州西湖", "url": "https://example.com", "content": "西湖文化景观。"}]}


class DummyClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        assert "tavily" in url
        assert "杭州" in json["query"]
        return DummyResponse()


def test_search_culture_elements_tavily(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "test-key")
    monkeypatch.setattr("mapgen.rag.search.httpx.Client", DummyClient)
    result = search_culture_elements([{"name": "杭州"}], {"provider": "tavily"})
    assert result["results"][0]["place_name"] == "杭州"
    assert result["results"][0]["sources"][0]["url"] == "https://example.com"
