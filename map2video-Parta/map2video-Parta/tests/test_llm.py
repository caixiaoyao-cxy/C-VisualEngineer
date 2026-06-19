from mapgen.llm import parse_json_object


def test_parse_json_object_from_fenced_block():
    data = parse_json_object('```json\n{"places":[{"name":"杭州"}]}\n```')
    assert data["places"][0]["name"] == "杭州"


def test_parse_json_object_embedded_text():
    data = parse_json_object('结果如下：{"inventory": []}')
    assert data == {"inventory": []}
