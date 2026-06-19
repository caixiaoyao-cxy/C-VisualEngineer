from pathlib import Path

import pytest

from mapgen.vision import contours

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_recognize_place_names_with_mocked_vision_model(tmp_path, monkeypatch):
    image_path = tmp_path / "map.png"
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Suzhou", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(image_path), image)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_recognize(self, image_path, model, prompt):
        return {
            "places": [
                {
                    "name": "苏州",
                    "type_guess": "市",
                    "confidence": "0.92",
                    "evidence": "地图中出现苏州相关文字",
                }
            ]
        }

    monkeypatch.setattr(contours.OpenAICompatibleClient, "recognize_places_from_image", fake_recognize)

    result = contours.recognize_place_names(str(image_path))

    assert result["places"] == [
        {
            "name": "苏州",
            "type_guess": "市",
            "confidence": 0.92,
            "evidence": "地图中出现苏州相关文字",
        }
    ]
