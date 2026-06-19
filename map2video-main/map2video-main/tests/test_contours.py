from pathlib import Path

import pytest

from mapgen.vision.contours import extract_map_contours

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_extract_map_contours_writes_artifacts(tmp_path):
    image_path = tmp_path / "map.png"
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (160, 160), (0, 0, 0), thickness=3)
    cv2.imwrite(str(image_path), image)

    result = extract_map_contours(str(image_path), {"output_dir": str(tmp_path), "min_area_ratio": 0.005})

    assert result["contours"]
    assert Path(result["artifacts"]["mask_path"]).exists()
    assert Path(result["artifacts"]["overlay_path"]).exists()
