from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from mapgen.config import get_settings
from mapgen.llm import OpenAICompatibleClient


def extract_map_contours(image_path: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    import cv2
    import numpy as np

    image = _read_image(path, cv2, np)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    output_dir = Path(opts.get("output_dir") or get_settings().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    max_contours = int(opts.get("max_contours", 5))
    min_area_ratio = float(opts.get("min_area_ratio", 0.01))
    epsilon_ratio = float(opts.get("epsilon_ratio", 0.01))
    detection_mode = str(opts.get("detection_mode", "auto"))

    contours, used_mode, boundary_mask = _find_contours(image, opts, detection_mode, cv2, np)
    min_area = width * height * min_area_ratio
    results: list[dict[str, Any]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        points = [[int(point[0][0]), int(point[0][1])] for point in approx]
        results.append(
            {
                "points": points,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": area,
                "confidence": min(1.0, area / float(width * height)),
            }
        )
        if len(results) >= max_contours:
            break

    run_id = opts.get("run_id") or uuid4().hex[:10]
    mask_path = output_dir / f"{path.stem}_{run_id}_mask.png"
    boundary_mask_path = output_dir / f"{path.stem}_{run_id}_boundary_mask.png"
    overlay_path = output_dir / f"{path.stem}_{run_id}_overlay.png"
    mask = np.zeros((height, width), dtype=np.uint8)
    overlay = image.copy()
    if results:
        drawable = [np.array(item["points"], dtype=np.int32).reshape((-1, 1, 2)) for item in results]
        cv2.drawContours(mask, drawable, -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(overlay, drawable, -1, (0, 0, 255), thickness=3)
    if boundary_mask is None:
        boundary_mask = mask
    _write_image(mask_path, mask, cv2)
    _write_image(boundary_mask_path, boundary_mask, cv2)
    _write_image(overlay_path, overlay, cv2)

    return {
        "image_path": str(path.resolve()),
        "image_size": {"width": int(width), "height": int(height)},
        "detection_mode": used_mode,
        "contours": results,
        "artifacts": {
            "mask_path": str(mask_path.resolve()),
            "boundary_mask_path": str(boundary_mask_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
        },
    }


def recognize_place_names(
    image_path: str,
    contour_result: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts = options or {}
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    settings = get_settings()
    model = opts.get("model") or settings.openai_vision_model
    client = OpenAICompatibleClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    prompt = opts.get("prompt") or _place_recognition_prompt(contour_result)
    response = client.recognize_places_from_image(path, model=model, prompt=prompt)
    places = response.get("places", [])
    normalized = [_normalize_place(place) for place in places if isinstance(place, dict)] if isinstance(places, list) else []
    return {"image_path": str(path.resolve()), "places": normalized, "raw_response": response}


def analyze_map(image_path: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    contour_result = extract_map_contours(image_path, opts.get("contour_options"))
    place_result = recognize_place_names(image_path, contour_result, opts.get("recognition_options"))
    return {
        "image_path": contour_result["image_path"],
        "contours": contour_result["contours"],
        "places": place_result["places"],
        "artifacts": contour_result["artifacts"],
    }


def _find_contours(image: Any, opts: dict[str, Any], detection_mode: str, cv2: Any, np: Any) -> tuple[list[Any], str, Any | None]:
    if detection_mode in {"auto", "color_boundary", "orange_boundary"}:
        color_contours, color_mask = _find_orange_boundary_contours(image, opts, cv2, np)
        image_area = image.shape[0] * image.shape[1]
        largest_area = max((float(cv2.contourArea(contour)) for contour in color_contours), default=0.0)
        if detection_mode != "auto" or largest_area >= image_area * float(opts.get("color_min_area_ratio", 0.01)):
            return color_contours, "color_boundary", color_mask
    if detection_mode not in {"auto", "edge", "canny"}:
        raise ValueError("detection_mode must be auto, color_boundary, orange_boundary, edge, or canny.")
    edge_contours, edge_mask = _find_edge_contours(image, opts, cv2, np)
    return edge_contours, "edge", edge_mask


def _find_orange_boundary_contours(image: Any, opts: dict[str, Any], cv2: Any, np: Any) -> tuple[list[Any], Any]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(opts.get("hsv_lower", [8, 90, 90]), dtype=np.uint8)
    upper = np.array(opts.get("hsv_upper", [30, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel_size = int(opts.get("color_morph_kernel", opts.get("morph_kernel", 5)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(opts.get("color_morph_iterations", 2)))
    dilated = cv2.dilate(closed, kernel, iterations=int(opts.get("color_dilate_iterations", 1)))
    found = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(found[0] if len(found) == 2 else found[1]), dilated


def _find_edge_contours(image: Any, opts: dict[str, Any], cv2: Any, np: Any) -> tuple[list[Any], Any]:
    blur_kernel = int(opts.get("blur_kernel", 5))
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, int(opts.get("canny_low", 50)), int(opts.get("canny_high", 150)))
    kernel_size = int(opts.get("morph_kernel", 5))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(opts.get("morph_iterations", 2)))
    found = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(found[0] if len(found) == 2 else found[1]), closed


def _read_image(path: Path, cv2: Any, np: Any) -> Any:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: Any, cv2: Any) -> None:
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    encoded.tofile(str(path))


def _place_recognition_prompt(contour_result: dict[str, Any] | None) -> str:
    contour_context = json.dumps(contour_result or {}, ensure_ascii=False)[:4000]
    return (
        "请识别这张地图中的地名。只返回 JSON 对象，格式为 "
        "{\"places\":[{\"name\":\"...\",\"type_guess\":\"省/市/县/景区/未知\","
        "\"confidence\":0.0,\"evidence\":\"简短依据\"}]}。"
        "不要返回 Markdown。轮廓检测上下文如下："
        f"{contour_context}"
    )


def _normalize_place(place: dict[str, Any]) -> dict[str, Any]:
    confidence = place.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "name": str(place.get("name", "")).strip(),
        "type_guess": str(place.get("type_guess", "未知")).strip() or "未知",
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": str(place.get("evidence", "")).strip(),
    }
