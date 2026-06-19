from __future__ import annotations

import io
import math
import urllib.parse
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

from mapgen.config import get_settings


def get_osm_contour(place_name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch administrative boundary polygon for a place via Nominatim.

    Returns a dict with mask_path, contour metadata, and fallback status.
    """
    opts = options or {}
    output_dir = Path(opts.get("output_dir") or get_settings().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(opts.get("output_width", 800)), int(opts.get("output_height", 600))
    user_agent = opts.get("user_agent", "Map2Video/0.1 (https://github.com/Pankeyi88/map2video)")

    geo = _geocode_with_polygon(place_name, user_agent)
    lat = geo.get("lat")
    lon = geo.get("lon")
    display_name = geo.get("display_name")
    polygon = geo.get("polygon")

    if polygon:
        mask_img = _polygon_to_mask(polygon, w, h)
        mask_path = output_dir / f"{_sanitize(place_name)}_{uuid4().hex[:10]}_mask.png"
        mask_img.save(mask_path)
        return {
            "place_name": display_name or place_name,
            "mask_path": str(mask_path.resolve()),
            "width": w,
            "height": h,
            "contour_points": [],
            "source": "nominatim_boundary",
            "fallback": False,
        }

    if lat is not None and lon is not None:
        result = _extract_contour_from_tiles(lat, lon, display_name, place_name, w, h, output_dir, user_agent)
        if result is not None:
            return result

    mask = _fallback_mask(w, h)
    mask_path = output_dir / f"{_sanitize(place_name)}_{uuid4().hex[:10]}_mask.png"
    mask.save(mask_path)
    return {
        "place_name": place_name,
        "mask_path": str(mask_path.resolve()),
        "width": w,
        "height": h,
        "contour_points": [],
        "source": "fallback",
        "fallback": True,
        "fallback_reason": "No boundary polygon or tile contour available",
    }


def _geocode_with_polygon(place_name: str, user_agent: str) -> dict[str, Any]:
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={urllib.parse.quote(place_name)}&format=json&limit=1&polygon_geojson=1"
    )
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
        if r.status_code == 200 and r.json():
            data = r.json()[0]
            result = {
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
                "display_name": data.get("display_name"),
                "polygon": data.get("geojson"),
            }
            return result
    except (requests.RequestException, ValueError, IndexError):
        pass
    return {"lat": None, "lon": None, "display_name": None, "polygon": None}


def _polygon_to_mask(polygon: dict[str, Any], w: int, h: int) -> Image.Image:
    """Render a GeoJSON polygon into a binary mask image.

    Polygon can be Polygon or MultiPolygon. Coordinates are [lon, lat].
    We auto-scale to fill 88% of the canvas and center.
    """
    coords_sets: list[list[tuple[float, float]]] = []
    if polygon["type"] == "Polygon":
        coords_sets.append(polygon["coordinates"][0])
    elif polygon["type"] == "MultiPolygon":
        for poly in polygon["coordinates"]:
            coords_sets.append(poly[0])

    all_lons, all_lats = [], []
    for ring in coords_sets:
        for lon, lat in ring:
            all_lons.append(lon)
            all_lats.append(lat)

    if not all_lons:
        return _fallback_mask(w, h)

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    span_lon = max_lon - min_lon if max_lon > min_lon else 1
    span_lat = max_lat - min_lat if max_lat > min_lat else 1

    target_w = int(w * 0.95)
    target_h = int(h * 0.95)
    scale = min(target_w / span_lon, target_h / span_lat)
    ox = (w - int(span_lon * scale)) // 2
    oy = (h - int(span_lat * scale)) // 2

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    for ring in coords_sets:
        pts = []
        for lon, lat in ring:
            px = int((lon - min_lon) * scale) + ox
            py = int((max_lat - lat) * scale) + oy
            pts.append((px, py))
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)

    arr = np.array(img)
    # smooth edges
    return Image.fromarray(arr).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))


def _extract_contour_from_tiles(
    lat: float, lon: float, display_name: str | None, place_name: str,
    w: int, h: int, output_dir: Path, user_agent: str,
) -> dict[str, Any] | None:
    """Fallback: extract landmass contour from OSM raster tiles."""
    zoom = 12
    tile_size = 256
    canvas_size = 400

    n = 2.0 ** zoom
    cx = int((lon + 180) / 360 * n)
    cy = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)

    half = int(math.ceil(canvas_size / tile_size / 2))
    composite = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    loaded = 0
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{cx + dx}/{cy + dy}.png"
            try:
                r = requests.get(url, headers={"User-Agent": user_agent}, timeout=10)
                if r.status_code == 200:
                    tile = Image.open(io.BytesIO(r.content))
                    px = (dx + half) * tile_size
                    py = (dy + half) * tile_size
                    composite.paste(tile, (px, py))
                    loaded += 1
            except requests.RequestException:
                continue

    if loaded == 0:
        return None

    import cv2

    gray = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours_found, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_found:
        return None

    largest = max(contours_found, key=cv2.contourArea)
    raw_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.drawContours(raw_mask, [largest], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(raw_mask > 0)
    cxo, cyo = int(xs.mean()), int(ys.mean())
    bw = int(xs.max() - xs.min()) + 20
    bh = int(ys.max() - ys.min()) + 20

    scale = min(w * 0.95 / bw, h * 0.95 / bh) if bw > 0 and bh > 0 else 1.0
    ox = (w - int(bw * scale)) // 2
    oy = (h - int(bh * scale)) // 2

    mask_img = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask_img)
    for y, x in zip(ys, xs):
        nx = int((x - (cxo - bw // 2)) * scale) + ox
        ny = int((y - (cyo - bh // 2)) * scale) + oy
        if 0 <= nx < w and 0 <= ny < h:
            md.point((nx, ny), fill=255)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(7))

    mask_arr_check = np.array(mask_img)
    filled_ratio = np.sum(mask_arr_check > 0) / (w * h)
    if filled_ratio < 0.15:
        return None

    mask_path = output_dir / f"{_sanitize(place_name)}_{uuid4().hex[:10]}_mask.png"
    mask_img.save(mask_path)

    approx = cv2.approxPolyDP(largest, 0.01 * cv2.arcLength(largest, True), True)
    points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

    return {
        "place_name": display_name or place_name,
        "mask_path": str(mask_path.resolve()),
        "width": w,
        "height": h,
        "contour_points": points,
        "source": "openstreetmap_tiles",
        "fallback": False,
    }


def _fallback_mask(w: int, h: int) -> Image.Image:
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.42), int(h * 0.38)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    d.ellipse([cx - rx + 14, cy - ry + 14, cx + rx - 14, cy + ry - 14], fill=0)
    for angle in [0, 90, 180, 270]:
        a = math.radians(angle)
        px, py = cx + (rx - 8) * math.cos(a), cy + (ry - 8) * math.sin(a)
        d.ellipse([px - 20, py - 20, px + 20, py + 20], fill=255)
    arr = np.array(img)
    return Image.fromarray(arr).filter(ImageFilter.MaxFilter(5))


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower() or "place"
