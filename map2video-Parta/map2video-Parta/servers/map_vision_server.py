from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapgen.fastmcp_compat import FastMCP
from mapgen.vision import analyze_map as analyze_map_impl
from mapgen.vision import extract_map_contours as extract_map_contours_impl
from mapgen.vision import recognize_place_names as recognize_place_names_impl

mcp = FastMCP("map-vision-server")


@mcp.tool()
def extract_map_contours(image_path: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract map contours with OpenCV and write mask/overlay artifacts."""
    return extract_map_contours_impl(image_path, options)


@mcp.tool()
def recognize_place_names(
    image_path: str,
    contour_result: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recognize place names in a map image with an OpenAI-compatible vision model."""
    return recognize_place_names_impl(image_path, contour_result, options)


@mcp.tool()
def analyze_map(image_path: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run contour extraction and place-name recognition in one call."""
    return analyze_map_impl(image_path, options)


if __name__ == "__main__":
    mcp.run()
