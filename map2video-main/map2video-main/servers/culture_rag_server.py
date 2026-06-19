from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapgen.fastmcp_compat import FastMCP
from mapgen.rag import build_culture_inventory as build_culture_inventory_impl
from mapgen.rag import generate_report as generate_report_impl
from mapgen.rag import search_culture_elements as search_culture_elements_impl

mcp = FastMCP("culture-rag-server")


@mcp.tool()
def search_culture_elements(places: list[dict[str, Any]], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Search the web for local culture materials for recognized places."""
    return search_culture_elements_impl(places, options)


@mcp.tool()
def build_culture_inventory(
    places: list[dict[str, Any]],
    search_results: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a categorized culture-element inventory from search results."""
    return build_culture_inventory_impl(places, search_results, options)


@mcp.tool()
def generate_report(inventory: dict[str, Any] | list[dict[str, Any]], format: str = "markdown") -> dict[str, Any]:
    """Generate a JSON or Markdown culture inventory report under OUTPUT_DIR."""
    return generate_report_impl(inventory, format)


if __name__ == "__main__":
    mcp.run()
