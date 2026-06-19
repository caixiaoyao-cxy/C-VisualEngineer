from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP  # type: ignore

__all__ = ["FastMCP"]
