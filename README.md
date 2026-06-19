# Map2Video FastMCP Servers

This project provides two Python FastMCP stdio servers:

- `map-vision-server`: extracts map contours with OpenCV and recognizes place names with an OpenAI-compatible vision model.
- `culture-rag-server`: searches the web for local culture elements and builds a structured inventory with an OpenAI-compatible text model.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item .env.example .env
```

Fill `.env` with API keys.

## Run Servers

```powershell
python servers/map_vision_server.py
python servers/culture_rag_server.py
```

Both commands run stdio MCP servers by default.

## Main Tools

### map-vision-server

- `extract_map_contours(image_path, options=None)`
- `recognize_place_names(image_path, contour_result=None, options=None)`
- `analyze_map(image_path, options=None)`

### culture-rag-server

- `search_culture_elements(places, options=None)`
- `build_culture_inventory(places, search_results=None, options=None)`
- `generate_report(inventory, format="markdown")`

## MCP Client Example

```json
{
  "mcpServers": {
    "map-vision-server": {
      "command": "python",
      "args": ["E:/pky/map2video/servers/map_vision_server.py"]
    },
    "culture-rag-server": {
      "command": "python",
      "args": ["E:/pky/map2video/servers/culture_rag_server.py"]
    }
  }
}
```
