from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_vision_model: str
    openai_text_model: str
    search_provider: str
    search_api_key: str
    output_dir: Path


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        openai_text_model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
        search_provider=os.getenv("SEARCH_PROVIDER", "tavily"),
        search_api_key=os.getenv("SEARCH_API_KEY", ""),
        output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
    )
