from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx


class LLMConfigurationError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 90.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def require_api_key(self) -> None:
        if not self.api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for LLM calls.")

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        self.require_api_key()
        payload = {"model": model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def recognize_places_from_image(self, image_path: str | Path, model: str, prompt: str) -> dict[str, Any]:
        image_url = image_to_data_url(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        raw = self.chat(model=model, messages=[{"role": "user", "content": content}], temperature=0.1)
        return parse_json_object(raw)

    def summarize_culture_inventory(self, model: str, prompt: str) -> dict[str, Any]:
        raw = self.chat(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return parse_json_object(raw)


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from model response.")
    return value
