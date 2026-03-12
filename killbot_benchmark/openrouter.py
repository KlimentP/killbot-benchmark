from __future__ import annotations

from base64 import b64encode
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from killbot_benchmark.config import ModelConfig, PromptConfig, RunSettings, ScenarioConfig, ToolConfig


logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, settings: RunSettings):
        try:
            from openai import OpenAI
            from httpx import Timeout
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "The `openai` package is required to run benchmark requests. Install project dependencies first."
            ) from exc

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        self._settings = settings
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.base_url,
            # Keep slow or stalled response bodies from holding the full batch open for minutes.
            timeout=Timeout(connect=5.0, read=90.0, write=30.0, pool=30.0),
        )

    def run_case(
        self,
        model: ModelConfig,
        prompt: PromptConfig,
        tool: ToolConfig,
        scenario: ScenarioConfig,
    ) -> Any:
        headers = {}
        if self._settings.http_referer:
            headers["HTTP-Referer"] = self._settings.http_referer
        if self._settings.x_title:
            headers["X-Title"] = self._settings.x_title

        logger.info(
            "Requesting model=%s prompt=%s tool=%s scenario=%s image=%s",
            model.id,
            prompt.id,
            tool.id,
            scenario.id,
            scenario.image_path,
        )

        return self._client.chat.completions.create(
            model=model.id,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
            messages=[
                {"role": "system", "content": prompt.text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._settings.user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _file_to_data_url(scenario.image_path)},
                        },
                    ],
                },
            ],
            tools=[tool.spec],
            tool_choice="auto",
            extra_headers=headers or None,
        )


def _file_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    payload = b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"
