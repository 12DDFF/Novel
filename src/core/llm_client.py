"""
LLM client wrapper. Uses OpenAI-compatible API (works with DeepSeek, OpenAI, etc.)
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: LLMConfig | None = None):
        if config is None:
            config = LLMConfig()
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text response.

        Args:
            model: Override the default model for this call (e.g. "deepseek-reasoner").
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        model: str | None = None,
    ) -> dict | list:
        """
        Send a chat request expecting JSON output.
        Extracts JSON from the response even if wrapped in markdown code blocks.

        Args:
            model: Override the default model for this call (e.g. "deepseek-reasoner").
        """
        if "JSON" not in system:
            system = system + "\nYou must respond with valid JSON only. No markdown, no explanation."

        raw = self.chat(prompt, system=system, temperature=temperature, max_tokens=max_tokens, model=model)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract JSON from text, handling markdown code blocks."""
        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding first [ or { and matching to last ] or }
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")
