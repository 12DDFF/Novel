"""Extract characters from chapter text using LLM."""

from __future__ import annotations

import logging

from src.core.llm_client import LLMClient
from src.models import Character

from .prompts import CHARACTER_EXTRACTION_PROMPT, CHARACTER_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)


def extract_characters(llm: LLMClient, chapter_text: str) -> list[Character]:
    """
    Extract all characters from a chapter using LLM analysis.

    Args:
        llm: LLM client instance.
        chapter_text: Full chapter text in Chinese.

    Returns:
        List of Character objects with names, aliases, descriptions, and roles.
    """
    prompt = CHARACTER_EXTRACTION_PROMPT.format(chapter_text=chapter_text)

    raw_characters = llm.chat_json(
        prompt=prompt,
        system=CHARACTER_EXTRACTION_SYSTEM,
        temperature=0.3,
    )

    if not isinstance(raw_characters, list):
        raise ValueError(f"Expected list of characters, got: {type(raw_characters)}")

    characters = []
    for raw in raw_characters:
        char = Character(
            name=raw.get("name", ""),
            aliases=raw.get("aliases", []),
            description=raw.get("description", ""),
            role=raw.get("role", "minor"),
        )
        if char.name:
            characters.append(char)

    logger.info(f"Extracted {len(characters)} characters from chapter text")
    return characters
