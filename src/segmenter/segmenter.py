"""
Scene segmenter: splits chapter text into visual scenes using LLM.

Two-pass approach:
  1. Extract characters from the full chapter
  2. Segment the chapter into visual scenes with image prompts
"""

from __future__ import annotations

import logging

from src.core.llm_client import LLMClient
from src.models import Character, Mood, Scene, TransitionType

from .character_extractor import extract_characters
from .prompts import SCENE_SEGMENTATION_PROMPT, SCENE_SEGMENTATION_SYSTEM

logger = logging.getLogger(__name__)

# Map string mood values to Mood enum
_MOOD_MAP = {m.value: m for m in Mood}
_TRANSITION_MAP = {t.value: t for t in TransitionType}

# Max characters to send to LLM in one request
_MAX_CHUNK_SIZE = 6000
_CHUNK_OVERLAP = 500


class SceneSegmenter:
    """Segments novel chapters into visual scenes for video production."""

    def __init__(self, llm: LLMClient, art_style: str = "cinematic anime style"):
        self.llm = llm
        self.art_style = art_style

    def process_chapter(
        self,
        chapter_text: str,
        existing_characters: list[Character] | None = None,
    ) -> tuple[list[Character], list[Scene]]:
        """
        Full pipeline: extract characters then segment into scenes.

        Args:
            chapter_text: Full chapter text.
            existing_characters: Characters from previous chapters (for continuity).

        Returns:
            Tuple of (characters, scenes).
        """
        # Pass 1: Character extraction
        new_characters = extract_characters(self.llm, chapter_text)

        # Merge with existing characters
        characters = self._merge_characters(existing_characters or [], new_characters)

        # Pass 2: Scene segmentation
        scenes = self.segment_scenes(chapter_text, characters)

        return characters, scenes

    def segment_scenes(
        self,
        chapter_text: str,
        characters: list[Character],
    ) -> list[Scene]:
        """
        Segment chapter text into visual scenes.

        For long chapters exceeding LLM context, splits into overlapping chunks
        and processes sequentially.
        """
        character_profiles = self._format_character_profiles(characters)

        if len(chapter_text) <= _MAX_CHUNK_SIZE:
            return self._segment_chunk(chapter_text, character_profiles)

        # Split long chapters into overlapping chunks
        return self._segment_long_chapter(chapter_text, character_profiles)

    def _segment_chunk(
        self,
        text: str,
        character_profiles: str,
        sequence_offset: int = 0,
    ) -> list[Scene]:
        """Segment a single chunk of text into scenes."""
        prompt = SCENE_SEGMENTATION_PROMPT.format(
            chapter_text=text,
            character_profiles=character_profiles,
            art_style=self.art_style,
        )

        raw_scenes = self.llm.chat_json(
            prompt=prompt,
            system=SCENE_SEGMENTATION_SYSTEM,
            temperature=0.4,
            max_tokens=8192,
        )

        if not isinstance(raw_scenes, list):
            raise ValueError(f"Expected list of scenes, got: {type(raw_scenes)}")

        scenes = []
        for raw in raw_scenes:
            scene = Scene(
                sequence=raw.get("sequence", 0) + sequence_offset,
                narration_text=raw.get("narration_text", ""),
                visual_description=raw.get("visual_description", ""),
                characters_present=raw.get("characters_present", []),
                mood=_MOOD_MAP.get(raw.get("mood", "dramatic"), Mood.DRAMATIC),
                setting=raw.get("setting", ""),
                image_prompt=raw.get("image_prompt", ""),
                transition=_TRANSITION_MAP.get(raw.get("transition", "crossfade"), TransitionType.CROSSFADE),
            )
            # Estimate duration based on narration length (~4 chars/second for Chinese)
            scene.duration_estimate_seconds = max(5.0, len(scene.narration_text) / 4.0)
            scenes.append(scene)

        logger.info(f"Segmented text into {len(scenes)} scenes")
        return scenes

    def _segment_long_chapter(
        self,
        chapter_text: str,
        character_profiles: str,
    ) -> list[Scene]:
        """Split a long chapter into chunks and segment each."""
        chunks = self._split_into_chunks(chapter_text)
        all_scenes = []
        sequence_offset = 0

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            scenes = self._segment_chunk(chunk, character_profiles, sequence_offset)
            all_scenes.extend(scenes)
            sequence_offset = len(all_scenes)

        # Re-number sequences
        for i, scene in enumerate(all_scenes):
            scene.sequence = i + 1

        return all_scenes

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        """Split text into overlapping chunks, breaking at sentence boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + _MAX_CHUNK_SIZE

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at a sentence boundary
            search_start = end - 200
            best_break = end
            for punct in ["。", "！", "？", "\n"]:
                pos = text.rfind(punct, search_start, end)
                if pos > search_start:
                    best_break = pos + 1
                    break

            chunks.append(text[start:best_break])
            start = best_break - _CHUNK_OVERLAP

        return chunks

    @staticmethod
    def _format_character_profiles(characters: list[Character]) -> str:
        """Format characters into a string for the LLM prompt."""
        if not characters:
            return "No character information available."

        lines = []
        for char in characters:
            aliases = f" (also called: {', '.join(char.aliases)})" if char.aliases else ""
            lines.append(f"- {char.name}{aliases}: {char.description} [{char.role}]")
        return "\n".join(lines)

    @staticmethod
    def _merge_characters(
        existing: list[Character],
        new: list[Character],
    ) -> list[Character]:
        """
        Merge new characters into existing list.
        If a character with the same name exists, update their info.
        """
        merged = {c.name: c for c in existing}

        for new_char in new:
            if new_char.name in merged:
                old = merged[new_char.name]
                # Update with new info if it's more detailed
                if len(new_char.description) > len(old.description):
                    old.description = new_char.description
                # Merge aliases
                for alias in new_char.aliases:
                    if alias not in old.aliases:
                        old.aliases.append(alias)
            else:
                merged[new_char.name] = new_char

        return list(merged.values())
