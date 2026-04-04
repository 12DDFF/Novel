"""
Visual Character/Creature Sheet for image generation.

Maps every entity (characters, creatures, objects) to a fixed English visual
description and a master reference image for Flux Kontext Pro consistency.

Separate from the Story Bible — the Bible tracks plot, this tracks appearance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.llm_client import LLMClient
from src.image_gen.base import BaseImageGenerator
from src.narration.bible import StoryBible

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

VISUAL_DESCRIPTION_SYSTEM = """You are a character designer for anime-style illustration.
You convert Chinese character/creature descriptions into precise English visual descriptions
suitable for AI image generation. Focus on VISUAL details only: hair, eyes, build, clothing,
distinguishing features. Do NOT include personality or story info.
Always respond in valid JSON format."""

VISUAL_DESCRIPTION_PROMPT = """Convert these characters and creatures into English visual descriptions for AI image generation.

CHARACTERS (from story):
{characters}

CREATURES/ENTITIES (from world building):
{creatures}

For each entity, provide a concise English visual description (2-3 sentences) covering:
- Physical appearance: age, build, hair color/style, eye color, skin tone
- Clothing: what they typically wear
- Distinguishing features: scars, weapons, accessories
- For creatures: size, color, body type, notable features

Return as JSON:
{{
  "characters": [
    {{
      "name": "顾杀",
      "visual_description_en": "18-year-old Chinese male with short messy black hair and cold sharp eyes, lean athletic build, wearing a torn white school uniform stained with blood, carrying a fire axe over his shoulder"
    }}
  ],
  "creatures": [
    {{
      "name": "丧尸",
      "visual_description_en": "humanoid zombie with grey rotting skin, sunken glowing red eyes, torn dirty clothing, hunched posture with arms hanging forward, blood stains around mouth"
    }}
  ]
}}"""


# ── Models ────────────────────────────────────────────────────────────────────


class VisualEntity(BaseModel):
    """A visual description of a character, creature, or object."""

    name: str
    archetype: str = ""
    entity_type: str = "character"  # character, creature, object
    visual_description_en: str = ""
    reference_image_path: str | None = None


class VisualSheet(BaseModel):
    """Registry of all visual entities for a novel."""

    novel_id: str = ""
    entities: dict[str, VisualEntity] = Field(default_factory=dict)

    def get_entity(self, name: str) -> VisualEntity | None:
        """Get entity by name."""
        return self.entities.get(name)

    def get_reference(self, name: str) -> str | None:
        """Get reference image path for an entity, or None."""
        entity = self.entities.get(name)
        if entity and entity.reference_image_path:
            if Path(entity.reference_image_path).exists():
                return entity.reference_image_path
        return None

    def get_visual_description(self, name: str) -> str:
        """Get English visual description for an entity."""
        entity = self.entities.get(name)
        return entity.visual_description_en if entity else ""

    def all_characters(self) -> dict[str, VisualEntity]:
        """Get all character entities."""
        return {n: e for n, e in self.entities.items() if e.entity_type == "character"}

    def all_creatures(self) -> dict[str, VisualEntity]:
        """Get all creature entities."""
        return {n: e for n, e in self.entities.items() if e.entity_type == "creature"}

    def save(self, path: Path) -> None:
        """Save visual sheet to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> VisualSheet:
        """Load visual sheet from JSON."""
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


# ── Builder ───────────────────────────────────────────────────────────────────


class VisualSheetBuilder:
    """Builds a VisualSheet from Story Bible data."""

    def __init__(
        self,
        llm: LLMClient,
        image_gen: BaseImageGenerator | None = None,
    ):
        self.llm = llm
        self.image_gen = image_gen

    def build(
        self,
        bible: StoryBible,
        archetype_map: dict[str, str],
        output_dir: Path,
        generate_images: bool = True,
    ) -> VisualSheet:
        """
        Build visual sheet: LLM generates descriptions, optionally Flux generates references.

        Args:
            bible: Story Bible with character/world info.
            archetype_map: {original_name: archetype_name}.
            output_dir: Where to store reference images.
            generate_images: If True, generate master reference images via Flux.

        Returns:
            VisualSheet with all entities.
        """
        output_dir = Path(output_dir)
        char_dir = output_dir / "shared_characters"
        creature_dir = output_dir / "shared_creatures"
        char_dir.mkdir(parents=True, exist_ok=True)
        creature_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate visual descriptions via LLM
        logger.info("Generating visual descriptions for %d characters...", len(bible.characters))
        descriptions = self._generate_descriptions(bible)

        # Step 2: Build VisualSheet
        sheet = VisualSheet(novel_id=bible.novel_id)

        # Add characters
        for char_desc in descriptions.get("characters", []):
            name = char_desc.get("name", "")
            if not name:
                continue
            visual_en = char_desc.get("visual_description_en", "")
            ref_path = str(char_dir / f"{name}.png")

            sheet.entities[name] = VisualEntity(
                name=name,
                archetype=archetype_map.get(name, ""),
                entity_type="character",
                visual_description_en=visual_en,
                reference_image_path=ref_path if Path(ref_path).exists() else None,
            )

        # Add creatures
        for creature_desc in descriptions.get("creatures", []):
            name = creature_desc.get("name", "")
            if not name:
                continue
            visual_en = creature_desc.get("visual_description_en", "")
            ref_path = str(creature_dir / f"{name}.png")

            sheet.entities[name] = VisualEntity(
                name=name,
                archetype=name,
                entity_type="creature",
                visual_description_en=visual_en,
                reference_image_path=ref_path if Path(ref_path).exists() else None,
            )

        # Step 3: Generate master reference images
        if generate_images and self.image_gen:
            self._generate_reference_images(sheet, char_dir, creature_dir)

        return sheet

    def _generate_descriptions(self, bible: StoryBible) -> dict:
        """LLM call to generate English visual descriptions."""
        # Build character info
        char_lines = []
        for name, char in bible.characters.items():
            if char.tier == "retired":
                continue
            char_lines.append(f"- {name}: {char.description[:100]}")

        # Extract creatures from world facts
        creature_keywords = ["丧尸", "魔物", "变异", "妖", "兽", "怪", "龙", "虫"]
        creature_lines = []
        seen_creatures = set()
        for fact in bible.world:
            for kw in creature_keywords:
                if kw in fact.fact and kw not in seen_creatures:
                    creature_lines.append(f"- {fact.fact[:80]}")
                    seen_creatures.add(kw)

        if not creature_lines:
            creature_lines = ["- (no creatures mentioned)"]

        prompt = VISUAL_DESCRIPTION_PROMPT.format(
            characters="\n".join(char_lines) if char_lines else "(no characters)",
            creatures="\n".join(creature_lines),
        )

        try:
            result = self.llm.chat_json(
                prompt=prompt,
                system=VISUAL_DESCRIPTION_SYSTEM,
                temperature=0.3,
            )
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning("Failed to generate visual descriptions: %s", e)

        return {"characters": [], "creatures": []}

    def _generate_reference_images(
        self,
        sheet: VisualSheet,
        char_dir: Path,
        creature_dir: Path,
    ) -> None:
        """Generate master reference images for entities without one."""
        for name, entity in sheet.entities.items():
            if entity.reference_image_path and Path(entity.reference_image_path).exists():
                logger.info("Reference exists for %s, skipping", name)
                continue

            if not entity.visual_description_en:
                logger.warning("No visual description for %s, skipping image gen", name)
                continue

            if entity.entity_type == "character":
                ref_path = str(char_dir / f"{name}.png")
            else:
                ref_path = str(creature_dir / f"{name}.png")

            logger.info("Generating reference image for %s...", name)
            try:
                self.image_gen.generate_character_sheet(
                    entity.visual_description_en,
                    ref_path,
                )
                entity.reference_image_path = ref_path
            except Exception as e:
                logger.warning("Failed to generate reference for %s: %s", name, e)
