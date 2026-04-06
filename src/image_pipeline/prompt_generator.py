"""
Image prompt generator.

Converts SceneAnalysis + VisualSheet into production English Flux prompts.
"""

from __future__ import annotations

import logging

from src.core.llm_client import LLMClient
from src.image_pipeline.prompts import IMAGE_PROMPT_SYSTEM, IMAGE_PROMPT_TEMPLATE
from src.image_pipeline.scene_analyzer import SceneAnalysis
from src.image_pipeline.visual_sheet import VisualSheet

logger = logging.getLogger(__name__)


class ImagePromptGenerator:
    """Generates English image prompts from scene analyses."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_batch(
        self,
        analyses: list[SceneAnalysis],
        visual_sheet: VisualSheet,
        sentences: list[str] | None = None,
    ) -> list[str]:
        """
        Generate image prompts for a batch of analyzed scenes.

        Each sentence gets the FULL surrounding scene context so the LLM
        can write specific, concrete prompts.

        Args:
            analyses: Scene analyses from SceneAnalyzer.
            visual_sheet: Visual descriptions for characters/creatures.
            sentences: Full list of sentences for context window.

        Returns:
            List of English image prompts (one per analysis).
        """
        if not analyses:
            return []

        sentences = sentences or [a.sentence for a in analyses]
        prompts = []
        prev_background = ""

        for i, analysis in enumerate(analyses):
            # Build scene context: 3 sentences before + current + 3 after
            idx = analysis.sentence_index
            context_start = max(0, idx - 3)
            context_end = min(len(sentences), idx + 4)
            scene_text = "\n".join(sentences[context_start:context_end])

            # Build character visuals for THIS scene (prefer archetype names)
            char_visuals = []
            seen = set()
            for name in analysis.characters_archetype + analysis.characters_present + analysis.creatures_present:
                if name in seen:
                    continue
                seen.add(name)
                entity = visual_sheet.get_entity(name)
                if entity:
                    has_ref = "[REF]" if visual_sheet.get_reference(name) else "[NO REF]"
                    char_visuals.append(
                        f"- {name} {has_ref}: {entity.visual_description_en}"
                    )

            prompt_input = IMAGE_PROMPT_TEMPLATE.format(
                scene_text=scene_text,
                current_sentence=analysis.sentence,
                character_visuals="\n".join(char_visuals) if char_visuals else "(no characters visible)",
                location=analysis.location or "unknown",
                mood=analysis.mood,
                previous_background=prev_background or "(first scene)",
                location_changed=analysis.location_changed,
            )

            try:
                raw = self.llm.chat_json(
                    prompt=prompt_input,
                    system=IMAGE_PROMPT_SYSTEM,
                    temperature=0.4,
                )
                if isinstance(raw, dict):
                    prompt_text = raw.get("image_prompt", "")
                elif isinstance(raw, str):
                    prompt_text = raw
                else:
                    prompt_text = ""

                if prompt_text:
                    prompt_text = self._sanitize_prompt(prompt_text)
                    prompts.append(prompt_text)
                    # Track background for continuity
                    if analysis.location:
                        prev_background = analysis.background_description or analysis.location
                else:
                    prompts.append(self._fallback_prompt(analysis))

            except Exception as e:
                logger.warning("Prompt gen failed for sentence %d: %s", idx, e)
                prompts.append(self._fallback_prompt(analysis))

            if (i + 1) % 10 == 0:
                logger.info("  Generated %d/%d prompts", i + 1, len(analyses))

        return prompts

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        """Remove violent/gory content that triggers Flux safety filters."""
        import re
        replacements = {
            # Gore/blood
            r"blood\w*": "red energy",
            r"bleed\w*": "injured",
            r"gore\w*": "damage",
            r"gory": "intense",
            r"bloody": "intense",
            r"splatter\w*": "burst",
            # Killing/death
            r"kill\w*": "defeat",
            r"murder\w*": "confront",
            r"decapitat\w*": "strike down",
            r"slash\w* ?(throat|neck)": "strike at opponent",
            r"stab\w*": "strike",
            r"slash\w*": "swing at",
            r"sever\w*": "cut through",
            r"dismember\w*": "overpower",
            r"corpse\w*": "fallen figure",
            r"dead bod\w*": "fallen figure",
            # Zombie gore
            r"rotting skin": "pale grey skin",
            r"rotting": "decayed",
            r"exposed (teeth|bone|flesh)": "menacing appearance",
            # General violence
            r"throat": "chest",
            r"brain\w* (splatter|burst|out)": "dramatic impact",
        }
        for pattern, replacement in replacements.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        return prompt

    @staticmethod
    def _fallback_prompt(analysis: SceneAnalysis) -> str:
        """Generate a basic fallback prompt from analysis data."""
        mood_lighting = {
            "tense": "dramatic side lighting, cool blue tones",
            "action": "bright directional light, high contrast",
            "joyful": "warm sunlight, golden hour",
            "melancholy": "overcast diffused light, muted tones",
            "romantic": "soft warm backlight, bokeh",
            "dramatic": "strong volumetric light, god rays",
            "mysterious": "dim ambient light, fog",
            "peaceful": "soft morning light, warm pastels",
            "horror": "harsh underlighting, deep shadows",
            "humorous": "bright even lighting, warm tones",
        }
        lighting = mood_lighting.get(analysis.mood, "natural ambient lighting")
        return (
            f"anime illustration, {analysis.camera_suggestion.replace('_', ' ')}, "
            f"scene depicting {analysis.key_action or 'a dramatic moment'}, "
            f"{lighting}, "
            f"natural hand proportions, correct human anatomy"
        )
