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
    ) -> list[str]:
        """
        Generate image prompts for a batch of analyzed scenes.

        Args:
            analyses: Scene analyses from SceneAnalyzer.
            visual_sheet: Visual descriptions for characters/creatures.

        Returns:
            List of English image prompts (one per analysis).
        """
        if not analyses:
            return []

        # Build character visual descriptions
        all_chars = set()
        for a in analyses:
            all_chars.update(a.characters_present)
            all_chars.update(a.creatures_present)

        char_visuals = []
        for name in all_chars:
            entity = visual_sheet.get_entity(name)
            if entity:
                has_ref = "YES" if visual_sheet.get_reference(name) else "NO"
                char_visuals.append(
                    f"- {name}: {entity.visual_description_en} [has reference image: {has_ref}]"
                )

        # Build scene descriptions
        import json
        scenes_data = []
        for a in analyses:
            scenes_data.append({
                "sentence_index": a.sentence_index,
                "sentence": a.sentence,
                "characters": a.characters_present,
                "location": a.location,
                "location_changed": a.location_changed,
                "mood": a.mood,
                "camera": a.camera_suggestion,
                "creatures": a.creatures_present,
                "key_action": a.key_action,
                "background": a.background_description,
            })

        prompt = IMAGE_PROMPT_TEMPLATE.format(
            character_visuals="\n".join(char_visuals) if char_visuals else "(no characters)",
            scenes=json.dumps(scenes_data, ensure_ascii=False, indent=2),
        )

        try:
            raw = self.llm.chat_json(
                prompt=prompt,
                system=IMAGE_PROMPT_SYSTEM,
                temperature=0.3,
            )

            prompts = [""] * len(analyses)
            items = raw if isinstance(raw, list) else raw.get("prompts", raw.get("scenes", []))
            for item in items:
                if isinstance(item, dict):
                    idx = item.get("sentence_index", 0)
                    prompt_text = item.get("image_prompt", "")
                    # Map to position in analyses list
                    for i, a in enumerate(analyses):
                        if a.sentence_index == idx:
                            prompts[i] = prompt_text
                            break

            # Fill any gaps with fallback prompts
            for i, p in enumerate(prompts):
                if not p:
                    prompts[i] = self._fallback_prompt(analyses[i])

            return prompts

        except Exception as e:
            logger.warning("Prompt generation failed: %s — using fallbacks", e)
            return [self._fallback_prompt(a) for a in analyses]

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
