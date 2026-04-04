"""
Scene Context Analyzer with sliding window context.

Analyzes each narration sentence to determine: who is present, where we are,
what mood/camera to use. Resolves pronouns using backward/forward context.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from src.core.llm_client import LLMClient
from src.narration.bible import StoryBible
from src.image_pipeline.prompts import SCENE_ANALYSIS_PROMPT, SCENE_ANALYSIS_SYSTEM

logger = logging.getLogger(__name__)


class SceneAnalysis(BaseModel):
    """Analysis result for a single narration sentence."""

    sentence_index: int = 0
    sentence: str = ""
    characters_present: list[str] = Field(default_factory=list)
    characters_archetype: list[str] = Field(default_factory=list)
    location: str = ""
    location_changed: bool = False
    mood: str = "dramatic"
    camera_suggestion: str = "medium_shot"
    creatures_present: list[str] = Field(default_factory=list)
    key_action: str = ""
    background_description: str = ""


class SceneAnalyzer:
    """Analyzes narration sentences for image generation using sliding window context."""

    BACKWARD_WINDOW = 5
    FORWARD_WINDOW = 3

    def __init__(
        self,
        llm: LLMClient,
        bible: StoryBible,
        archetype_map: dict[str, str],
    ):
        self.llm = llm
        self.bible = bible
        self.archetype_map = archetype_map
        self._reverse_map = {v: k for k, v in archetype_map.items()}

    def analyze_all(
        self,
        sentences: list[str],
        previous_sentences: list[str] | None = None,
        batch_size: int = 8,
    ) -> list[SceneAnalysis]:
        """
        Analyze all sentences with sliding window context.

        Args:
            sentences: List of narration sentences for this episode.
            previous_sentences: Last N sentences from previous episode (for context carry-over).
            batch_size: How many sentences to analyze per LLM call.

        Returns:
            List of SceneAnalysis, one per sentence.
        """
        previous_sentences = previous_sentences or []
        all_analyses: list[SceneAnalysis] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Build sliding window context
            backward = self._get_backward_context(
                sentences, i, previous_sentences
            )
            forward = self._get_forward_context(sentences, i + len(batch))

            logger.info(
                "Analyzing sentences %d-%d (backward=%d, forward=%d)...",
                i, i + len(batch) - 1, len(backward), len(forward),
            )

            batch_analyses = self._analyze_batch(batch, backward, forward, start_index=i)
            all_analyses.extend(batch_analyses)

        # Post-process: fill in archetype names
        for analysis in all_analyses:
            analysis.characters_archetype = [
                self.archetype_map.get(name, name)
                for name in analysis.characters_present
            ]

        return all_analyses

    def _get_backward_context(
        self,
        sentences: list[str],
        current_index: int,
        previous_sentences: list[str],
    ) -> list[str]:
        """Get backward context sentences."""
        # From current episode
        start = max(0, current_index - self.BACKWARD_WINDOW)
        from_current = sentences[start:current_index]

        # If not enough, supplement from previous episode
        needed = self.BACKWARD_WINDOW - len(from_current)
        if needed > 0 and previous_sentences:
            from_prev = previous_sentences[-needed:]
            return from_prev + from_current

        return from_current

    def _get_forward_context(
        self,
        sentences: list[str],
        after_index: int,
    ) -> list[str]:
        """Get forward context sentences."""
        end = min(len(sentences), after_index + self.FORWARD_WINDOW)
        return sentences[after_index:end]

    def _analyze_batch(
        self,
        sentences: list[str],
        backward: list[str],
        forward: list[str],
        start_index: int = 0,
    ) -> list[SceneAnalysis]:
        """Analyze a batch of sentences via LLM."""
        # Build character/creature lists from Bible
        char_list = "\n".join(
            f"- {name} ({self.archetype_map.get(name, '')}): {char.description[:60]}"
            for name, char in self.bible.characters.items()
            if char.tier == "active"
        ) or "(no characters)"

        creature_list = "\n".join(
            f"- {fact.fact[:60]}"
            for fact in self.bible.world
            if any(kw in fact.fact for kw in ["丧尸", "魔", "兽", "妖", "怪", "虫", "变异"])
        ) or "(no creatures)"

        # Format sentences with indices
        sent_text = "\n".join(
            f"[{start_index + i}] {s}" for i, s in enumerate(sentences)
        )
        back_text = "\n".join(backward) if backward else "(episode start)"
        fwd_text = "\n".join(forward) if forward else "(episode end)"

        prompt = SCENE_ANALYSIS_PROMPT.format(
            character_list=char_list,
            creature_list=creature_list,
            backward_context=back_text,
            sentences=sent_text,
            forward_context=fwd_text,
        )

        try:
            raw = self.llm.chat_json(
                prompt=prompt,
                system=SCENE_ANALYSIS_SYSTEM,
                temperature=0.2,
            )

            if isinstance(raw, list):
                analyses = []
                for j, item in enumerate(raw):
                    if isinstance(item, dict):
                        item["sentence_index"] = start_index + j
                        if j < len(sentences):
                            item["sentence"] = sentences[j]
                        analyses.append(SceneAnalysis.model_validate(item))
                return analyses

            if isinstance(raw, dict) and "scenes" in raw:
                # LLM wrapped it
                return [SceneAnalysis.model_validate(item) for item in raw["scenes"]]

        except Exception as e:
            logger.warning("Scene analysis failed: %s — using defaults", e)

        # Fallback: return basic analyses
        return [
            SceneAnalysis(
                sentence_index=start_index + j,
                sentence=s,
                mood="dramatic",
                camera_suggestion="medium_shot",
            )
            for j, s in enumerate(sentences)
        ]
