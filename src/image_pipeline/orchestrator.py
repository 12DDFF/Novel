"""
Image pipeline orchestrator.

Wires sentence splitting → scene analysis → prompt generation → reference selection → image generation.
Processes one episode at a time with sliding window context carry-over.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.llm_client import LLMClient
from src.image_gen.base import BaseImageGenerator, ImageResult
from src.image_pipeline.prompt_generator import ImagePromptGenerator
from src.image_pipeline.ref_selector import select_reference
from src.image_pipeline.scene_analyzer import SceneAnalyzer, SceneAnalysis
from src.image_pipeline.sentence_splitter import split_narration
from src.image_pipeline.visual_sheet import VisualSheet
from src.narration.bible import StoryBible

logger = logging.getLogger(__name__)


class ImagePipelineResult:
    """Result of processing one episode."""

    def __init__(self):
        self.sentences: list[str] = []
        self.analyses: list[SceneAnalysis] = []
        self.prompts: list[str] = []
        self.references: list[str | None] = []
        self.images: list[ImageResult] = []
        self.carryover_sentences: list[str] = []  # for next episode


class ImagePipelineOrchestrator:
    """Orchestrates the full image generation pipeline."""

    def __init__(
        self,
        llm: LLMClient,
        image_gen: BaseImageGenerator,
        visual_sheet: VisualSheet,
        bible: StoryBible,
        archetype_map: dict[str, str],
    ):
        self.image_gen = image_gen
        self.visual_sheet = visual_sheet
        self.analyzer = SceneAnalyzer(llm, bible, archetype_map)
        self.prompt_gen = ImagePromptGenerator(llm)

    def process_episode(
        self,
        episode_text: str,
        episode_num: int,
        output_dir: Path,
        previous_sentences: list[str] | None = None,
    ) -> ImagePipelineResult:
        """
        Process one episode: split → analyze → prompt → reference → generate.

        Args:
            episode_text: Raw episode narration text.
            episode_num: Episode number (for file naming).
            output_dir: Where to save generated images.
            previous_sentences: Last N sentences from previous episode (context carry-over).

        Returns:
            ImagePipelineResult with all intermediate and final data.
        """
        result = ImagePipelineResult()
        output_dir = Path(output_dir)
        episode_dir = output_dir / f"ep_{episode_num:02d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Split narration into sentences
        logger.info("Episode %d: Splitting narration...", episode_num)
        result.sentences = split_narration(episode_text)
        logger.info("  %d sentences", len(result.sentences))

        if not result.sentences:
            return result

        # Step 2: Analyze scenes with sliding window context
        logger.info("Episode %d: Analyzing scenes...", episode_num)
        result.analyses = self.analyzer.analyze_all(
            result.sentences,
            previous_sentences=previous_sentences,
        )
        logger.info("  %d analyses", len(result.analyses))

        # Step 2b: Guardrail — ensure all detected characters have reference images
        self._ensure_references(result.analyses)

        # Step 3: Generate image prompts
        logger.info("Episode %d: Generating image prompts...", episode_num)
        result.prompts = self.prompt_gen.generate_batch(
            result.analyses,
            self.visual_sheet,
            sentences=result.sentences,
        )
        logger.info("  %d prompts", len(result.prompts))

        # Step 4: Select references + generate images
        logger.info("Episode %d: Generating %d images...", episode_num, len(result.analyses))
        for i, (analysis, prompt) in enumerate(zip(result.analyses, result.prompts)):
            # Select reference
            ref = select_reference(analysis, self.visual_sheet)
            result.references.append(ref)

            # Generate image
            img_path = str(episode_dir / f"img_{i:03d}.png")
            try:
                img_result = self.image_gen.generate(
                    prompt=prompt,
                    output_path=img_path,
                    reference_image=ref,
                )
                result.images.append(img_result)
            except Exception as e:
                logger.warning("Image generation failed for sentence %d: %s", i, e)
                result.images.append(ImageResult(image_path="", prompt=prompt))

            if (i + 1) % 10 == 0:
                logger.info("  Generated %d/%d images", i + 1, len(result.analyses))

        # Set carryover for next episode
        result.carryover_sentences = result.sentences[-SceneAnalyzer.BACKWARD_WINDOW:]

        logger.info("Episode %d: Done — %d images generated", episode_num, len(result.images))
        return result

    def _ensure_references(self, analyses: list[SceneAnalysis]) -> None:
        """Guardrail: generate master reference images for any character missing one.

        Scans all analyses, finds characters/creatures without references,
        and generates them via Flux Dev before proceeding to scene generation.
        """
        # Collect all unique entities across all scenes
        missing: dict[str, str] = {}  # name → visual_description
        for analysis in analyses:
            for name in analysis.characters_present + analysis.creatures_present:
                if name in missing:
                    continue
                ref = self.visual_sheet.get_reference(name)
                if ref:
                    continue
                entity = self.visual_sheet.get_entity(name)
                if entity and entity.visual_description_en:
                    missing[name] = entity.visual_description_en

        if not missing:
            return

        logger.info("Generating %d missing reference images...", len(missing))
        for name, visual_desc in missing.items():
            entity = self.visual_sheet.get_entity(name)
            if not entity:
                continue

            # Determine save path
            if entity.entity_type == "creature":
                ref_dir = Path(self.visual_sheet.novel_id) if self.visual_sheet.novel_id else Path(".")
                # Use the visual sheet's existing path pattern or create new
                ref_path = entity.reference_image_path or str(
                    Path("shared_creatures") / f"{name}.png"
                )
            else:
                ref_path = entity.reference_image_path or str(
                    Path("shared_characters") / f"{name}.png"
                )

            # Make path absolute if needed
            if not Path(ref_path).is_absolute():
                # Try to find a reasonable base dir from existing refs
                for _, e in self.visual_sheet.entities.items():
                    if e.reference_image_path and Path(e.reference_image_path).exists():
                        base = Path(e.reference_image_path).parent.parent
                        if entity.entity_type == "creature":
                            ref_path = str(base / "shared_creatures" / f"{name}.png")
                        else:
                            ref_path = str(base / "shared_characters" / f"{name}.png")
                        break

            Path(ref_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info("  Generating reference: %s → %s", name, Path(ref_path).name)
            try:
                self.image_gen.generate_character_sheet(visual_desc, ref_path)
                entity.reference_image_path = ref_path
                logger.info("  Done: %s", name)
            except Exception as e:
                logger.warning("  Failed to generate reference for %s: %s", name, e)

    def process_all_episodes(
        self,
        episodes: list[str],
        output_dir: Path,
    ) -> list[ImagePipelineResult]:
        """
        Process all episodes with context carry-over between them.

        Args:
            episodes: List of episode narration texts.
            output_dir: Base output directory.

        Returns:
            List of ImagePipelineResult, one per episode.
        """
        output_dir = Path(output_dir)
        results = []
        carryover = None

        for ep_num, ep_text in enumerate(episodes, start=1):
            result = self.process_episode(
                ep_text, ep_num, output_dir,
                previous_sentences=carryover,
            )
            results.append(result)
            carryover = result.carryover_sentences

        return results
