"""
Narration pipeline v2 orchestrator.

Ties together: Harvester → Bible Builder → Archetype Assigner → Narrator v2
into a single end-to-end pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.core.config import PipelineConfig
from src.core.llm_client import LLMClient
from src.narration.archetype import ArchetypeAssigner
from src.narration.bible import BibleBuilder, StoryBible
from src.narration.harvester import CharacterHarvester
from src.narration.narrator_v2 import NarrationManifest, NarratorV2

logger = logging.getLogger(__name__)


class NarrationPipeline:
    """End-to-end narration pipeline v2."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        reasoning_model: str | None = None,  # None = use same model as config
        creative_model: str | None = None,  # None = use default from config
    ):
        if config is None:
            config = PipelineConfig.load(
                env_path=Path(__file__).parent.parent.parent / ".env"
            )
        self.config = config
        self.llm = LLMClient(config.llm)
        self.harvester = CharacterHarvester()
        self.bible_builder = BibleBuilder(
            self.llm, self.harvester, model_override=reasoning_model
        )
        self.archetype_assigner = ArchetypeAssigner(
            self.llm, model_override=reasoning_model
        )
        self.narrator = NarratorV2(self.llm)
        self.creative_model = creative_model

    def run(
        self,
        chapters: list[tuple[int, str]],
        novel_id: str,
        output_dir: Path,
        bible: StoryBible | None = None,
        manifest: NarrationManifest | None = None,
        target_minutes: float = 3.0,
        target_scenes: int = 15,
    ) -> dict:
        """
        Run the full pipeline on a batch of chapters.

        Args:
            chapters: List of (chapter_num, chapter_text).
            novel_id: Novel identifier.
            output_dir: Where to save Bible, manifest, and script.
            bible: Existing Bible (None = start fresh).
            manifest: Existing manifest (None = start fresh).
            target_minutes: Target video duration.
            target_scenes: Target scene count.

        Returns:
            Dict with: bible, manifest, script, scenes, video_record, harvested, archetype_map
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load state
        if bible is None:
            bible_path = output_dir / "story_bible.json"
            if bible_path.exists():
                bible = StoryBible.load(bible_path)
                logger.info("Loaded existing Bible (last ch: %d)", bible.last_processed_chapter)
            else:
                bible = StoryBible(novel_id=novel_id)

        if manifest is None:
            manifest_path = output_dir / "narration_manifest.json"
            if manifest_path.exists():
                manifest = NarrationManifest.load(manifest_path)
                logger.info("Loaded existing manifest (%d videos)", len(manifest.videos))
            else:
                manifest = NarrationManifest(novel_id=novel_id)

        # ── Step 1: Harvest characters (regex, instant) ──────────────────────
        logger.info("Step 1: Harvesting characters from %d chapters...", len(chapters))
        all_harvested = self.harvester.harvest_novel(chapters)
        logger.info("  Found %d characters via regex", len(all_harvested))
        for h in all_harvested[:10]:
            logger.info("    %s (freq=%d)", h.name, h.frequency)

        # ── Step 1b: Seed Bible with top harvested characters ────────────
        # The LLM sometimes skips the protagonist (assumes already in Bible).
        # Pre-seed the top characters from the harvester to prevent this.
        from src.narration.bible import CharacterBible
        from src.narration.harvester import _split_name
        for h in all_harvested[:10]:
            if h.name not in bible.characters and h.frequency >= 5:
                surname, given = _split_name(h.name)
                bible.characters[h.name] = CharacterBible(
                    name=h.name,
                    aliases=h.aliases,
                    surname=surname,
                    role="",  # LLM will fill this in
                    description="",
                    first_appeared=min(h.chapter_appearances.keys()) if h.chapter_appearances else 1,
                    last_appeared=max(h.chapter_appearances.keys()) if h.chapter_appearances else 1,
                    tier="active",
                )
                logger.info("  Pre-seeded character: %s (freq=%d)", h.name, h.frequency)

        # ── Step 2: Build Bible in batches (5 chapters per LLM call) ────────
        batch_size = 5
        unprocessed = [
            (ch_num, text) for ch_num, text in chapters
            if ch_num > bible.last_processed_chapter
        ]

        if unprocessed:
            total_batches = (len(unprocessed) + batch_size - 1) // batch_size
            logger.info("Step 2: Building Story Bible (%d chapters in %d batches)...",
                        len(unprocessed), total_batches)

            for i in range(0, len(unprocessed), batch_size):
                batch = unprocessed[i:i + batch_size]
                batch_num = i // batch_size + 1
                ch_range = f"{batch[0][0]}-{batch[-1][0]}"
                logger.info("  Batch %d/%d (chapters %s)...", batch_num, total_batches, ch_range)

                # Harvest characters for this batch
                batch_harvested = self.harvester.harvest_novel(batch)

                bible = self.bible_builder.build_batch(
                    bible, batch, harvested=batch_harvested
                )
                # Checkpoint after each batch
                bible.save(output_dir / "story_bible.json")
                logger.info("    Bible now has %d characters, %d world facts",
                            len(bible.characters), len(bible.world))
        else:
            logger.info("Step 2: Bible already up to date (last ch: %d)",
                        bible.last_processed_chapter)

        logger.info("  Bible has %d characters, %d world facts",
                     len(bible.characters), len(bible.world))

        # ── Step 3: Assign archetypes ────────────────────────────────────────
        logger.info("Step 3: Assigning archetypes...")
        archetype_map = self.archetype_assigner.assign(
            bible, all_harvested, locked=manifest.locked_archetypes
        )
        logger.info("  Assignments:")
        for name, arch in archetype_map.items():
            logger.info("    %s → %s", name, arch)

        # ── Step 4: Generate narration script ────────────────────────────────
        logger.info("Step 4: Generating narration script...")
        chapter_texts = [text for _, text in chapters]
        chapter_nums = [num for num, _ in chapters]

        result = self.narrator.generate_script(
            chapters_text=chapter_texts,
            chapter_numbers=chapter_nums,
            bible=bible,
            archetype_map=archetype_map,
            manifest=manifest,
            target_minutes=target_minutes,
            target_scenes=target_scenes,
        )

        # ── Step 5: Save everything ──────────────────────────────────────────
        logger.info("Step 5: Saving results...")

        # Update manifest
        manifest.add_video(result["video_record"])
        manifest.save(output_dir / "narration_manifest.json")

        # Save script
        with open(output_dir / "narration_script.txt", "w", encoding="utf-8") as f:
            f.write(result["script"])

        # Save scenes
        with open(output_dir / "scenes.json", "w", encoding="utf-8") as f:
            json.dump(result["scenes"], f, ensure_ascii=False, indent=2)

        # Save archetype map
        with open(output_dir / "archetype_map.json", "w", encoding="utf-8") as f:
            json.dump(archetype_map, f, ensure_ascii=False, indent=2)

        # Save harvested characters
        with open(output_dir / "harvested_characters.json", "w", encoding="utf-8") as f:
            json.dump(
                [{"name": h.name, "surname": h.surname, "frequency": h.frequency,
                  "aliases": h.aliases, "chapters": h.chapter_appearances}
                 for h in all_harvested],
                f, ensure_ascii=False, indent=2,
            )

        logger.info("Done! Script: %d chars, %d scenes",
                     len(result["script"]), len(result["scenes"]))

        result["bible"] = bible
        result["manifest"] = manifest
        result["harvested"] = all_harvested
        return result
