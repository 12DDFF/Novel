"""
Full novel pipeline: scrape ALL chapters → build Bible → narrate in episodes.

Usage:
    PYTHONIOENCODING=utf-8 python run_full_novel.py --novel-id 5 --platform hetushu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full novel narration pipeline")
    parser.add_argument("--novel-id", "-n", default="5")
    parser.add_argument("--platform", "-p", default="hetushu", choices=["fanqie", "hetushu"])
    parser.add_argument("--max-chapters", type=int, default=None, help="Limit chapters (None=all)")
    parser.add_argument("--chapters-per-episode", type=int, default=50)
    parser.add_argument("--bible-batch-size", type=int, default=5)
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-bible", action="store_true", help="Use existing Bible")
    parser.add_argument("--target-minutes", type=float, default=5.0)
    parser.add_argument("--target-scenes", type=int, default=20)
    args = parser.parse_args()

    novel_id = args.novel_id
    output_dir = Path(f"data/projects/novel_{novel_id}/full_narration")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: SCRAPE ALL CHAPTERS
    # ═══════════════════════════════════════════════════════════════════════
    chapters = []

    if not args.skip_scrape:
        logger.info("═══ SCRAPING NOVEL %s ═══", novel_id)
        if args.platform == "hetushu":
            from src.scraper.hetushu import HetushuScraper
            scraper = HetushuScraper(rate_limit=0.8)
        else:
            from src.scraper.fanqie import FanqieScraper
            scraper = FanqieScraper(rate_limit=1.5)

        try:
            chapter_list = scraper.get_chapter_list(novel_id)
            total = len(chapter_list)
            if args.max_chapters:
                total = min(total, args.max_chapters)
            logger.info("Novel has %d chapters, scraping %d", len(chapter_list), total)

            for i, ch_info in enumerate(chapter_list[:total]):
                ch_num = i + 1
                ch_path = raw_dir / f"ch_{ch_num:03d}.txt"

                if ch_path.exists():
                    text = ch_path.read_text(encoding="utf-8")
                    if i % 50 == 0:
                        logger.info("  Chapter %d/%d: cached (%d chars)", ch_num, total, len(text))
                else:
                    content = scraper.download_chapter(ch_info.chapter_id)
                    text = content.text
                    ch_path.write_text(text, encoding="utf-8")
                    if i % 20 == 0:
                        logger.info("  Chapter %d/%d: %d chars - %s", ch_num, total, len(text), ch_info.title)

                if len(text) > 50:  # skip empty/broken chapters
                    chapters.append((ch_num, text))
        finally:
            scraper.close()
    else:
        logger.info("Loading cached chapters...")
        i = 1
        while True:
            ch_path = raw_dir / f"ch_{i:03d}.txt"
            if not ch_path.exists():
                break
            text = ch_path.read_text(encoding="utf-8")
            if len(text) > 50:
                chapters.append((i, text))
            i += 1
            if args.max_chapters and i > args.max_chapters:
                break

    logger.info("Loaded %d chapters (%d total chars)\n",
                len(chapters), sum(len(t) for _, t in chapters))

    if not chapters:
        logger.error("No chapters found!")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: HARVEST ALL CHARACTERS (instant)
    # ═══════════════════════════════════════════════════════════════════════
    from src.narration.harvester import CharacterHarvester
    from src.narration.bible import BibleBuilder, StoryBible
    from src.narration.archetype import ArchetypeAssigner
    from src.narration.narrator_v2 import NarrationManifest, NarratorV2
    from src.core.config import PipelineConfig
    from src.core.llm_client import LLMClient

    logger.info("═══ STEP 1: HARVESTING CHARACTERS ═══")
    harvester = CharacterHarvester()
    all_harvested = harvester.harvest_novel(chapters)
    logger.info("Found %d characters via regex", len(all_harvested))
    for h in all_harvested[:15]:
        logger.info("  %s (freq=%d, %d chapters)", h.name, h.frequency, len(h.chapter_appearances))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: BUILD STORY BIBLE (incremental, batched)
    # ═══════════════════════════════════════════════════════════════════════
    config = PipelineConfig.load(env_path=Path(__file__).parent / ".env")
    llm = LLMClient(config.llm)

    bible_path = output_dir / "story_bible.json"
    if args.skip_bible and bible_path.exists():
        bible = StoryBible.load(bible_path)
        logger.info("═══ STEP 2: LOADED EXISTING BIBLE (ch %d) ═══", bible.last_processed_chapter)
    else:
        bible = StoryBible(novel_id=novel_id)
        if bible_path.exists():
            bible = StoryBible.load(bible_path)
            logger.info("Resuming Bible from chapter %d", bible.last_processed_chapter)

        # Pre-seed Bible with top harvested characters (prevents protagonist being skipped)
        from src.narration.bible import CharacterBible
        from src.narration.harvester import _split_name
        for h in all_harvested[:15]:
            if h.name not in bible.characters and h.frequency >= 10:
                surname, given = _split_name(h.name)
                bible.characters[h.name] = CharacterBible(
                    name=h.name, aliases=h.aliases, surname=surname,
                    first_appeared=min(h.chapter_appearances.keys()) if h.chapter_appearances else 1,
                    last_appeared=max(h.chapter_appearances.keys()) if h.chapter_appearances else 1,
                    tier="active",
                )
                logger.info("  Pre-seeded: %s (freq=%d)", h.name, h.frequency)

        builder = BibleBuilder(llm, harvester)
        unprocessed = [(n, t) for n, t in chapters if n > bible.last_processed_chapter]

        if unprocessed:
            batch_size = args.bible_batch_size
            total_batches = (len(unprocessed) + batch_size - 1) // batch_size
            logger.info("═══ STEP 2: BUILDING BIBLE (%d chapters, %d batches) ═══",
                        len(unprocessed), total_batches)

            for i in range(0, len(unprocessed), batch_size):
                batch = unprocessed[i:i + batch_size]
                batch_num = i // batch_size + 1
                logger.info("  Batch %d/%d (ch %d-%d)...",
                            batch_num, total_batches, batch[0][0], batch[-1][0])

                batch_harvested = harvester.harvest_novel(batch)
                try:
                    bible = builder.build_batch(bible, batch, harvested=batch_harvested)
                except Exception as e:
                    logger.error("  Batch failed: %s — saving checkpoint and continuing", e)

                bible.save(bible_path)

                if batch_num % 10 == 0:
                    logger.info("  [Checkpoint] Bible: %d chars, %d facts, %d threads",
                                len(bible.characters), len(bible.world), len(bible.loose_threads))
        else:
            logger.info("═══ STEP 2: BIBLE UP TO DATE ═══")

    logger.info("Bible: %d characters, %d world facts\n", len(bible.characters), len(bible.world))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: ASSIGN ARCHETYPES
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("═══ STEP 3: ASSIGNING ARCHETYPES ═══")
    manifest_path = output_dir / "narration_manifest.json"
    if manifest_path.exists():
        manifest = NarrationManifest.load(manifest_path)
    else:
        manifest = NarrationManifest(novel_id=novel_id)

    assigner = ArchetypeAssigner(llm)
    archetype_map = assigner.assign(bible, all_harvested, locked=manifest.locked_archetypes)

    for name, arch in archetype_map.items():
        logger.info("  %s → %s", name, arch)

    with open(output_dir / "archetype_map.json", "w", encoding="utf-8") as f:
        json.dump(archetype_map, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: GENERATE NARRATION EPISODES
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n═══ STEP 4: GENERATING NARRATION ═══")
    narrator = NarratorV2(llm)
    ep_size = args.chapters_per_episode
    all_scripts = []

    for ep_start in range(0, len(chapters), ep_size):
        ep_chapters = chapters[ep_start:ep_start + ep_size]
        ep_num = ep_start // ep_size + 1
        ch_range = f"{ep_chapters[0][0]}-{ep_chapters[-1][0]}"

        logger.info("  Episode %d (chapters %s)...", ep_num, ch_range)

        result = narrator.generate_script(
            chapters_text=[t for _, t in ep_chapters],
            chapter_numbers=[n for n, _ in ep_chapters],
            bible=bible,
            archetype_map=archetype_map,
            manifest=manifest,
            target_minutes=args.target_minutes,
            target_scenes=args.target_scenes,
        )

        manifest.add_video(result["video_record"])
        manifest.save(manifest_path)

        ep_header = f"\n{'='*60}\n第{ep_num}集 (章节 {ch_range})\n{'='*60}\n"
        all_scripts.append(ep_header + result["script"])

        logger.info("    → %d chars, %d scenes", len(result["script"]), len(result["scenes"]))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: SAVE FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    full_script = "\n".join(all_scripts)
    script_path = output_dir / "full_narration_script.txt"
    script_path.write_text(full_script, encoding="utf-8")

    logger.info("\n" + "=" * 70)
    logger.info("DONE!")
    logger.info("=" * 70)
    logger.info("Chapters: %d", len(chapters))
    logger.info("Bible: %d characters, %d world facts", len(bible.characters), len(bible.world))
    logger.info("Episodes: %d", len(all_scripts))
    logger.info("Total script: %d chars", len(full_script))
    logger.info("")
    logger.info("Files:")
    logger.info("  %s  — FULL NARRATION SCRIPT", script_path)
    logger.info("  %s  — story bible", bible_path)
    logger.info("  %s  — archetype map", output_dir / "archetype_map.json")
    logger.info("  %s  — audience state", manifest_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
