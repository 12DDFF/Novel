"""
Full end-to-end test: scrape chapters → harvest → bible → archetypes → narrate.

Usage:
    PYTHONIOENCODING=utf-8 python run_full_test.py --novel-id 7330513034248457278 --chapters 20
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Full narration pipeline test")
    parser.add_argument("--novel-id", "-n", default="5")
    parser.add_argument("--chapters", "-c", type=int, default=50, help="Number of chapters to process")
    parser.add_argument("--skip-scrape", action="store_true", help="Use already-scraped chapters")
    parser.add_argument("--platform", "-p", default="hetushu", choices=["fanqie", "hetushu"])
    parser.add_argument("--target-minutes", type=float, default=3.0)
    parser.add_argument("--target-scenes", type=int, default=15)
    args = parser.parse_args()

    novel_id = args.novel_id
    num_chapters = args.chapters
    output_dir = Path(f"data/projects/novel_{novel_id}/narration_v2_full")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Scrape chapters ──────────────────────────────────────────────
    chapters = []

    if not args.skip_scrape:
        logger.info("Scraping %d chapters from novel %s (%s)...", num_chapters, novel_id, args.platform)

        if args.platform == "hetushu":
            from src.scraper.hetushu import HetushuScraper
            scraper = HetushuScraper(rate_limit=1.0)
        else:
            from src.scraper.fanqie import FanqieScraper
            scraper = FanqieScraper(rate_limit=1.5)

        try:
            chapter_list = scraper.get_chapter_list(novel_id)
            logger.info("Novel has %d chapters total", len(chapter_list))

            to_scrape = min(num_chapters, len(chapter_list))
            for i, ch_info in enumerate(chapter_list[:to_scrape]):
                ch_num = i + 1
                ch_path = raw_dir / f"ch_{ch_num:03d}.txt"

                if ch_path.exists():
                    text = ch_path.read_text(encoding="utf-8")
                    logger.info("  Chapter %d: cached (%d chars)", ch_num, len(text))
                else:
                    logger.info("  Scraping chapter %d: %s ...", ch_num, ch_info.title)
                    content = scraper.download_chapter(ch_info.chapter_id)
                    text = content.text
                    ch_path.write_text(text, encoding="utf-8")
                    logger.info("  Chapter %d: %d chars", ch_num, len(text))

                chapters.append((ch_num, text))
        finally:
            scraper.close()
    else:
        # Load from existing files
        logger.info("Loading already-scraped chapters...")
        # Check narration_v2_full/raw first, then fall back to ch_XXX/raw
        for i in range(1, num_chapters + 1):
            ch_path = raw_dir / f"ch_{i:03d}.txt"
            if not ch_path.exists():
                # Try the original pipeline location
                alt_path = Path(f"data/projects/novel_{novel_id}/ch_{i:03d}/raw/chapter_{i:03d}.txt")
                if alt_path.exists():
                    ch_path = alt_path
                else:
                    logger.info("  No more chapters found after %d", i - 1)
                    break
            text = ch_path.read_text(encoding="utf-8")
            chapters.append((i, text))
            logger.info("  Loaded chapter %d: %d chars", i, len(text))

    if not chapters:
        logger.error("No chapters found!")
        return

    logger.info("\nTotal: %d chapters, %d total chars\n",
                len(chapters), sum(len(t) for _, t in chapters))

    # ── Step 1-5: Run narration pipeline ─────────────────────────────────────
    from src.narration.orchestrator import NarrationPipeline

    pipeline = NarrationPipeline()
    result = pipeline.run(
        chapters=chapters,
        novel_id=novel_id,
        output_dir=output_dir,
        target_minutes=args.target_minutes,
        target_scenes=args.target_scenes,
    )

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nChapters processed: {len(chapters)}")
    print(f"Characters harvested: {len(result['harvested'])}")
    for h in result["harvested"][:15]:
        print(f"  {h.name} (freq={h.frequency}, chapters={len(h.chapter_appearances)})")

    print(f"\nBible characters: {len(result['bible'].characters)}")
    for name, char in result["bible"].characters.items():
        rels = []
        for k, v in char.relationships.items():
            if v:
                rels.append(f"{k}:{v[-1].state}")
        rel_str = f" | Rels: {', '.join(rels)}" if rels else ""
        print(f"  {name} [{char.role}] - {char.arc_status}{rel_str}")

    print(f"\nWorld facts: {len(result['bible'].world)}")
    for fact in result["bible"].world[:10]:
        print(f"  [{fact.category}] {fact.fact}")

    print(f"\nArchetype map:")
    for name, arch in result["archetype_map"].items():
        print(f"  {name} → {arch}")

    print(f"\nScript length: {len(result['script'])} chars")
    print(f"Scenes: {len(result['scenes'])}")

    print(f"\nCliffhanger: {result['video_record'].cliffhanger}")
    print(f"\nAudience knows ({len(result['video_record'].audience_knows)}):")
    for fact in result["video_record"].audience_knows[:5]:
        print(f"  - {fact}")

    print(f"\n{'=' * 70}")
    print(f"Files saved to: {output_dir}")
    print(f"  narration_script.txt  — the full script")
    print(f"  story_bible.json      — complete character/world knowledge")
    print(f"  archetype_map.json    — character → nickname mapping")
    print(f"  narration_manifest.json — audience state tracking")
    print(f"  scenes.json           — parsed scene breakdown")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
