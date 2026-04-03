"""End-to-end test of the narration pipeline v2 on real novel data."""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.narration.orchestrator import NarrationPipeline

# Load chapters
novel_dir = Path("data/projects/novel_7330513034248457278")
chapters = []
for i in range(1, 6):
    ch_file = novel_dir / f"ch_{i:03d}" / "raw" / f"chapter_{i:03d}.txt"
    if ch_file.exists():
        chapters.append((i, ch_file.read_text(encoding="utf-8")))
        print(f"Loaded chapter {i}: {len(chapters[-1][1])} chars")

print(f"\nTotal: {len(chapters)} chapters loaded\n")

# Run pipeline
output_dir = Path("data/projects/novel_7330513034248457278/narration_v2_test")
pipeline = NarrationPipeline()

result = pipeline.run(
    chapters=chapters,
    novel_id="7330513034248457278",
    output_dir=output_dir,
    target_minutes=3.0,
    target_scenes=15,
)

# Print results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nCharacters harvested: {len(result['harvested'])}")
for h in result["harvested"][:10]:
    print(f"  {h.name} (freq={h.frequency})")

print(f"\nBible characters: {len(result['bible'].characters)}")
for name, char in result["bible"].characters.items():
    rels = ", ".join(f"{k}:{v[-1].state}" for k, v in char.relationships.items() if v)
    print(f"  {name} [{char.role}] - {char.arc_status}")
    if rels:
        print(f"    Relationships: {rels}")

print(f"\nArchetype map:")
for name, arch in result["archetype_map"].items():
    print(f"  {name} → {arch}")

print(f"\nScenes: {len(result['scenes'])}")
for scene in result["scenes"][:3]:
    print(f"  [{scene['index']}] {scene['narration'][:80]}...")

print(f"\nVideo record:")
vr = result["video_record"]
print(f"  Chapters: {vr.chapters_covered}")
print(f"  Cliffhanger: {vr.cliffhanger}")
print(f"  Audience knows: {vr.audience_knows[:3]}")

print(f"\nFull script saved to: {output_dir / 'narration_script.txt'}")
print(f"Story Bible saved to: {output_dir / 'story_bible.json'}")
