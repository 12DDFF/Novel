"""Quick run: 末世财阀 narration with improved prompts."""
import logging, sys, json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

from src.narration.orchestrator import NarrationPipeline

# Load 10 full chapters
chapters = []
novel_dir = Path("data/projects/novel_7330513034248457278")
for i in range(1, 11):
    for pattern in [f"ch_{i:03d}/raw/chapter_{i:03d}.txt", f"narration_v2_full/raw/ch_{i:03d}.txt"]:
        p = novel_dir / pattern
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if len(text) > 500:
                chapters.append((i, text))
                break

print(f"Loaded {len(chapters)} chapters")

output_dir = novel_dir / "final_narration"
pipeline = NarrationPipeline()
result = pipeline.run(
    chapters=chapters,
    novel_id="7330513034248457278",
    output_dir=output_dir,
    target_minutes=8.0,
    target_scenes=30,
)

# Print
print("\n" + "=" * 60)
print(f"Characters: {len(result['bible'].characters)}")
print(f"Archetypes:")
for name, arch in result["archetype_map"].items():
    print(f"  {name} → {arch}")
print(f"\nScript: {len(result['script'])} chars, {len(result['scenes'])} scenes")
print(f"\nSaved to: {output_dir / 'narration_script.txt'}")

# Open it
import subprocess
subprocess.Popen(["start", "", str(output_dir / "narration_script.txt")], shell=True)
