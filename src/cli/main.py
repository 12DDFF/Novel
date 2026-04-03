"""
n2v CLI — Novel to Video pipeline.

Usage:
    python -m src.cli.main run --novel-id 7402200659753176126 --chapter 1
    python -m src.cli.main run --novel-id 7402200659753176126 --chapter 1 --placeholder
    python -m src.cli.main info --novel-id 7402200659753176126
    python -m src.cli.main chapters --novel-id 7402200659753176126
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Ensure ffmpeg is findable on Windows
_ffmpeg_path = r"C:\Users\zihao\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
if os.path.isdir(_ffmpeg_path) and _ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

app = typer.Typer(name="n2v", help="Novel to Video pipeline")

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

console = Console(highlight=False)


@app.command()
def narrate(
    novel_id: str = typer.Option(..., "--novel-id", "-n", help="Fanqie novel ID"),
    start_chapter: int = typer.Option(1, "--start", "-s", help="First chapter to include"),
    end_chapter: int = typer.Option(5, "--end", "-e", help="Last chapter to include"),
    target_minutes: float = typer.Option(3.0, "--minutes", "-m", help="Target video duration"),
    target_scenes: int = typer.Option(15, "--scenes", help="Target number of scenes"),
    placeholder: bool = typer.Option(False, "--placeholder", "-p", help="Use placeholder images"),
    verbose: bool = typer.Option(False, "--verbose", help="Show debug logs"),
):
    """Generate a 视频解说 style video from multiple chapters."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    from src.core.config import PipelineConfig
    config = PipelineConfig.load()

    project_dir = str(_project_root / "data" / "projects" / f"novel_{novel_id}" / f"narrate_{start_chapter}-{end_chapter}")

    from pathlib import Path
    from src.scraper.fanqie import FanqieScraper
    from src.core.llm_client import LLMClient
    from src.segmenter.narrator import VideoNarrator
    from src.tts.multi_voice import MultiVoiceNarrator, VoiceAssigner
    from src.subtitles.generator import SubtitleGenerator
    from src.image_gen.replicate_backend import ReplicateGenerator
    from src.tts.base import TTSResult
    import json, re

    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Scrape chapters ──────────────────────────────────
    console.print("\n[bold cyan][1/6] Scraping chapters...[/]")
    raw_dir = project_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    script_path = raw_dir / "narration_script.txt"
    scenes_path = raw_dir / "scenes.json"

    scraper = FanqieScraper(rate_limit=2.0)
    info = scraper.get_novel_info(novel_id)
    chapter_list = scraper.get_chapter_list(novel_id)
    console.print(f"  {info.title} — chapters {start_chapter}-{end_chapter}")

    chapters_text = []
    for i in range(start_chapter - 1, min(end_chapter, len(chapter_list))):
        ch = chapter_list[i]
        cache_path = raw_dir / f"ch_{i + 1:03d}.txt"
        if cache_path.exists():
            chapters_text.append(cache_path.read_text(encoding="utf-8"))
            console.print(f"  Ch {i + 1}: cached")
        else:
            content = scraper.download_chapter(ch.chapter_id)
            cache_path.write_text(content.text, encoding="utf-8")
            chapters_text.append(content.text)
            console.print(f"  Ch {i + 1}: {ch.title} ({len(content.text)} chars)")
    scraper.close()

    # ── Step 2: Generate narration script ─────────────────────────
    if not script_path.exists():
        console.print("\n[bold cyan][2/6] Generating 视频解说 script...[/]")
        llm = LLMClient(config.llm)
        narrator = VideoNarrator(llm)
        result = narrator.generate_script(
            chapters_text,
            target_minutes=target_minutes,
            target_scenes=target_scenes,
        )
        script_path.write_text(result["script"], encoding="utf-8")
        scenes_path.write_text(json.dumps(result["scenes"], ensure_ascii=False, indent=2), encoding="utf-8")
        (raw_dir / "arc_plan.json").write_text(json.dumps(result["arc_plan"], ensure_ascii=False, indent=2), encoding="utf-8")
        (raw_dir / "character_map.json").write_text(json.dumps(result["character_map"], ensure_ascii=False, indent=2), encoding="utf-8")

        console.print(f"  Script: {len(result['script'])} chars, {len(result['scenes'])} scenes")
        for c in result["arc_plan"].get("characters", []):
            console.print(f"    {c.get('original_name', '?')} -> {c.get('nickname', '?')}: {c.get('one_line', '')[:40]}")
    else:
        console.print("\n[dim][2/6] Script: cached[/]")

    script = script_path.read_text(encoding="utf-8")
    if scenes_path.exists():
        scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    else:
        # Re-parse scenes from cached script
        from src.segmenter.narrator import VideoNarrator
        scenes = VideoNarrator._parse_scenes(script)
        # Generate image prompts if we have the arc plan
        arc_plan_path = raw_dir / "arc_plan.json"
        if arc_plan_path.exists():
            llm = LLMClient(config.llm)
            narrator = VideoNarrator(llm)
            char_visuals = narrator._build_character_visuals(
                json.loads(arc_plan_path.read_text(encoding="utf-8"))
            )
            scenes = narrator._generate_image_prompts(scenes, char_visuals)
        scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"  Re-parsed: {len(scenes)} scenes")

    # ── Step 3: Generate character + scene images ─────────────────
    console.print(f"\n[bold cyan][3/6] Generating images ({len(scenes)} scenes)...[/]")
    img_dir = project_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Use shared character images if available
    shared_chars = Path(_project_root / "data" / "projects" / f"novel_{novel_id}" / "shared_characters")

    if placeholder:
        from src.image_gen.placeholder import PlaceholderGenerator
        gen = PlaceholderGenerator()
    else:
        gen = ReplicateGenerator()

    for i, scene in enumerate(scenes):
        img_path = str(img_dir / f"scene_{i:03d}.png")
        if Path(img_path).exists():
            console.print(f"  Scene {i + 1}/{len(scenes)}: cached")
            scene["image_path"] = img_path
            continue

        console.print(f"  Scene {i + 1}/{len(scenes)}: {scene.get('narration', '')[:40]}...")
        prompt = scene.get("image_prompt", "")
        if not prompt:
            prompt = f"anime illustration, {scene.get('visual_note', 'dramatic scene')}"

        # Find character reference
        ref_image = None
        for char_nick in scene.get("characters_in_scene", []):
            ref_path = shared_chars / f"{char_nick}.png" if shared_chars.exists() else None
            if ref_path and ref_path.exists():
                ref_image = str(ref_path)
                break

        if placeholder:
            gen.generate(prompt, img_path)
        else:
            gen.generate(prompt, img_path, reference_image=ref_image)
        scene["image_path"] = img_path

    # ── Step 4: TTS ──────────────────────────────────────────────
    console.print("\n[bold cyan][4/6] Generating narration audio...[/]")
    audio_dir = project_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    voice_assigner = VoiceAssigner()
    tts = MultiVoiceNarrator(rate="+25%", voice_assigner=voice_assigner)

    tts_results = []
    for i, scene in enumerate(scenes):
        audio_path = str(audio_dir / f"scene_{i:03d}.mp3")
        if Path(audio_path).exists():
            console.print(f"  Scene {i + 1}/{len(scenes)}: cached")
            from src.tts.audio_processing import get_duration_ms
            tts_results.append(TTSResult(audio_path=audio_path, duration_ms=get_duration_ms(audio_path)))
            scene["audio_path"] = audio_path
            continue

        console.print(f"  Scene {i + 1}/{len(scenes)}")
        narration = scene.get("narration", "")
        result = tts.synthesize(narration, audio_path, characters_present=scene.get("characters_in_scene", []))
        tts_results.append(result)
        scene["audio_path"] = audio_path

    # ── Step 5: Subtitles ────────────────────────────────────────
    console.print("\n[bold cyan][5/6] Generating subtitles...[/]")
    import pysubs2
    from src.subtitles.generator import _DEFAULT_STYLE

    sub_path = str(project_path / "subtitles.ass")
    subs = pysubs2.SSAFile()
    subs.styles["Default"] = _DEFAULT_STYLE

    cumulative_ms = 0
    for scene, tts_result in zip(scenes, tts_results):
        narration = scene.get("narration", "")
        duration_ms = tts_result.duration_ms
        if not narration.strip() or duration_ms <= 0:
            cumulative_ms += duration_ms
            continue

        # Clean for display
        clean = re.sub(r'\[.+?\][：:]\s*', '', narration)
        clean = re.sub(r'\[.+?\]', '', clean)
        for q in '"""\'\'「」':
            clean = clean.replace(q, '')
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Split into sentences
        parts = re.split(r'([。！？；…\n])', clean)
        sentences = []
        current = ""
        for part in parts:
            current += part
            if part in ("。", "！", "？", "；", "…", "\n"):
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""
        if current.strip():
            sentences.append(current.strip())

        # Break long sentences on commas
        final_sents = []
        for sent in sentences:
            if len(sent) <= 25:
                final_sents.append(sent)
            else:
                comma_parts = re.split(r'([，、：])', sent)
                chunk = ""
                for cp in comma_parts:
                    if len(chunk) + len(cp) > 25 and chunk:
                        final_sents.append(chunk.strip())
                        chunk = cp
                    else:
                        chunk += cp
                if chunk.strip():
                    final_sents.append(chunk.strip())

        total_chars = max(1, sum(len(s) for s in final_sents))
        current_ms = cumulative_ms
        for sent in final_sents:
            sent_dur = max(600, int(duration_ms * len(sent) / total_chars))
            end_ms = min(current_ms + sent_dur, cumulative_ms + duration_ms)
            subs.events.append(pysubs2.SSAEvent(start=current_ms, end=end_ms, text=sent, style="Default"))
            current_ms = end_ms
        cumulative_ms += duration_ms

    subs.save(sub_path, format_="ass")

    # ── Step 6: Assemble ─────────────────────────────────────────
    console.print("\n[bold cyan][6/6] Assembling video...[/]")
    from src.assembler.assembler import VideoAssembler

    assembler = VideoAssembler(
        target_resolution=config.video.target_resolution,
        fps=config.video.target_fps,
    )

    scene_data = []
    for scene, tts_result in zip(scenes, tts_results):
        if scene.get("image_path") and scene.get("audio_path"):
            scene_data.append({
                "image_path": scene["image_path"],
                "audio_path": scene["audio_path"],
                "effect": "random",
            })

    output_dir = project_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"narrate_ch{start_chapter}-{end_chapter}.mp4")

    final_path = assembler.assemble_scenes(scene_data, output_path, subtitle_path=sub_path)
    total_dur = sum(r.duration_ms for r in tts_results) / 1000.0

    console.print(f"\n[bold green]Done![/] {final_path}")
    console.print(f"  Duration: {total_dur:.1f}s ({total_dur / 60:.1f} min)")
    console.print(f"  Scenes: {len(scene_data)}")


@app.command()
def run(
    novel_id: str = typer.Option(..., "--novel-id", "-n", help="Fanqie novel ID"),
    chapter: int = typer.Option(1, "--chapter", "-c", help="Chapter number to process"),
    project_dir: str = typer.Option("", "--project-dir", "-d", help="Project directory (auto-generated if empty)"),
    style: str = typer.Option("anime", "--style", "-s", help="Art style for images"),
    placeholder: bool = typer.Option(False, "--placeholder", "-p", help="Use placeholder images (no API cost)"),
    voice: str = typer.Option("zh-CN-YunxiNeural", "--voice", "-v", help="TTS voice"),
    from_step: str = typer.Option("", "--from-step", "-f", help="Re-run from this step: scrape/rewrite/segment/char_images/scene_images/tts/subtitles/assemble"),
    verbose: bool = typer.Option(False, "--verbose", help="Show debug logs"),
):
    """Run the full pipeline: scrape → images → TTS → video. Caches each step."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    from src.core.config import PipelineConfig
    config = PipelineConfig.load()
    config.image.style = style
    config.tts.voice = voice

    if not project_dir:
        project_dir = str(_project_root / "data" / "projects" / f"novel_{novel_id}" / f"ch_{chapter:03d}")

    from src.core.pipeline import Pipeline
    pipeline = Pipeline(project_dir=project_dir, config=config)

    try:
        final_path = pipeline.run(
            novel_id=novel_id,
            chapter_num=chapter,
            use_placeholder_images=placeholder,
            from_step=from_step if from_step else None,
        )
        console.print(f"\n[bold green]Video saved to:[/] {final_path}")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info(
    novel_id: str = typer.Option(..., "--novel-id", "-n", help="Fanqie novel ID"),
):
    """Show novel information."""
    from src.scraper.fanqie import FanqieScraper

    scraper = FanqieScraper(rate_limit=1.5)
    novel_info = scraper.get_novel_info(novel_id)
    chapters = scraper.get_chapter_list(novel_id)
    scraper.close()

    console.print(f"\n[bold]{novel_info.title}[/]")
    console.print(f"Author: {novel_info.author}")
    console.print(f"Chapters: {len(chapters)}")
    if novel_info.cover_url:
        console.print(f"Cover: {novel_info.cover_url}")
    console.print(f"\n{novel_info.description}")


@app.command()
def chapters(
    novel_id: str = typer.Option(..., "--novel-id", "-n", help="Fanqie novel ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max chapters to show"),
):
    """List chapters of a novel."""
    from src.scraper.fanqie import FanqieScraper

    scraper = FanqieScraper(rate_limit=1.5)
    chapter_list = scraper.get_chapter_list(novel_id)
    scraper.close()

    table = Table(title=f"Chapters (showing {min(limit, len(chapter_list))}/{len(chapter_list)})")
    table.add_column("#", style="dim")
    table.add_column("Title")
    table.add_column("ID", style="dim")

    for ch in chapter_list[:limit]:
        table.add_row(str(ch.sequence), ch.title, ch.chapter_id)

    console.print(table)


@app.command()
def plan_characters(
    novel_id: str = typer.Option(..., "--novel-id", "-n", help="Fanqie novel ID"),
    scan_chapters: int = typer.Option(5, "--scan", "-s", help="How many chapters to scan for characters"),
    generate: bool = typer.Option(True, "--generate/--no-generate", help="Generate character images"),
    placeholder: bool = typer.Option(False, "--placeholder", "-p", help="Use placeholder images"),
):
    """Scan first N chapters and create character sheets for the entire novel."""
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")

    from src.core.config import PipelineConfig
    from src.core.llm_client import LLMClient
    from src.image_gen.replicate_backend import ReplicateGenerator
    from src.scraper.fanqie import FanqieScraper
    from src.segmenter.rewriter import ChapterRewriter

    config = PipelineConfig.load()
    scraper = FanqieScraper(rate_limit=2.0)
    llm = LLMClient(config.llm)
    rewriter = ChapterRewriter(llm)

    novel_info = scraper.get_novel_info(novel_id)
    chapter_list = scraper.get_chapter_list(novel_id)
    console.print(f"\nScanning [bold]{novel_info.title}[/] — first {scan_chapters} chapters\n")

    # Collect characters from all scanned chapters
    all_characters: dict[str, dict] = {}
    for i in range(min(scan_chapters, len(chapter_list))):
        ch = chapter_list[i]
        console.print(f"  Scanning chapter {i + 1}: {ch.title}...")
        content = scraper.download_chapter(ch.chapter_id)
        chars = rewriter.extract_characters_visual(content.text)
        for c in chars:
            name = c["name"]
            if name not in all_characters:
                all_characters[name] = c
            else:
                # Update if new description is more detailed
                if len(c.get("appearance", "")) > len(all_characters[name].get("appearance", "")):
                    all_characters[name]["appearance"] = c["appearance"]
                    all_characters[name]["appearance_en"] = c.get("appearance_en", "")

    scraper.close()

    console.print(f"\n[bold]Found {len(all_characters)} characters:[/]")
    for name, c in all_characters.items():
        console.print(f"  {name} ({c.get('role', '?')}): {c.get('appearance', '')[:60]}...")

    # Save character plan
    import json
    plan_dir = _project_root / "data" / "projects" / f"novel_{novel_id}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "character_plan.json"
    plan_path.write_text(json.dumps(list(all_characters.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\nSaved character plan: {plan_path}")

    # Generate character images
    if generate:
        char_dir = plan_dir / "shared_characters"
        char_dir.mkdir(parents=True, exist_ok=True)

        if placeholder:
            from src.image_gen.placeholder import PlaceholderGenerator
            gen = PlaceholderGenerator()
        else:
            gen = ReplicateGenerator()

        console.print(f"\n[bold]Generating character images...[/]")
        for name, c in all_characters.items():
            img_path = str(char_dir / f"{name}.png")
            if Path(img_path).exists():
                console.print(f"  {name}: already exists")
                continue
            appearance = c.get("appearance_en", c.get("appearance", name))
            console.print(f"  Generating {name}...")
            gen.generate_character_sheet(appearance, img_path, style="anime")

        console.print(f"\n[bold green]Done![/] Character images in: {char_dir}")


@app.command()
def test_image(
    prompt: str = typer.Option("anime girl standing in a garden, detailed background", "--prompt", "-p"),
    output: str = typer.Option("data/test/test_output.png", "--output", "-o"),
    reference: str = typer.Option("", "--reference", "-r", help="Reference image for character consistency"),
):
    """Test image generation with Replicate."""
    from src.image_gen.replicate_backend import ReplicateGenerator

    gen = ReplicateGenerator()
    ref = reference if reference else None
    result = gen.generate(prompt, output, reference_image=ref)
    console.print(f"[green]Generated:[/] {result.image_path} ({result.width}x{result.height})")


if __name__ == "__main__":
    app()
