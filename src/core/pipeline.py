"""
Pipeline orchestrator: runs the full novel-to-video pipeline.

Each step checks for existing work and skips if already done.
Individual steps can be re-run with --from-step flag.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.core.config import PipelineConfig
from src.core.llm_client import LLMClient
from src.core.manifest_manager import ManifestManager
from src.image_gen.replicate_backend import ReplicateGenerator
from src.models import AssetStatus, Character, Manifest, Scene, Source
from src.scraper.fanqie import FanqieScraper
from src.segmenter.rewriter import ChapterRewriter
from src.segmenter.segmenter import SceneSegmenter
from src.subtitles.generator import SubtitleGenerator
from src.tts.base import TTSResult
from src.tts.edge_tts_backend import EdgeTTSNarrator
from src.tts.multi_voice import MultiVoiceNarrator, VoiceAssigner

logger = logging.getLogger(__name__)

STEPS = ["scrape", "rewrite", "segment", "char_images", "scene_images", "tts", "subtitles", "assemble"]


class Pipeline:
    """Full novel-to-video pipeline orchestrator."""

    def __init__(self, project_dir: str, config: PipelineConfig | None = None):
        self.project_dir = Path(project_dir)
        self.config = config or PipelineConfig.load()
        self.manifest_mgr = ManifestManager(self.project_dir)

        # Initialize modules
        self.scraper = FanqieScraper(rate_limit=1.5)
        self.llm = LLMClient(self.config.llm)
        self.segmenter = SceneSegmenter(self.llm, art_style=self.config.image.style)
        self.rewriter = ChapterRewriter(self.llm)
        self.image_gen = ReplicateGenerator()
        self.voice_assigner = VoiceAssigner()
        self.tts = MultiVoiceNarrator(
            rate="+30%",
            voice_assigner=self.voice_assigner,
        )
        self.subtitle_gen = SubtitleGenerator()

    def run(
        self,
        novel_url: str | None = None,
        novel_id: str | None = None,
        chapter_num: int = 1,
        use_placeholder_images: bool = False,
        from_step: str | None = None,
    ) -> str:
        """
        Run the pipeline. Skips already-completed steps unless from_step is set.
        from_step: start from this step (e.g., "subtitles" to only redo subs + assembly)
        """
        from rich.console import Console
        console = Console(highlight=False)

        # Determine which steps to run
        if from_step:
            start_idx = STEPS.index(from_step) if from_step in STEPS else 0
        else:
            start_idx = 0

        def should_run(step: str) -> bool:
            return STEPS.index(step) >= start_idx

        # ── Step 1: Scrape ────────────────────────────────────────────
        raw_dir = self.project_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"chapter_{chapter_num:03d}.txt"
        rewrite_path = raw_dir / f"chapter_{chapter_num:03d}_rewritten.txt"

        if should_run("scrape") and not raw_path.exists():
            console.print("\n[bold cyan][1/8] Scraping chapter...[/]")
            if novel_url:
                novel_id = FanqieScraper.extract_novel_id(novel_url)
            if not novel_id:
                raise ValueError("Provide novel_url or novel_id")

            info = self.scraper.get_novel_info(novel_id)
            chapters = self.scraper.get_chapter_list(novel_id)
            console.print(f"  Novel: {info.title} by {info.author} ({len(chapters)} chapters)")

            if chapter_num < 1 or chapter_num > len(chapters):
                raise ValueError(f"Chapter {chapter_num} out of range (1-{len(chapters)})")

            ch = chapters[chapter_num - 1]
            content = self.scraper.download_chapter(ch.chapter_id)
            raw_path.write_text(content.text, encoding="utf-8")
            console.print(f"  Chapter {chapter_num}: {ch.title} ({len(content.text)} chars)")

            if not self.manifest_mgr.exists():
                manifest = self.manifest_mgr.create(project_name=info.title)
                manifest.source = Source(
                    platform="fanqie", novel_id=novel_id or "",
                    novel_title=info.title, chapter_id=ch.chapter_id,
                    chapter_title=ch.title, chapter_number=chapter_num,
                    raw_text_path=str(raw_path),
                )
                self.manifest_mgr.save(manifest)
        else:
            console.print("\n[dim][1/8] Scrape: cached[/]")

        chapter_text = raw_path.read_text(encoding="utf-8")
        manifest = self.manifest_mgr.load() if self.manifest_mgr.exists() else self.manifest_mgr.create()

        # ── Step 2: Rewrite ───────────────────────────────────────────
        char_dicts_path = raw_dir / f"chapter_{chapter_num:03d}_characters.json"

        if should_run("rewrite") and not rewrite_path.exists():
            console.print("\n[bold cyan][2/8] Extracting characters & rewriting...[/]")
            char_dicts, rewritten = self.rewriter.process_chapter(chapter_text)

            console.print(f"  Found {len(char_dicts)} characters:")
            for cd in char_dicts:
                console.print(f"    - {cd['name']} ({cd['role']}): {cd['appearance'][:50]}...")

            rewrite_path.write_text(rewritten, encoding="utf-8")

            import json
            char_dicts_path.write_text(json.dumps(char_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            console.print("\n[dim][2/8] Rewrite: cached[/]")

        rewritten = rewrite_path.read_text(encoding="utf-8")
        import json
        char_dicts = json.loads(char_dicts_path.read_text(encoding="utf-8")) if char_dicts_path.exists() else []

        characters = []
        for cd in char_dicts:
            characters.append(Character(
                name=cd["name"],
                role=cd.get("role", "minor"),
                description=cd.get("appearance_en", cd.get("appearance", "")),
            ))

        # ── Step 3: Segment ───────────────────────────────────────────
        scenes_path = raw_dir / f"chapter_{chapter_num:03d}_scenes.json"

        if should_run("segment") and not scenes_path.exists():
            console.print("\n[bold cyan][3/8] Segmenting into scenes...[/]")
            _, scenes = self.segmenter.process_chapter(rewritten, existing_characters=characters)
            console.print(f"  {len(scenes)} scenes created")

            scenes_data = [s.model_dump(mode="json") for s in scenes]
            scenes_path.write_text(json.dumps(scenes_data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            console.print("\n[dim][3/8] Segment: cached[/]")

        scenes_data = json.loads(scenes_path.read_text(encoding="utf-8"))
        scenes = [Scene.model_validate(s) for s in scenes_data]

        # ── Step 4: Character Images (shared across all chapters) ─────
        # Store character images at novel level, not chapter level
        char_dir = self.project_dir.parent / "shared_characters"
        char_dir.mkdir(parents=True, exist_ok=True)

        if should_run("char_images"):
            console.print("\n[bold cyan][4/8] Generating character images...[/]")
            for i, char in enumerate(characters):
                ref_path = str(char_dir / f"{char.name}.png")
                if Path(ref_path).exists():
                    console.print(f"  {char.name}: cached")
                    char.reference_image_path = ref_path
                    continue

                appearance_en = char_dicts[i].get("appearance_en", char.description) if i < len(char_dicts) else char.description
                console.print(f"  Generating {char.name}...")

                if use_placeholder_images:
                    from src.image_gen.placeholder import PlaceholderGenerator
                    PlaceholderGenerator().generate_character_sheet(appearance_en, ref_path, style=self.config.image.style)
                else:
                    self.image_gen.generate_character_sheet(appearance_en, ref_path, style=self.config.image.style)

                char.reference_image_path = ref_path
        else:
            console.print("\n[dim][4/8] Character images: skipped[/]")
            for char in characters:
                ref_path = str(char_dir / f"{char.name}.png")
                if Path(ref_path).exists():
                    char.reference_image_path = ref_path

        # ── Step 5: Scene Images ──────────────────────────────────────
        scene_dir = self.project_dir / "images" / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        char_refs = {c.name: c.reference_image_path for c in characters if c.reference_image_path}

        if should_run("scene_images"):
            console.print("\n[bold cyan][5/8] Generating scene images...[/]")
            for i, scene in enumerate(scenes):
                img_path = str(scene_dir / f"scene_{i:03d}.png")
                if Path(img_path).exists():
                    console.print(f"  Scene {i + 1}/{len(scenes)}: cached")
                    scene.assets.image_path = img_path
                    scene.assets.image_status = AssetStatus.COMPLETE
                    continue

                console.print(f"  Scene {i + 1}/{len(scenes)}: {scene.narration_text[:40]}...")
                ref_image = None
                for char_name in scene.characters_present:
                    if char_name in char_refs:
                        ref_image = char_refs[char_name]
                        break

                if use_placeholder_images:
                    from src.image_gen.placeholder import PlaceholderGenerator
                    PlaceholderGenerator().generate(scene.image_prompt, img_path)
                else:
                    self.image_gen.generate(scene.image_prompt, img_path, reference_image=ref_image)

                scene.assets.image_path = img_path
                scene.assets.image_status = AssetStatus.COMPLETE
        else:
            console.print("\n[dim][5/8] Scene images: skipped[/]")
            for i, scene in enumerate(scenes):
                img_path = str(scene_dir / f"scene_{i:03d}.png")
                if Path(img_path).exists():
                    scene.assets.image_path = img_path
                    scene.assets.image_status = AssetStatus.COMPLETE

        # ── Step 6: TTS ───────────────────────────────────────────────
        audio_dir = self.project_dir / "audio" / "narration"
        audio_dir.mkdir(parents=True, exist_ok=True)

        if should_run("tts"):
            console.print("\n[bold cyan][6/8] Generating narration (multi-voice)...[/]")
            voice_char_dicts = []
            for i, c in enumerate(characters):
                gender = char_dicts[i].get("gender", "unknown") if i < len(char_dicts) else "unknown"
                voice_char_dicts.append({"name": c.name, "role": c.role, "gender": gender})
            voice_map = self.tts.assign_character_voices(voice_char_dicts)
            for name, voice in voice_map.items():
                console.print(f"  Voice: {name} -> {voice}")

            tts_results: list[TTSResult] = []
            for i, scene in enumerate(scenes):
                audio_path = str(audio_dir / f"scene_{i:03d}.mp3")
                if Path(audio_path).exists() and from_step not in ("tts",):
                    console.print(f"  Scene {i + 1}/{len(scenes)}: cached")
                    from src.tts.audio_processing import get_duration_ms
                    tts_results.append(TTSResult(audio_path=audio_path, duration_ms=get_duration_ms(audio_path)))
                    scene.assets.audio_path = audio_path
                    scene.assets.audio_status = AssetStatus.COMPLETE
                    continue

                console.print(f"  Scene {i + 1}/{len(scenes)}")
                result = self.tts.synthesize(scene.narration_text, audio_path, characters_present=scene.characters_present)
                tts_results.append(result)
                scene.assets.audio_path = audio_path
                scene.assets.audio_status = AssetStatus.COMPLETE
                scene.duration_estimate_seconds = result.duration_ms / 1000.0
        else:
            console.print("\n[dim][6/8] TTS: loading cached[/]")
            tts_results = []
            for i, scene in enumerate(scenes):
                audio_path = str(audio_dir / f"scene_{i:03d}.mp3")
                if Path(audio_path).exists():
                    from src.tts.audio_processing import get_duration_ms
                    tts_results.append(TTSResult(audio_path=audio_path, duration_ms=get_duration_ms(audio_path)))
                    scene.assets.audio_path = audio_path

        # ── Step 7: Subtitles (always regenerate — instant + settings may change) ──
        console.print("\n[bold cyan][7/8] Generating subtitles...[/]")
        sub_dir = self.project_dir / "subtitles"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_path = str(sub_dir / f"chapter_{chapter_num:03d}.ass")

        # Generate subtitles: 1 sentence at a time, strip speaker tags
        import re
        import pysubs2
        from src.subtitles.generator import _DEFAULT_STYLE

        def _clean_for_display(text: str) -> str:
            """Strip [Speaker]: tags and visual notes for subtitle display."""
            text = re.sub(r'\[.+?\][：:]\s*', '', text)
            text = re.sub(r'\[(?:画面|外貌)[：:].+?\]', '', text)
            text = text.replace('---SCENE---', '')
            # Remove quotes that wrap dialogue
            for q in '"""\'\'「」':
                text = text.replace(q, '')
            return re.sub(r'\s+', ' ', text).strip()

        def _split_sentences(text: str, max_len: int = 25) -> list[str]:
            """Split into short subtitle lines. Break on sentence-end punctuation first, then commas."""
            # First split on sentence endings
            parts = re.split(r'([。！？；…\n])', text)
            raw_sentences = []
            current = ""
            for part in parts:
                current += part
                if part in ("。", "！", "？", "；", "…", "\n"):
                    s = current.strip()
                    if s:
                        raw_sentences.append(s)
                    current = ""
            if current.strip():
                raw_sentences.append(current.strip())

            # Then break any sentence longer than max_len on commas
            final = []
            for sent in raw_sentences:
                if len(sent) <= max_len:
                    final.append(sent)
                else:
                    # Split on commas
                    comma_parts = re.split(r'([，、：])', sent)
                    chunk = ""
                    for cp in comma_parts:
                        if len(chunk) + len(cp) > max_len and chunk:
                            final.append(chunk.strip())
                            chunk = cp
                        else:
                            chunk += cp
                    if chunk.strip():
                        final.append(chunk.strip())
            return final

        subs = pysubs2.SSAFile()
        subs.styles["Default"] = _DEFAULT_STYLE

        cumulative_ms = 0
        for scene, tts_result in zip(scenes, tts_results):
            raw_text = scene.narration_text
            duration_ms = tts_result.duration_ms
            if not raw_text.strip() or duration_ms <= 0:
                cumulative_ms += duration_ms
                continue

            # Clean for display, then split into sentences
            clean = _clean_for_display(raw_text)
            sentences = _split_sentences(clean)
            if not sentences:
                cumulative_ms += duration_ms
                continue

            total_chars = max(1, sum(len(s) for s in sentences))
            current_ms = cumulative_ms

            for sentence in sentences:
                char_count = len(sentence)
                sent_dur = max(800, int(duration_ms * char_count / total_chars))
                end_ms = min(current_ms + sent_dur, cumulative_ms + duration_ms)
                subs.events.append(pysubs2.SSAEvent(
                    start=current_ms, end=end_ms,
                    text=sentence, style="Default",
                ))
                current_ms = end_ms

            cumulative_ms += duration_ms

        subs.save(sub_path, format_="ass")
        console.print(f"  {sub_path}")

        # ── Step 8: Assemble (always regenerate — uses latest subs/settings) ──
        console.print("\n[bold cyan][8/8] Assembling video...[/]")
        from src.assembler.assembler import VideoAssembler

        assembler = VideoAssembler(
            target_resolution=self.config.video.target_resolution,
            fps=self.config.video.target_fps,
        )

        scene_data = []
        for i, (scene, tts_result) in enumerate(zip(scenes, tts_results)):
            if scene.assets.image_path and scene.assets.audio_path:
                scene_data.append({
                    "image_path": scene.assets.image_path,
                    "audio_path": scene.assets.audio_path,
                    "effect": "random",
                })

        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"chapter_{chapter_num:03d}.mp4")

        final_path = assembler.assemble_scenes(scene_data, output_path, subtitle_path=sub_path)

        # Update manifest
        manifest.characters = characters
        manifest.scenes = scenes
        manifest.output.final_video_path = final_path
        total_dur = sum(r.duration_ms for r in tts_results) / 1000.0
        manifest.output.total_duration_seconds = total_dur
        from src.models.enums import ProjectStatus
        manifest.output.status = ProjectStatus.COMPLETE
        self.manifest_mgr.save(manifest)

        console.print(f"\n[bold green]Done![/] Video: {final_path}")
        console.print(f"  Duration: {total_dur:.1f}s ({total_dur / 60:.1f} min)")
        console.print(f"  Scenes: {len(scene_data)}")

        return final_path
