"""
Narrator v2: 视频解说 script generator with audience state tracking.

Uses Story Bible + archetype map + narration manifest to generate
consistent, contextualized narration scripts across multiple videos.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.llm_client import LLMClient
from src.narration.bible import StoryBible
from src.narration.prompts import (
    AUDIENCE_EXTRACTION_PROMPT,
    BRIDGE_CONTINUATION,
    BRIDGE_FIRST_VIDEO,
    NARRATOR_V2_PROMPT,
    NARRATOR_V2_SYSTEM,
)

logger = logging.getLogger(__name__)


# ── Narration Manifest ────────────────────────────────────────────────────────


class VideoNarrationRecord(BaseModel):
    """What was narrated in a single video."""

    video_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    chapters_covered: list[int] = Field(default_factory=list)
    archetypes_used: dict[str, str] = Field(default_factory=dict)  # name → archetype
    audience_knows: list[str] = Field(default_factory=list)
    audience_does_not_know: list[str] = Field(default_factory=list)
    cliffhanger: str = ""
    last_scene_summary: str = ""


class NarrationManifest(BaseModel):
    """Tracks narration state across all videos for a novel."""

    novel_id: str
    videos: list[VideoNarrationRecord] = Field(default_factory=list)
    locked_archetypes: dict[str, str] = Field(default_factory=dict)  # name → archetype

    def get_audience_state(self) -> dict:
        """Aggregate what the audience knows across all videos."""
        all_knows: list[str] = []
        all_unknowns: list[str] = []
        for video in self.videos:
            all_knows.extend(video.audience_knows)
            all_unknowns.extend(video.audience_does_not_know)

        # Things revealed in later videos are no longer unknown
        knows_set = set(all_knows)
        still_unknown = [u for u in all_unknowns if u not in knows_set]

        return {
            "audience_knows": list(knows_set),
            "audience_does_not_know": still_unknown,
        }

    def get_previous_video(self) -> VideoNarrationRecord | None:
        """Get the most recent video record."""
        return self.videos[-1] if self.videos else None

    def lock_archetype(self, name: str, archetype: str) -> None:
        """Lock an archetype assignment so it persists across all future videos."""
        self.locked_archetypes[name] = archetype

    def add_video(self, record: VideoNarrationRecord) -> None:
        """Add a video record and lock its archetypes."""
        self.videos.append(record)
        for name, archetype in record.archetypes_used.items():
            self.lock_archetype(name, archetype)

    def save(self, path: Path) -> None:
        """Save manifest to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> NarrationManifest:
        """Load manifest from JSON."""
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


# ── Narrator v2 ──────────────────────────────────────────────────────────────


class NarratorV2:
    """Generates 视频解说 scripts using Bible context and audience state tracking."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_script(
        self,
        chapters_text: list[str],
        chapter_numbers: list[int],
        bible: StoryBible,
        archetype_map: dict[str, str],
        manifest: NarrationManifest,
        target_minutes: float = 3.0,
        target_scenes: int = 15,
    ) -> dict:
        """
        Full narration pipeline using pre-built context.

        Args:
            chapters_text: Raw chapter texts.
            chapter_numbers: Corresponding chapter numbers.
            bible: Story Bible with character/world context.
            archetype_map: {original_name: archetype_name}.
            manifest: Narration Manifest tracking audience state.
            target_minutes: Target video duration.
            target_scenes: Target number of scenes.

        Returns:
            Dict with keys: script, scenes, video_record
        """
        combined_text = "\n\n---章节分割---\n\n".join(chapters_text)
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "\n\n[后续章节省略...]"

        # Build context
        character_sheet = self._build_character_sheet(archetype_map, bible)
        story_context = bible.get_context_for_chapter()
        bridge = self._build_bridge_instructions(manifest)

        target_length = f"{int(target_minutes)}分钟左右（约{int(target_minutes * 250)}字）"

        # Generate narration
        prompt = NARRATOR_V2_PROMPT.format(
            character_sheet=character_sheet,
            story_context=story_context,
            bridge_instructions=bridge,
            chapters_text=combined_text,
            target_length=target_length,
            target_scenes=target_scenes,
        )

        logger.info("Generating narration script...")
        script = self.llm.chat(
            prompt=prompt,
            system=NARRATOR_V2_SYSTEM,
            temperature=0.6,
            max_tokens=8192,
        )

        # Validate: replace any leaked original names with archetype names
        script = self._validate_script(script, archetype_map)

        # Clean: remove 咱们, cliffhangers, subscription hooks
        script = self._clean_script(script)

        # Parse scenes
        scenes = self._parse_scenes(script)

        # Dedup: remove repeated scenes (LLM looping issue)
        scenes = self._dedup_scenes(scenes)

        # Rebuild clean script from deduped scenes
        script = "\n\n---SCENE---\n".join(
            s["narration"] + (f"\n[画面：{s['visual_note']}]" if s.get("visual_note") else "")
            for s in scenes
        )

        # Extract audience state updates
        logger.info("Extracting audience state updates...")
        audience_updates = self._extract_audience_updates(script, bible)

        # Build video record
        video_record = VideoNarrationRecord(
            chapters_covered=chapter_numbers,
            archetypes_used=archetype_map,
            audience_knows=audience_updates.get("audience_knows", []),
            audience_does_not_know=audience_updates.get("audience_does_not_know", []),
            cliffhanger=audience_updates.get("cliffhanger", ""),
            last_scene_summary=audience_updates.get("last_scene_summary", ""),
        )

        return {
            "script": script,
            "scenes": scenes,
            "video_record": video_record,
            "archetype_map": archetype_map,
        }

    @staticmethod
    def _build_character_sheet(archetype_map: dict[str, str], bible: StoryBible) -> str:
        """Build the character sheet for the narration prompt."""
        lines = []
        for original_name, archetype in archetype_map.items():
            char = bible.characters.get(original_name)
            if char:
                lines.append(f"- {archetype} (原名{original_name}): {char.description}")
                if char.arc_status:
                    lines.append(f"  当前状态: {char.arc_status}")
            else:
                lines.append(f"- {archetype} (原名{original_name})")
        return "\n".join(lines) if lines else "No characters defined."

    @staticmethod
    def _build_bridge_instructions(manifest: NarrationManifest) -> str:
        """Build bridge instructions based on previous video state."""
        prev = manifest.get_previous_video()
        if prev is None:
            return BRIDGE_FIRST_VIDEO

        state = manifest.get_audience_state()
        knows = "\n".join(f"  - {k}" for k in state["audience_knows"][-10:]) or "  (nothing yet)"
        unknowns = "\n".join(f"  - {u}" for u in state["audience_does_not_know"][-5:]) or "  (none)"

        # Shorten cliffhanger for the bridge opening example
        cliffhanger_short = prev.cliffhanger[:30] + "..." if len(prev.cliffhanger) > 30 else prev.cliffhanger

        return BRIDGE_CONTINUATION.format(
            audience_knows=knows,
            audience_does_not_know=unknowns,
            cliffhanger=prev.cliffhanger,
            cliffhanger_short=cliffhanger_short,
        )

    def _extract_audience_updates(self, script: str, bible: StoryBible) -> dict:
        """Use LLM to extract what the audience learned from this narration."""
        prompt = AUDIENCE_EXTRACTION_PROMPT.format(
            script=script[:6000],
            bible_context=bible.get_context_for_chapter()[:3000],
        )

        try:
            result = self.llm.chat_json(
                prompt=prompt,
                system=NARRATOR_V2_SYSTEM,
                temperature=0.2,
            )
            if isinstance(result, dict):
                return result
        except (ValueError, Exception) as e:
            logger.warning("Failed to extract audience updates: %s", e)

        return {
            "audience_knows": [],
            "audience_does_not_know": [],
            "cliffhanger": "",
            "last_scene_summary": "",
        }

    @staticmethod
    def _dedup_scenes(scenes: list[dict]) -> list[dict]:
        """Remove repeated/looping scenes from parsed scene list.

        DeepSeek sometimes gets stuck repeating the same scenes. This detects
        duplicates by comparing narration text similarity and removes repeats.
        """
        if len(scenes) < 3:
            return scenes

        kept: list[dict] = []
        seen_narrations: list[str] = []
        duplicates_removed = 0

        for scene in scenes:
            narration = scene.get("narration", "").strip()
            if not narration:
                continue

            is_duplicate = False
            for seen in seen_narrations:
                if _text_similarity(narration, seen) > 0.7:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicates_removed += 1
            else:
                seen_narrations.append(narration)
                kept.append(scene)

        if duplicates_removed > 0:
            logger.info("Removed %d duplicate scenes", duplicates_removed)
            # Re-index
            for i, scene in enumerate(kept):
                scene["index"] = i

        return kept

    @staticmethod
    def _clean_script(script: str) -> str:
        """Post-process script to remove unwanted patterns."""
        # Remove 咱们/咱们的
        script = re.sub(r"咱们的(\w)", r"这位\1", script)
        script = re.sub(r"咱们", "故事", script)

        # Remove cliffhanger/hook lines
        hook_patterns = [
            r"想知道.*？.*关注.*",
            r"想知道.*？.*下[期集回].*",
            r"点赞.*关注.*",
            r"赶紧.*关注.*",
            r"下[期集回].*更[精刺].*",
            r"点赞过.*万.*",
            r"别忘了关注.*",
            r"速看下[期集回].*",
            r"咱们下[期集回].*",
        ]
        for pattern in hook_patterns:
            script = re.sub(pattern, "", script)

        # Remove leaked structure labels (LLM outputting its template)
        # These are bold/header markers from the prompt structure
        structure_patterns = [
            r"\*\*设定\*\*\n?",
            r"\*\*主角出场\*\*\n?",
            r"\*\*矛盾铺垫\*\*\n?",
            r"\*\*冲突爆发\*\*\n?",
            r"\*\*爽点\*\*\n?",
            r"\*\*后续\*\*\n?",
            r"\*\*结局\*\*\n?",
            r"\*\*总结\*\*\n?",
            r"\*\*结束\*\*\n?",
            r"\*\*画面\*\*\n?",
            r"\*\*[^*]{1,10}\*\*\n?",  # any short bold header
            r"###\s+.+\n",  # markdown h3 headers
            r"##\s+.+\n",   # markdown h2 headers
            r"- \*\*.+\*\*：.+\n",  # bullet point definitions
        ]
        for pattern in structure_patterns:
            script = re.sub(pattern, "", script)

        # Remove LLM meta-commentary / planning text
        meta_patterns = [
            r"好的，我将按照.*\n",
            r"请您看看.*\n",
            r"如果您有其他.*\n",
            r"以下是.*脚本.*\n",
            r"让我们拭目以待。?\n?",
        ]
        for pattern in meta_patterns:
            script = re.sub(pattern, "", script)

        # Remove empty lines left behind
        script = re.sub(r"\n{3,}", "\n\n", script)
        return script.strip()

    @staticmethod
    def _validate_script(script: str, archetype_map: dict[str, str]) -> str:
        """Replace any leaked original character names with their archetype names."""
        # Sort by name length descending to replace longest matches first
        replacements = sorted(archetype_map.items(), key=lambda x: len(x[0]), reverse=True)
        for original_name, archetype in replacements:
            if original_name in script:
                logger.info("Replacing leaked name '%s' → '%s' in script", original_name, archetype)
                script = script.replace(original_name, archetype)
        # Clean up double archetypes (e.g., "大佬大佬" from "大佬苏明硕" → "大佬大佬")
        for _, archetype in replacements:
            double = archetype + archetype
            if double in script:
                script = script.replace(double, archetype)
        return script

    @staticmethod
    def _parse_scenes(script: str) -> list[dict]:
        """Parse ---SCENE--- markers and [画面：] notes from the narration script."""
        scenes = []
        raw_scenes = re.split(r"\n-{3,}(?:SCENE)?-{0,}\n", script)

        for i, raw in enumerate(raw_scenes):
            raw = raw.strip()
            if not raw:
                continue

            # Extract visual note
            visual = ""
            visual_match = re.search(r"\[画面[：:](.+?)\]", raw)
            if visual_match:
                visual = visual_match.group(1).strip()

            # Get narration text (remove visual notes)
            narration = re.sub(r"\[画面[：:].+?\]", "", raw).strip()
            narration = re.sub(r"\[.+?\]", "", narration).strip()

            if narration:
                scenes.append({
                    "index": i,
                    "narration": narration,
                    "visual_note": visual,
                    "image_prompt": "",
                    "characters_in_scene": [],
                    "mood": "dramatic",
                })

        return scenes


def _text_similarity(a: str, b: str) -> float:
    """Fast character-level similarity between two Chinese text strings.

    Returns 0.0 (completely different) to 1.0 (identical).
    Uses character n-gram overlap (bigrams) for speed.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # Use character bigrams
    def bigrams(text: str) -> set[str]:
        return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}

    a_grams = bigrams(a)
    b_grams = bigrams(b)

    if not a_grams or not b_grams:
        return 0.0

    overlap = len(a_grams & b_grams)
    total = max(len(a_grams), len(b_grams))
    return overlap / total if total > 0 else 0.0
