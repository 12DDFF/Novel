"""
Multi-voice TTS: assigns distinct voices to narrator and each character.

Splits narration text into dialogue segments and narration segments,
synthesizes each with the appropriate voice, then concatenates.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydub import AudioSegment

from .base import SubtitleCue, TTSResult
from .edge_tts_backend import EdgeTTSNarrator

logger = logging.getLogger(__name__)

# Default voice assignments
DEFAULT_NARRATOR_VOICE = "zh-CN-YunxiNeural"  # narrator voice (young, neutral)
DEFAULT_RATE = "+30%"

# Voice pool for character assignment
VOICE_POOL = {
    "female": [
        "zh-CN-XiaoxiaoNeural",   # warm, expressive
        "zh-CN-XiaoyiNeural",     # gentle
        "zh-CN-liaoning-XiaobeiNeural",  # fun, northeast accent
    ],
    "male": [
        "zh-CN-YunyangNeural",    # warm, mature, charismatic (protagonist)
        "zh-CN-YunxiNeural",      # younger male (side characters)
        "zh-CN-YunxiaNeural",     # youngest male
    ],
}

def _strip_speaker_tags(text: str) -> str:
    """Remove [Speaker]: prefix before quotes, and visual/appearance brackets, for TTS/display."""
    # Remove [Name]： only when followed by a quote (speaker tag pattern)
    text = re.sub(rf'\[[^\]]+\][：:]\s*(?=[{_OPEN_QUOTES}])', '', text)
    # Remove [画面：...] and [外貌：...] visual notes
    text = re.sub(r'\[(?:画面|外貌)[：:][^\]]+\]', '', text)
    # Remove ---SCENE--- markers
    text = text.replace('---SCENE---', '')
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Chinese quotes: \u201c = " \u201d = " \u300c = 「 \u300d = 」
_OPEN_QUOTES = '"\u201c\u300c'
_CLOSE_QUOTES = '"\u201d\u300d'

# Pattern to detect dialogue: "text" or 「text」
_DIALOGUE_PATTERN = re.compile(
    rf'[{_OPEN_QUOTES}]([^{_CLOSE_QUOTES}]+)[{_CLOSE_QUOTES}]'
)

# Pattern to detect speaker tag from rewriter: [角色名]："对话"
_REWRITER_TAG_PATTERN = re.compile(
    rf'\[([^\]]+)\][：:]\s*[{_OPEN_QUOTES}]([^{_CLOSE_QUOTES}]+)[{_CLOSE_QUOTES}]'
)

# Pattern to detect speaker before dialogue: 某某说/道/问/笑道 etc
_SPEAKER_PATTERN = re.compile(
    r'([\u4e00-\u9fff]{1,4})(?:说|道|问|笑道|冷笑|怒道|叹道|喊道|叫道|回答|开口|低声|沉声|淡淡地|微微一笑)(?:：|:)?'
)


class VoiceAssigner:
    """Assigns voices to characters based on their role and gender."""

    def __init__(self):
        self._assignments: dict[str, str] = {}
        self._female_idx = 0
        self._male_idx = 0

    def assign(self, character_name: str, gender: str = "unknown", role: str = "minor") -> str:
        """Get or assign a voice for a character."""
        if character_name in self._assignments:
            return self._assignments[character_name]

        # Elder/father/authority characters get the deepest voice
        if self._guess_elder(character_name, role):
            voice = "zh-CN-YunjianNeural"
        elif gender == "female" or self._guess_female(character_name):
            pool = VOICE_POOL["female"]
            voice = pool[self._female_idx % len(pool)]
            self._female_idx += 1
        else:
            pool = VOICE_POOL["male"]
            voice = pool[self._male_idx % len(pool)]
            self._male_idx += 1

        self._assignments[character_name] = voice
        logger.info(f"Assigned voice {voice} to {character_name}")
        return voice

    def get_assignments(self) -> dict[str, str]:
        return dict(self._assignments)

    @staticmethod
    def _guess_elder(name: str, role: str) -> bool:
        """Detect elder/father/authority characters who need a deep voice."""
        elder_keywords = {"父", "爷", "爸", "叔", "伯", "公", "老", "长老", "将军", "大人", "师"}
        role_keywords = {"father", "elder", "general", "commander", "leader"}
        return (
            any(k in name for k in elder_keywords)
            or any(k in role.lower() for k in role_keywords)
        )

    @staticmethod
    def _guess_female(name: str) -> bool:
        """Simple heuristic: common female name characters."""
        female_chars = set("香婉娘姐妹嫂娇莲花玉凤燕")
        return any(c in female_chars for c in name)


class MultiVoiceNarrator:
    """
    Narrates scenes using different voices for narrator and dialogue.

    For each scene:
    1. Parse narration text to identify dialogue vs narration
    2. Detect which character is speaking
    3. Synthesize each segment with the appropriate voice
    4. Concatenate into one audio file
    """

    def __init__(
        self,
        narrator_voice: str = DEFAULT_NARRATOR_VOICE,
        rate: str = DEFAULT_RATE,
        voice_assigner: VoiceAssigner | None = None,
    ):
        self.narrator_voice = narrator_voice
        self.rate = rate
        self.voice_assigner = voice_assigner or VoiceAssigner()
        self._narrators: dict[str, EdgeTTSNarrator] = {}

    def _get_narrator(self, voice: str) -> EdgeTTSNarrator:
        """Get or create a narrator for a specific voice."""
        if voice not in self._narrators:
            self._narrators[voice] = EdgeTTSNarrator(voice=voice, rate=self.rate)
        return self._narrators[voice]

    def assign_character_voices(self, characters: list[dict]) -> dict[str, str]:
        """
        Pre-assign voices to all characters.

        Args:
            characters: List of dicts with 'name' and optionally 'gender', 'role'.
        """
        for char in characters:
            self.voice_assigner.assign(
                char["name"],
                gender=char.get("gender", "unknown"),
                role=char.get("role", "minor"),
            )
        return self.voice_assigner.get_assignments()

    def synthesize(self, text: str, output_path: str, characters_present: list[str] | None = None) -> TTSResult:
        """
        Synthesize narration with multi-voice support.

        Parses the text for dialogue, assigns voices, and concatenates.
        """
        segments = self._parse_segments(text, characters_present or [])

        if len(segments) <= 1:
            # Simple case: just narration, use narrator voice
            narrator = self._get_narrator(segments[0]["voice"] if segments else self.narrator_voice)
            clean_text = _strip_speaker_tags(segments[0]["text"] if segments else text)
            return narrator.synthesize(clean_text, output_path)

        # Multi-segment: synthesize each, then concatenate
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(output_path).parent / "_temp"
        temp_dir.mkdir(exist_ok=True)

        combined = AudioSegment.empty()
        all_cues: list[SubtitleCue] = []
        cumulative_ms = 0

        for i, seg in enumerate(segments):
            temp_path = str(temp_dir / f"seg_{i:03d}.mp3")
            voice = seg["voice"]
            narrator = self._get_narrator(voice)
            # Strip speaker tags before sending to TTS — voice is already assigned
            clean_text = _strip_speaker_tags(seg["text"])
            result = narrator.synthesize(clean_text, temp_path)

            audio = AudioSegment.from_file(temp_path)
            combined += audio

            # Offset cues
            for cue in result.cues:
                all_cues.append(SubtitleCue(
                    text=cue.text,
                    start_ms=cue.start_ms + cumulative_ms,
                    end_ms=cue.end_ms + cumulative_ms,
                ))

            cumulative_ms += len(audio)

        # Export combined
        combined.export(output_path, format="mp3")

        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return TTSResult(
            audio_path=output_path,
            duration_ms=len(combined),
            cues=all_cues if all_cues else [SubtitleCue(text=text, start_ms=0, end_ms=len(combined))],
        )

    def synthesize_scenes(
        self,
        scenes: list[dict],
        output_dir: str,
    ) -> list[TTSResult]:
        """Synthesize all scenes with multi-voice support."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for scene in scenes:
            scene_id = scene["id"]
            text = scene["narration_text"]
            chars = scene.get("characters_present", [])

            if not text.strip():
                continue

            output_path = str(output_dir / f"{scene_id}.mp3")
            result = self.synthesize(text, output_path, characters_present=chars)
            results.append(result)

        return results

    def _parse_segments(self, text: str, characters_present: list[str]) -> list[dict]:
        """
        Parse text into segments of narration and dialogue.

        First tries rewriter format: [Speaker]："dialogue"
        Then falls back to generic dialogue detection.
        """
        # Try rewriter tagged format first: [角色名]："对话"
        rewriter_matches = list(_REWRITER_TAG_PATTERN.finditer(text))
        if rewriter_matches:
            return self._parse_rewriter_format(text, rewriter_matches, characters_present)

        # Fallback: generic dialogue detection
        return self._parse_generic_dialogue(text, characters_present)

    def _parse_rewriter_format(self, text: str, matches: list, characters_present: list[str]) -> list[dict]:
        """Parse text with explicit [Speaker]："dialogue" tags."""
        segments = []
        last_end = 0

        for match in matches:
            # Narration before this dialogue
            before = text[last_end:match.start()].strip()
            # Clean bracket annotations from narration
            before = re.sub(r'\[(?:画面|外貌)[：:].+?\]', '', before).strip()
            if before:
                segments.append({
                    "text": before,
                    "voice": self.narrator_voice,
                    "type": "narration",
                })

            speaker = match.group(1)
            dialogue_text = match.group(2)
            voice = self.voice_assigner.assign(speaker)

            segments.append({
                "text": dialogue_text,
                "voice": voice,
                "type": "dialogue",
                "speaker": speaker,
            })

            last_end = match.end()

        # Remaining narration
        remaining = text[last_end:].strip()
        remaining = re.sub(r'\[(?:画面|外貌)[：:].+?\]', '', remaining).strip()
        if remaining:
            segments.append({
                "text": remaining,
                "voice": self.narrator_voice,
                "type": "narration",
            })

        return segments if segments else [{"text": text, "voice": self.narrator_voice, "type": "narration"}]

    def _parse_generic_dialogue(self, text: str, characters_present: list[str]) -> list[dict]:
        """Fallback: detect dialogue from quotes and speaker attribution."""
        segments = []
        last_end = 0

        for match in _DIALOGUE_PATTERN.finditer(text):
            before = text[last_end:match.start()].strip()
            if before:
                segments.append({
                    "text": before,
                    "voice": self.narrator_voice,
                    "type": "narration",
                })

            dialogue_text = match.group(0)
            speaker = self._detect_speaker(text[max(0, match.start() - 20):match.start()], characters_present)
            voice = self.voice_assigner.assign(speaker) if speaker else self.narrator_voice

            segments.append({
                "text": dialogue_text,
                "voice": voice,
                "type": "dialogue",
                "speaker": speaker,
            })
            last_end = match.end()

        remaining = text[last_end:].strip()
        if remaining:
            segments.append({"text": remaining, "voice": self.narrator_voice, "type": "narration"})

        return segments if segments else [{"text": text, "voice": self.narrator_voice, "type": "narration"}]

    def _detect_speaker(self, context: str, characters_present: list[str]) -> str | None:
        """Try to detect who is speaking from the context before dialogue."""
        match = _SPEAKER_PATTERN.search(context)
        if match:
            name = match.group(1)
            # Check if it matches any known character
            for char_name in characters_present:
                if name in char_name or char_name in name:
                    return char_name
            # Return the detected name even if not in character list
            return name
        return None
