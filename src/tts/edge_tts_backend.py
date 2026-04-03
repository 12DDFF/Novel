"""
Edge TTS backend — free Microsoft neural TTS via edge-tts library.

Supports Chinese (zh-CN) with multiple voices. Provides sentence-level
timing boundaries for subtitle synchronization.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import edge_tts

from .base import SubtitleCue, TTSResult

logger = logging.getLogger(__name__)

# Popular Chinese voices
VOICES = {
    "yunxi": "zh-CN-YunxiNeural",       # Male, narrator style
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # Female, warm
    "yunyang": "zh-CN-YunyangNeural",    # Male, professional
    "xiaoyi": "zh-CN-XiaoyiNeural",     # Female, gentle
}


def _split_sentences(text: str) -> list[str]:
    """
    Split Chinese text into sentences on punctuation boundaries.
    Used to generate subtitle cues when edge-tts only gives sentence-level timing.
    """
    parts = re.split(r'([。！？；…\n])', text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if part in ("。", "！", "？", "；", "…", "\n"):
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


class EdgeTTSNarrator:
    """Text-to-speech using Microsoft Edge TTS (free, no API key)."""

    def __init__(
        self,
        voice: str = "zh-CN-YunxiNeural",
        rate: str = "+0%",
        volume: str = "+0%",
    ):
        """
        Args:
            voice: Voice ID (e.g., "zh-CN-YunxiNeural") or shortname (e.g., "yunxi").
            rate: Speed adjustment (e.g., "+10%", "-20%").
            volume: Volume adjustment (e.g., "+0%").
        """
        self.voice = VOICES.get(voice, voice)
        self.rate = rate
        self.volume = volume

    async def synthesize_async(self, text: str, output_path: str) -> TTSResult:
        """
        Synthesize text to audio file with subtitle timing.

        Args:
            text: Chinese text to speak.
            output_path: Path to save the .mp3 file.

        Returns:
            TTSResult with audio path, duration, and subtitle cues.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        audio_data = b""
        sentence_boundaries = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
            elif chunk["type"] == "SentenceBoundary":
                sentence_boundaries.append({
                    "offset_ms": chunk["offset"] // 10000,  # 100ns ticks -> ms
                    "duration_ms": chunk["duration"] // 10000,
                    "text": chunk["text"],
                })

        if not audio_data:
            raise RuntimeError(f"No audio generated for text: {text[:50]}...")

        # Write audio file
        with open(output_path, "wb") as f:
            f.write(audio_data)

        # Build subtitle cues from sentence boundaries
        cues = self._build_cues(text, sentence_boundaries)

        # Get actual duration from audio
        duration_ms = self._get_audio_duration_ms(output_path)

        return TTSResult(
            audio_path=output_path,
            duration_ms=duration_ms,
            cues=cues,
        )

    def synthesize(self, text: str, output_path: str) -> TTSResult:
        """Synchronous wrapper for synthesize_async."""
        return asyncio.run(self.synthesize_async(text, output_path))

    def synthesize_scenes(
        self,
        scenes: list[dict],
        output_dir: str,
    ) -> list[TTSResult]:
        """
        Synthesize narration for multiple scenes.

        Args:
            scenes: List of dicts with 'id' and 'narration_text' keys.
            output_dir: Directory to save audio files.

        Returns:
            List of TTSResult objects.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for scene in scenes:
            scene_id = scene["id"]
            text = scene["narration_text"]
            if not text.strip():
                logger.warning(f"Skipping scene {scene_id}: empty narration text")
                continue

            output_path = str(output_dir / f"{scene_id}.mp3")
            logger.info(f"Synthesizing scene {scene_id}: {text[:30]}...")
            result = self.synthesize(text, output_path)
            results.append(result)

        return results

    def _build_cues(
        self,
        text: str,
        sentence_boundaries: list[dict],
    ) -> list[SubtitleCue]:
        """
        Build subtitle cues from sentence boundaries.

        If edge-tts provides sentence boundaries, use those directly.
        If not (or too few), generate cues by splitting text into sentences
        and distributing timing evenly.
        """
        cues = []

        if sentence_boundaries:
            for sb in sentence_boundaries:
                cues.append(SubtitleCue(
                    text=sb["text"],
                    start_ms=sb["offset_ms"],
                    end_ms=sb["offset_ms"] + sb["duration_ms"],
                ))
            return cues

        # Fallback: split by punctuation and estimate timing
        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Estimate ~200ms per Chinese character
        total_chars = sum(len(s) for s in sentences)
        ms_per_char = 250  # rough estimate for Chinese speech

        current_ms = 0
        for sentence in sentences:
            duration = len(sentence) * ms_per_char
            cues.append(SubtitleCue(
                text=sentence,
                start_ms=current_ms,
                end_ms=current_ms + duration,
            ))
            current_ms += duration

        return cues

    @staticmethod
    def _get_audio_duration_ms(audio_path: str) -> int:
        """Get audio duration in milliseconds using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio)
        except Exception:
            # Fallback: estimate from file size (mp3 ~16kbps for speech)
            file_size = Path(audio_path).stat().st_size
            return int(file_size / 16 * 8)  # rough estimate

    @staticmethod
    async def list_voices(language: str = "zh") -> list[dict]:
        """List available voices for a language."""
        voices = await edge_tts.list_voices()
        return [
            {"name": v["Name"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in voices
            if v["Locale"].startswith(language)
        ]
