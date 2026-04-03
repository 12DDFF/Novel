"""
Subtitle generator: produces SRT and ASS files from TTS timing cues.
"""

from __future__ import annotations

from pathlib import Path

import pysubs2

from src.tts.base import SubtitleCue, TTSResult

from .chunker import split_into_subtitle_groups

# Default ASS style
_DEFAULT_STYLE = pysubs2.SSAStyle(
    fontname="Microsoft YaHei",
    fontsize=7,
    primarycolor=pysubs2.Color(255, 255, 255),  # white
    outlinecolor=pysubs2.Color(0, 0, 0),         # black outline
    backcolor=pysubs2.Color(0, 0, 0, 128),        # semi-transparent bg
    outline=0.8,
    shadow=0.3,
    alignment=2,  # bottom center
    marginl=40,
    marginr=40,
    marginv=55,   # lower, just above bottom UI elements
    bold=False,
)


class SubtitleGenerator:
    """Generates SRT and ASS subtitle files from TTS results."""

    def __init__(
        self,
        style: pysubs2.SSAStyle | None = None,
        max_chars_per_line: int = 20,
        max_lines: int = 4,
        min_duration_ms: int = 800,
    ):
        self.style = style or _DEFAULT_STYLE
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.min_duration_ms = min_duration_ms

    def generate_from_results(
        self,
        tts_results: list[TTSResult],
        output_path: str,
        format: str = "ass",
    ) -> str:
        """
        Generate a single subtitle file from multiple TTS results (scenes).

        The cues from each scene are offset by the cumulative duration
        of previous scenes, producing one continuous subtitle track.
        """
        subs = pysubs2.SSAFile()
        subs.styles["Default"] = self.style

        cumulative_ms = 0
        for result in tts_results:
            for cue in result.cues:
                self._add_cue(subs, cue, offset_ms=cumulative_ms)
            cumulative_ms += result.duration_ms

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subs.save(output_path, format_=format)
        return output_path

    def generate_for_scene(
        self,
        tts_result: TTSResult,
        output_path: str,
        format: str = "ass",
    ) -> str:
        """Generate subtitle file for a single scene."""
        subs = pysubs2.SSAFile()
        subs.styles["Default"] = self.style

        for cue in tts_result.cues:
            self._add_cue(subs, cue, offset_ms=0)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subs.save(output_path, format_=format)
        return output_path

    def generate_from_text_and_duration(
        self,
        text: str,
        duration_ms: int,
        output_path: str,
        format: str = "ass",
    ) -> str:
        """
        Generate subtitles from plain text + total duration.
        Distributes timing evenly across subtitle chunks.
        Useful when no TTS cues are available.
        """
        groups = split_into_subtitle_groups(
            text, self.max_chars_per_line, self.max_lines
        )
        if not groups:
            # Write empty subtitle file
            subs = pysubs2.SSAFile()
            subs.styles["Default"] = self.style
            subs.save(output_path, format_=format)
            return output_path

        # Distribute time evenly across groups, weighted by character count
        total_chars = sum(len(g.replace("\n", "")) for g in groups)
        if total_chars == 0:
            total_chars = 1

        subs = pysubs2.SSAFile()
        subs.styles["Default"] = self.style

        current_ms = 0
        for group in groups:
            char_count = len(group.replace("\n", ""))
            group_duration = max(
                self.min_duration_ms,
                int(duration_ms * char_count / total_chars),
            )
            end_ms = min(current_ms + group_duration, duration_ms)

            event = pysubs2.SSAEvent(
                start=current_ms,
                end=end_ms,
                text=group.replace("\n", "\\N"),  # ASS line break
                style="Default",
            )
            subs.events.append(event)
            current_ms = end_ms

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subs.save(output_path, format_=format)
        return output_path

    def _add_cue(
        self,
        subs: pysubs2.SSAFile,
        cue: SubtitleCue,
        offset_ms: int = 0,
    ) -> None:
        """Add a single cue to the subtitle file, splitting into display groups."""
        groups = split_into_subtitle_groups(
            cue.text, self.max_chars_per_line, self.max_lines
        )
        if not groups:
            return

        total_chars = sum(len(g.replace("\n", "")) for g in groups)
        if total_chars == 0:
            return

        cue_duration = cue.end_ms - cue.start_ms
        current_ms = cue.start_ms + offset_ms

        for group in groups:
            char_count = len(group.replace("\n", ""))
            group_duration = max(
                self.min_duration_ms,
                int(cue_duration * char_count / total_chars),
            )
            end_ms = current_ms + group_duration

            event = pysubs2.SSAEvent(
                start=current_ms,
                end=end_ms,
                text=group.replace("\n", "\\N"),
                style="Default",
            )
            subs.events.append(event)
            current_ms = end_ms
