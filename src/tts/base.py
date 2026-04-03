from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubtitleCue:
    """A single subtitle timing entry."""
    text: str = ""
    start_ms: int = 0  # milliseconds from start
    end_ms: int = 0    # milliseconds from start


@dataclass
class TTSResult:
    """Result of synthesizing one scene's narration."""
    audio_path: str = ""
    duration_ms: int = 0
    cues: list[SubtitleCue] = field(default_factory=list)
    sample_rate: int = 24000
