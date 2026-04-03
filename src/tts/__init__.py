from .audio_processing import (
    add_silence_padding,
    concatenate_audio,
    get_duration_ms,
    normalize_volume,
    process_scene_audio,
)
from .base import SubtitleCue, TTSResult
from .edge_tts_backend import EdgeTTSNarrator

__all__ = [
    "EdgeTTSNarrator",
    "SubtitleCue",
    "TTSResult",
    "add_silence_padding",
    "concatenate_audio",
    "get_duration_ms",
    "normalize_volume",
    "process_scene_audio",
]
