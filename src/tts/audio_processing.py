"""
Audio post-processing utilities.
Normalization, silence padding, concatenation.
"""

from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment


def normalize_volume(audio: AudioSegment, target_dbfs: float = -16.0) -> AudioSegment:
    """Normalize audio volume to a target dBFS level."""
    if audio.dBFS == float("-inf"):
        return audio  # silent audio, can't normalize
    change = target_dbfs - audio.dBFS
    return audio.apply_gain(change)


def add_silence_padding(
    audio: AudioSegment,
    start_ms: int = 300,
    end_ms: int = 300,
) -> AudioSegment:
    """Add silence padding at the start and end of audio."""
    silence_start = AudioSegment.silent(duration=start_ms)
    silence_end = AudioSegment.silent(duration=end_ms)
    return silence_start + audio + silence_end


def get_duration_ms(audio_path: str) -> int:
    """Get duration of an audio file in milliseconds."""
    audio = AudioSegment.from_file(audio_path)
    return len(audio)


def concatenate_audio(
    audio_paths: list[str],
    output_path: str,
    crossfade_ms: int = 0,
    normalize: bool = True,
    target_dbfs: float = -16.0,
) -> str:
    """
    Concatenate multiple audio files into one.

    Args:
        audio_paths: List of audio file paths.
        output_path: Where to save the concatenated audio.
        crossfade_ms: Crossfade duration between clips (0 = no crossfade).
        normalize: Whether to normalize volume.
        target_dbfs: Target volume level.

    Returns:
        Path to the output file.
    """
    if not audio_paths:
        raise ValueError("No audio files to concatenate")

    combined = AudioSegment.from_file(audio_paths[0])
    if normalize:
        combined = normalize_volume(combined, target_dbfs)

    for path in audio_paths[1:]:
        segment = AudioSegment.from_file(path)
        if normalize:
            segment = normalize_volume(segment, target_dbfs)
        if crossfade_ms > 0:
            combined = combined.append(segment, crossfade=crossfade_ms)
        else:
            combined = combined + segment

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3")
    return output_path


def process_scene_audio(
    audio_path: str,
    output_path: str | None = None,
    target_dbfs: float = -16.0,
    pad_start_ms: int = 200,
    pad_end_ms: int = 300,
) -> str:
    """
    Post-process a single scene's audio: normalize + pad.

    Args:
        audio_path: Input audio file.
        output_path: Where to save (defaults to overwriting input).
        target_dbfs: Target volume.
        pad_start_ms: Silence to add at start.
        pad_end_ms: Silence to add at end.

    Returns:
        Path to the processed audio.
    """
    if output_path is None:
        output_path = audio_path

    audio = AudioSegment.from_file(audio_path)
    audio = normalize_volume(audio, target_dbfs)
    audio = add_silence_padding(audio, pad_start_ms, pad_end_ms)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="mp3")
    return output_path
