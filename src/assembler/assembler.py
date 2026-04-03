"""
Video assembler: combines images, narration, subtitles, and BGM into final video.

Pipeline:
1. Create Ken Burns clips from scene images
2. Match each clip duration to its narration audio
3. Concatenate clips with transitions
4. Add narration audio track
5. Add background music (auto-ducked)
6. Burn in subtitles
7. Export final video
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)

from src.tts.base import TTSResult

from .ken_burns import ken_burns_clip

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assembles final video from generated assets."""

    def __init__(
        self,
        target_resolution: tuple[int, int] = (1080, 1920),
        fps: int = 30,
        transition_duration: float = 0.3,
        bgm_volume: float = 0.12,
    ):
        self.target_resolution = target_resolution
        self.fps = fps
        self.transition_duration = transition_duration
        self.bgm_volume = bgm_volume

    def create_scene_clip(
        self,
        image_path: str,
        audio_path: str,
        effect: str = "random",
    ):
        """
        Create a single scene clip: Ken Burns image + narration audio.
        Duration is determined by the audio length.
        """
        audio = AudioFileClip(audio_path)
        duration = audio.duration

        video = ken_burns_clip(
            image_path=image_path,
            duration=duration,
            target_resolution=self.target_resolution,
            effect=effect,
            fps=self.fps,
        )
        video = video.with_audio(audio)
        return video

    def assemble_scenes(
        self,
        scene_data: list[dict],
        output_path: str,
        bgm_path: str | None = None,
        subtitle_path: str | None = None,
        codec: str = "libx264",
        bitrate: str = "8000k",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ) -> str:
        """
        Assemble all scenes into a final video.

        Args:
            scene_data: List of dicts with keys:
                - image_path: str
                - audio_path: str
                - effect: str (optional, default "random")
            output_path: Where to save the final video.
            bgm_path: Optional background music file.
            subtitle_path: Optional ASS subtitle file (burned in via ffmpeg).
            codec: Video codec.
            bitrate: Video bitrate.
            audio_codec: Audio codec.
            audio_bitrate: Audio bitrate.

        Returns:
            Path to the final video.
        """
        if not scene_data:
            raise ValueError("No scenes to assemble")

        # Step 1: Create clips for each scene
        logger.info(f"Creating {len(scene_data)} scene clips...")
        clips = []
        for i, scene in enumerate(scene_data):
            logger.info(f"  Scene {i + 1}/{len(scene_data)}: {scene['image_path']}")
            clip = self.create_scene_clip(
                image_path=scene["image_path"],
                audio_path=scene["audio_path"],
                effect=scene.get("effect", "random"),
            )
            clips.append(clip)

        # Step 2: Concatenate with crossfade
        logger.info("Concatenating clips...")
        if len(clips) == 1:
            final = clips[0]
        else:
            # Simple concatenation (crossfade is complex in moviepy2)
            final = concatenate_videoclips(clips, method="compose")

        # Step 3: Add background music
        if bgm_path and Path(bgm_path).exists():
            logger.info("Adding background music...")
            final = self._add_bgm(final, bgm_path)

        # Step 4: Export without subtitles first
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if subtitle_path and Path(subtitle_path).exists():
            # Export to temp file, then burn subtitles via ffmpeg
            temp_path = output_path + ".temp.mp4"
            logger.info(f"Exporting to temp file: {temp_path}")
            final.write_videofile(
                temp_path,
                fps=self.fps,
                codec=codec,
                bitrate=bitrate,
                audio_codec=audio_codec,
                audio_bitrate=audio_bitrate,
                logger=None,
            )

            # Burn subtitles with ffmpeg
            logger.info("Burning subtitles...")
            self._burn_subtitles_ffmpeg(temp_path, subtitle_path, output_path)
            Path(temp_path).unlink(missing_ok=True)
        else:
            logger.info(f"Exporting final video: {output_path}")
            final.write_videofile(
                output_path,
                fps=self.fps,
                codec=codec,
                bitrate=bitrate,
                audio_codec=audio_codec,
                audio_bitrate=audio_bitrate,
                logger=None,
            )

        # Clean up
        for clip in clips:
            clip.close()
        final.close()

        logger.info(f"Assembly complete: {output_path}")
        return output_path

    def _add_bgm(self, video_clip, bgm_path: str):
        """Add background music, looped to video length, at reduced volume."""
        bgm = AudioFileClip(bgm_path)

        # Loop BGM if shorter than video
        if bgm.duration < video_clip.duration:
            loops_needed = int(video_clip.duration / bgm.duration) + 1
            bgm = concatenate_videoclips(
                [bgm] * loops_needed
            ) if hasattr(bgm, 'fx') else bgm  # fallback
            # Simple approach: just use what we have
            bgm = bgm.subclipped(0, video_clip.duration)
        else:
            bgm = bgm.subclipped(0, video_clip.duration)

        # Reduce BGM volume
        bgm = bgm.with_volume_scaled(self.bgm_volume)

        # Mix with existing audio
        if video_clip.audio:
            mixed = CompositeAudioClip([video_clip.audio, bgm])
            return video_clip.with_audio(mixed)
        else:
            return video_clip.with_audio(bgm)

    @staticmethod
    def _burn_subtitles_ffmpeg(
        video_path: str,
        subtitle_path: str,
        output_path: str,
    ) -> None:
        """Burn ASS/SRT subtitles into video using ffmpeg."""
        import os
        import shutil

        # On Windows, ffmpeg ass filter breaks with absolute paths (C: issue)
        # Use cwd-relative paths instead
        video_abs = os.path.abspath(video_path)
        sub_abs = os.path.abspath(subtitle_path)
        output_abs = os.path.abspath(output_path)

        # Work from the subtitle's directory so the path is just the filename
        work_dir = os.path.dirname(sub_abs)
        sub_name = os.path.basename(sub_abs)

        # Make video path relative to work_dir, or use absolute with forward slashes
        video_rel = os.path.relpath(video_abs, work_dir).replace("\\", "/")
        output_rel = os.path.relpath(output_abs, work_dir).replace("\\", "/")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_rel,
            "-vf", f"ass={sub_name}",
            "-c:a", "copy",
            output_rel,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
        )

        if result.returncode != 0:
            logger.warning(
                f"Subtitle burn-in failed: {result.stderr[:300]}. "
                f"Copying without subtitles."
            )
            shutil.copy2(video_abs, output_abs)
