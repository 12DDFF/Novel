"""Tests for video assembler."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.assembler.ken_burns import ken_burns_clip


class TestKenBurns:
    def _make_test_image(self, tmp_path, w=1200, h=2100):
        """Create a test image."""
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)
        return path

    def test_creates_clip(self, tmp_path):
        img = self._make_test_image(tmp_path)
        clip = ken_burns_clip(img, duration=2.0, target_resolution=(540, 960), fps=10)
        assert clip.duration == 2.0
        frame = clip.get_frame(0)
        assert frame.shape == (960, 540, 3)
        clip.close()

    def test_zoom_in(self, tmp_path):
        img = self._make_test_image(tmp_path)
        clip = ken_burns_clip(img, duration=1.0, target_resolution=(540, 960), effect="zoom_in", fps=10)
        f0 = clip.get_frame(0)
        f1 = clip.get_frame(0.9)
        assert f0.shape == f1.shape == (960, 540, 3)
        # Frames should differ (zoom is happening)
        assert not np.array_equal(f0, f1)
        clip.close()

    def test_pan_right(self, tmp_path):
        img = self._make_test_image(tmp_path)
        clip = ken_burns_clip(img, duration=1.0, target_resolution=(540, 960), effect="pan_right", fps=10)
        f0 = clip.get_frame(0)
        f1 = clip.get_frame(0.9)
        assert not np.array_equal(f0, f1)
        clip.close()


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "0") == "1",
    reason="Network tests disabled",
)
class TestAssemblerIntegration:
    """Integration test: image + TTS audio -> video clip."""

    def test_single_scene_assembly(self, tmp_path):
        from src.assembler.assembler import VideoAssembler
        from src.tts.edge_tts_backend import EdgeTTSNarrator

        # Generate test image
        img = Image.fromarray(np.random.randint(0, 255, (1792, 1024, 3), dtype=np.uint8))
        img_path = str(tmp_path / "scene.png")
        img.save(img_path)

        # Generate test audio
        narrator = EdgeTTSNarrator(voice="yunxi")
        audio_path = str(tmp_path / "narration.mp3")
        narrator.synthesize("这是一个测试场景。", audio_path)

        # Assemble
        assembler = VideoAssembler(target_resolution=(540, 960), fps=10)
        output = str(tmp_path / "output.mp4")
        result = assembler.assemble_scenes(
            [{"image_path": img_path, "audio_path": audio_path, "effect": "zoom_in"}],
            output,
        )

        assert Path(result).exists()
        assert Path(result).stat().st_size > 1000
