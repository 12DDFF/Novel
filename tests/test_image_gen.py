"""Tests for image generation module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.image_gen.base import ImageResult
from src.image_gen.placeholder import PlaceholderGenerator


class TestPlaceholderGenerator:
    def test_generate(self, tmp_path):
        gen = PlaceholderGenerator()
        output = str(tmp_path / "test.png")
        result = gen.generate("cinematic anime, warrior on cliff", output)

        assert Path(result.image_path).exists()
        assert result.width == 1024
        assert result.height == 1792
        assert result.prompt == "cinematic anime, warrior on cliff"

        img = Image.open(result.image_path)
        assert img.size == (1024, 1792)

    def test_generate_custom_size(self, tmp_path):
        gen = PlaceholderGenerator()
        output = str(tmp_path / "custom.png")
        result = gen.generate("test", output, width=512, height=512)
        img = Image.open(result.image_path)
        assert img.size == (512, 512)

    def test_generate_deterministic_seed(self, tmp_path):
        gen = PlaceholderGenerator()
        r1 = gen.generate("same prompt", str(tmp_path / "a.png"), seed=42)
        r2 = gen.generate("same prompt", str(tmp_path / "b.png"), seed=42)
        assert r1.seed == r2.seed == 42

    def test_generate_character_sheet(self, tmp_path):
        gen = PlaceholderGenerator()
        output = str(tmp_path / "char.png")
        result = gen.generate_character_sheet("tall warrior, black hair", output, style="anime")

        assert Path(result.image_path).exists()
        assert "CHARACTER SHEET" in result.prompt
        assert result.width == 1024
        assert result.height == 1024

    def test_generate_batch(self, tmp_path):
        gen = PlaceholderGenerator()
        prompts = [
            {"prompt": "scene 1 forest", "filename": "s1.png"},
            {"prompt": "scene 2 cave", "filename": "s2.png"},
            {"prompt": "scene 3 mountain", "filename": "s3.png"},
        ]
        results = gen.generate_batch(prompts, str(tmp_path / "batch"))

        assert len(results) == 3
        for r in results:
            assert Path(r.image_path).exists()

    def test_different_prompts_different_colors(self, tmp_path):
        gen = PlaceholderGenerator()
        r1 = gen.generate("forest scene", str(tmp_path / "a.png"))
        r2 = gen.generate("ocean scene", str(tmp_path / "b.png"))

        img1 = Image.open(r1.image_path)
        img2 = Image.open(r2.image_path)
        # Different prompts should produce different background colors
        assert img1.getpixel((500, 500)) != img2.getpixel((500, 500))


class TestComfyUIGenerator:
    def test_health_check_unreachable(self):
        from src.image_gen.comfyui import ComfyUIGenerator
        gen = ComfyUIGenerator(server_url="http://localhost:99999")
        assert gen.health_check() is False

    @patch("src.image_gen.comfyui.httpx.post")
    @patch("src.image_gen.comfyui.httpx.get")
    def test_queue_and_poll(self, mock_get, mock_post):
        from src.image_gen.comfyui import ComfyUIGenerator

        # Mock queue response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"prompt_id": "test-123"},
            raise_for_status=lambda: None,
        )

        # Mock history response (completed)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "test-123": {
                    "outputs": {
                        "9": {
                            "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                        }
                    }
                }
            },
            raise_for_status=lambda: None,
            content=b"fake image data",
        )

        gen = ComfyUIGenerator(server_url="http://fake:8188")
        images = gen._wait_for_result("test-123")
        assert len(images) == 1
        assert images[0]["filename"] == "out.png"
