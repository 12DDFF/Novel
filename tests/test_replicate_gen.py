"""Tests for Replicate image generation backend."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from PIL import Image
import numpy as np

from src.image_gen.replicate_backend import ReplicateGenerator


class TestReplicateGenerator:
    def test_init_with_token(self):
        gen = ReplicateGenerator(api_token="test-token")
        assert gen.api_token == "test-token"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token")
        gen = ReplicateGenerator()
        assert gen.api_token == "env-token"

    def test_extract_url_string(self):
        assert ReplicateGenerator._extract_url("https://example.com/img.png") == "https://example.com/img.png"

    def test_extract_url_list(self):
        result = ReplicateGenerator._extract_url(["https://example.com/img.png"])
        assert result == "https://example.com/img.png"

    def test_stitch_references(self, tmp_path):
        # Create two test images
        img1 = Image.fromarray(np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (100, 60, 3), dtype=np.uint8))
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        img1.save(p1)
        img2.save(p2)

        output = str(tmp_path / "stitched.png")
        result = ReplicateGenerator._stitch_references([p1, p2], output)

        assert Path(result).exists()
        stitched = Image.open(result)
        assert stitched.width == 80 + 60  # combined width
        assert stitched.height == 100

    def test_stitch_references_different_heights(self, tmp_path):
        img1 = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8))
        p1 = str(tmp_path / "tall.png")
        p2 = str(tmp_path / "short.png")
        img1.save(p1)
        img2.save(p2)

        output = str(tmp_path / "stitched.png")
        result = ReplicateGenerator._stitch_references([p1, p2], output)
        stitched = Image.open(result)
        assert stitched.height == 100  # resized to shortest

    @patch("src.image_gen.replicate_backend.replicate.run")
    @patch("src.image_gen.replicate_backend.httpx.get")
    def test_generate_txt2img(self, mock_get, mock_run, tmp_path):
        # Mock replicate.run returns URL
        mock_run.return_value = ["https://example.com/output.png"]

        # Mock image download
        fake_img = Image.new("RGB", (1024, 1792), (255, 0, 0))
        import io
        buf = io.BytesIO()
        fake_img.save(buf, format="PNG")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        gen = ReplicateGenerator(api_token="fake")
        output = str(tmp_path / "test.png")
        result = gen.generate("anime warrior on cliff", output, seed=42)

        assert result.image_path == output
        assert result.seed == 42
        assert Path(output).exists()
        mock_run.assert_called_once()

    @patch("src.image_gen.replicate_backend.replicate.run")
    @patch("src.image_gen.replicate_backend.httpx.get")
    def test_generate_with_reference(self, mock_get, mock_run, tmp_path):
        # Create a reference image
        ref_img = Image.new("RGB", (512, 512), (0, 255, 0))
        ref_path = str(tmp_path / "ref.png")
        ref_img.save(ref_path)

        # Mock replicate output
        mock_run.return_value = ["https://example.com/scene.png"]

        # Mock download
        output_img = Image.new("RGB", (1024, 1792), (0, 0, 255))
        import io
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        gen = ReplicateGenerator(api_token="fake")
        output = str(tmp_path / "scene.png")
        result = gen.generate("character in classroom", output, reference_image=ref_path)

        assert Path(result.image_path).exists()
        # Should have called kontext model (image-to-image)
        call_args = mock_run.call_args
        assert "kontext" in call_args[0][0]

    @patch("src.image_gen.replicate_backend.replicate.run")
    @patch("src.image_gen.replicate_backend.httpx.get")
    def test_generate_character_sheet(self, mock_get, mock_run, tmp_path):
        mock_run.return_value = ["https://example.com/char.png"]

        char_img = Image.new("RGB", (1024, 1024), (128, 128, 128))
        import io
        buf = io.BytesIO()
        char_img.save(buf, format="PNG")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        gen = ReplicateGenerator(api_token="fake")
        output = str(tmp_path / "char.png")
        result = gen.generate_character_sheet("young woman, black hair, school uniform", output, style="anime")

        assert Path(result.image_path).exists()
        assert "character portrait" in mock_run.call_args[1]["input"]["prompt"]
        assert "anime" in mock_run.call_args[1]["input"]["prompt"]

    @patch("src.image_gen.replicate_backend.replicate.run")
    @patch("src.image_gen.replicate_backend.httpx.get")
    def test_generate_scene_multi_character(self, mock_get, mock_run, tmp_path):
        # Create two character refs
        for name in ["char1.png", "char2.png"]:
            img = Image.new("RGB", (512, 512), (100, 100, 100))
            img.save(str(tmp_path / name))

        mock_run.return_value = ["https://example.com/scene.png"]

        scene_img = Image.new("RGB", (1024, 1792), (50, 50, 50))
        import io
        buf = io.BytesIO()
        scene_img.save(buf, format="PNG")
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        gen = ReplicateGenerator(api_token="fake")
        output = str(tmp_path / "multi.png")
        result = gen.generate_scene(
            "two students talking in hallway",
            output,
            character_refs=[str(tmp_path / "char1.png"), str(tmp_path / "char2.png")],
        )

        assert Path(result.image_path).exists()
        # Should have created a stitched reference
        assert Path(output + ".refs.png").exists()

    def test_prepare_image_input_file(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        path = str(tmp_path / "test.png")
        img.save(path)
        result = ReplicateGenerator._prepare_image_input(path)
        assert hasattr(result, "read")  # file handle
        result.close()

    def test_prepare_image_input_url(self):
        result = ReplicateGenerator._prepare_image_input("https://example.com/img.png")
        assert result == "https://example.com/img.png"

    def test_prepare_image_input_missing(self):
        with pytest.raises(FileNotFoundError):
            ReplicateGenerator._prepare_image_input("/nonexistent/path.png")


# ── Integration Test (requires REPLICATE_API_TOKEN) ──────────────────────────


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set",
)
class TestReplicateIntegration:

    def test_generate_character_live(self, tmp_path):
        gen = ReplicateGenerator()
        output = str(tmp_path / "character.png")
        result = gen.generate_character_sheet(
            "young Chinese woman, long black hair, school uniform, gentle smile",
            output,
            style="anime",
        )
        assert Path(result.image_path).exists()
        assert Path(result.image_path).stat().st_size > 10000
        img = Image.open(result.image_path)
        assert img.width > 0
