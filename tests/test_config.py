"""Tests for PipelineConfig."""
from pathlib import Path

import pytest
import yaml

from src.core.config import PipelineConfig
from src.models.enums import GPUProvider, ImageModel, TTSBackend


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.gpu.provider == GPUProvider.RUNPOD
        assert config.llm.provider == "deepseek"
        assert config.image.model == ImageModel.FLUX_DEV
        assert config.tts.backend == TTSBackend.EDGE_TTS
        assert config.tts.voice == "zh-CN-YunxiNeural"
        assert config.video.image_hold_seconds == 10.0

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "image": {"style": "realistic", "model": "sdxl"},
            "tts": {"voice": "zh-CN-XiaoxiaoNeural"},
            "video": {"image_hold_seconds": 8.0},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.load(config_path=config_file)
        assert config.image.style == "realistic"
        assert config.image.model == ImageModel.SDXL
        assert config.tts.voice == "zh-CN-XiaoxiaoNeural"
        assert config.video.image_hold_seconds == 8.0
        # Defaults still work for unset values
        assert config.gpu.provider == GPUProvider.RUNPOD

    def test_load_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.setenv("RUNPOD_API_KEY", "rp-test-456")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        # Use non-existent env_path to prevent loading real .env file
        config = PipelineConfig.load(env_path=tmp_path / ".env.nonexistent")
        assert config.llm.api_key == "sk-test-123"
        assert config.gpu.runpod_api_key == "rp-test-456"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        config_data = {"llm": {"api_key": "from-yaml"}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        config = PipelineConfig.load(config_path=config_file, env_path=tmp_path / ".env.nonexistent")
        assert config.llm.api_key == "from-env"

    def test_save_excludes_api_keys(self, tmp_path):
        config = PipelineConfig()
        config.llm.api_key = "secret-key"
        config.gpu.runpod_api_key = "secret-runpod"

        save_path = tmp_path / "saved.yaml"
        config.save(save_path)

        with open(save_path) as f:
            saved = yaml.safe_load(f)

        assert "api_key" not in saved["llm"]
        assert "runpod_api_key" not in saved["gpu"]

    def test_set_value(self):
        config = PipelineConfig()
        config.set_value("image.style", "comic")
        assert config.image.style == "comic"

        config.set_value("video.target_fps", "24")
        assert config.video.target_fps == 24

        config.set_value("gpu.auto_start", "false")
        assert config.gpu.auto_start is False

    def test_set_value_invalid_section(self):
        config = PipelineConfig()
        with pytest.raises(ValueError, match="Unknown config section"):
            config.set_value("nonexistent.field", "value")

    def test_set_value_invalid_field(self):
        config = PipelineConfig()
        with pytest.raises(ValueError, match="Unknown field"):
            config.set_value("image.nonexistent", "value")

    def test_set_value_bad_format(self):
        config = PipelineConfig()
        with pytest.raises(ValueError, match="section.field"):
            config.set_value("just_one_part", "value")

    def test_load_nonexistent_yaml(self):
        """Should work fine with defaults when YAML doesn't exist."""
        config = PipelineConfig.load(config_path="/nonexistent/path.yaml")
        assert config.gpu.provider == GPUProvider.RUNPOD
