from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.models.enums import GPUProvider, ImageModel, TTSBackend


class GPUConfig(BaseModel):
    provider: GPUProvider = GPUProvider.RUNPOD
    gpu_type: str = "RTX4090"
    auto_start: bool = True
    auto_stop: bool = True
    runpod_api_key: str = ""
    vastai_api_key: str = ""


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    api_key: str = ""
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"


class ImageConfig(BaseModel):
    model: ImageModel = ImageModel.FLUX_DEV
    resolution: tuple[int, int] = (1024, 1792)
    style: str = "anime"
    batch_size: int = 5
    max_retries: int = 2


class TTSConfig(BaseModel):
    backend: TTSBackend = TTSBackend.EDGE_TTS
    voice: str = "zh-CN-YunxiNeural"
    speed: float = 1.0


class VideoConfig(BaseModel):
    target_resolution: tuple[int, int] = (1080, 1920)
    target_fps: int = 30
    image_hold_seconds: float = 10.0
    transition_duration: float = 0.5
    bgm_volume: float = 0.15
    export_codec: str = "libx264"
    export_bitrate: str = "8000k"


class SegmenterConfig(BaseModel):
    scenes_per_1000_chars: int = 4
    max_scene_narration_length: int = 300


class PipelineConfig(BaseModel):
    """Top-level configuration for the entire pipeline."""
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    segmenter: SegmenterConfig = Field(default_factory=SegmenterConfig)
    projects_dir: str = ""

    @classmethod
    def load(cls, config_path: str | Path | None = None, env_path: str | Path | None = None) -> PipelineConfig:
        """
        Load config from YAML file + environment variables.
        Environment variables override YAML values.
        """
        data: dict = {}

        # Load YAML if provided
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

        # Load .env
        env_file = Path(env_path) if env_path else None
        load_dotenv(dotenv_path=env_file)

        config = cls.model_validate(data)

        # Override with environment variables
        # OpenRouter takes priority if LLM_PROVIDER=openrouter
        llm_provider = os.getenv("LLM_PROVIDER", "")
        if llm_provider == "openrouter" and (api_key := os.getenv("OPENROUTER_API_KEY")):
            config.llm.provider = "openrouter"
            config.llm.api_key = api_key
            config.llm.base_url = "https://openrouter.ai/api/v1"
            config.llm.model = "deepseek/deepseek-chat"
        elif api_key := os.getenv("DEEPSEEK_API_KEY"):
            config.llm.api_key = api_key
        if api_key := os.getenv("RUNPOD_API_KEY"):
            config.gpu.runpod_api_key = api_key
        if api_key := os.getenv("VASTAI_API_KEY"):
            config.gpu.vastai_api_key = api_key

        # Default projects dir
        if not config.projects_dir:
            config.projects_dir = str(Path.home() / "projects" / "novel-to-video" / "data")

        return config

    def save(self, config_path: str | Path) -> None:
        """Save config to YAML file (excludes API keys)."""
        data = self.model_dump(mode="json")
        # Strip API keys from saved config
        data["llm"].pop("api_key", None)
        data["gpu"].pop("runpod_api_key", None)
        data["gpu"].pop("vastai_api_key", None)

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def set_value(self, key: str, value: str) -> None:
        """
        Set a nested config value using dot notation.
        Example: config.set_value("image.model", "flux-schnell")
        """
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"Key must be in 'section.field' format, got: {key}")

        section_name, field_name = parts
        section = getattr(self, section_name, None)
        if section is None:
            raise ValueError(f"Unknown config section: {section_name}")
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown field '{field_name}' in section '{section_name}'")

        # Get the field type and cast
        field_info = type(section).model_fields[field_name]
        field_type = field_info.annotation

        if field_type == bool:
            value = value.lower() in ("true", "1", "yes")
        elif field_type == int:
            value = int(value)
        elif field_type == float:
            value = float(value)

        setattr(section, field_name, value)
