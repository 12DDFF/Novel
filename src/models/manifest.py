from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from .enums import (
    AssetStatus,
    GPUProvider,
    ImageModel,
    Mood,
    ProjectStatus,
    TransitionType,
    TTSBackend,
)


class Source(BaseModel):
    """Where the content was scraped from."""
    platform: str = ""
    novel_id: str = ""
    novel_title: str = ""
    chapter_id: str = ""
    chapter_title: str = ""
    chapter_number: int = 0
    raw_text_path: str = ""


class Settings(BaseModel):
    """Pipeline settings for a project."""
    target_language: str = "zh"
    narration_language: str = "zh"
    image_style: str = "anime"
    image_model: ImageModel = ImageModel.FLUX_DEV
    tts_backend: TTSBackend = TTSBackend.EDGE_TTS
    tts_voice: str = "zh-CN-YunxiNeural"
    gpu_provider: GPUProvider = GPUProvider.RUNPOD
    target_resolution: tuple[int, int] = (1080, 1920)
    target_fps: int = 30
    image_hold_seconds: float = 10.0
    scenes_per_1000_chars: int = 4
    max_scene_narration_length: int = 300


class Character(BaseModel):
    """A character extracted from the story."""
    id: str = Field(default_factory=lambda: f"char_{uuid.uuid4().hex[:8]}")
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    role: str = ""  # protagonist, antagonist, side
    reference_image_path: str | None = None
    image_prompt_prefix: str = ""


class SceneAssets(BaseModel):
    """Generated assets for a single scene."""
    image_path: str | None = None
    image_status: AssetStatus = AssetStatus.PENDING
    audio_path: str | None = None
    audio_status: AssetStatus = AssetStatus.PENDING
    subtitle_path: str | None = None
    subtitle_status: AssetStatus = AssetStatus.PENDING


class QualityReview(BaseModel):
    """Quality review results for a scene."""
    image_approved: bool = False
    audio_approved: bool = False
    auto_checks_passed: bool = False
    notes: str = ""
    retry_count: int = 0


class Scene(BaseModel):
    """A single visual scene in the video."""
    id: str = Field(default_factory=lambda: f"scene_{uuid.uuid4().hex[:8]}")
    sequence: int = 0
    narration_text: str = ""
    visual_description: str = ""
    characters_present: list[str] = Field(default_factory=list)  # character IDs
    mood: Mood = Mood.DRAMATIC
    setting: str = ""
    image_prompt: str = ""
    motion_description: str = ""
    transition: TransitionType = TransitionType.CROSSFADE
    duration_estimate_seconds: float = 10.0
    assets: SceneAssets = Field(default_factory=SceneAssets)
    quality_review: QualityReview = Field(default_factory=QualityReview)


class CostEntry(BaseModel):
    """A single cost event."""
    timestamp: datetime = Field(default_factory=datetime.now)
    category: str = ""  # gpu, llm, tts, image
    amount: float = 0.0
    details: str = ""


class Output(BaseModel):
    """Final video output info."""
    final_video_path: str | None = None
    total_duration_seconds: float | None = None
    status: ProjectStatus = ProjectStatus.CREATED


class Manifest(BaseModel):
    """Root manifest: the full state of a video project."""
    project_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    project_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source: Source = Field(default_factory=Source)
    settings: Settings = Field(default_factory=Settings)
    characters: list[Character] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    costs: list[CostEntry] = Field(default_factory=list)
    output: Output = Field(default_factory=Output)

    def get_character(self, char_id: str) -> Character | None:
        """Find a character by ID."""
        for char in self.characters:
            if char.id == char_id:
                return char
        return None

    def get_character_by_name(self, name: str) -> Character | None:
        """Find a character by name or alias."""
        for char in self.characters:
            if char.name == name or name in char.aliases:
                return char
        return None

    def get_scene(self, scene_id: str) -> Scene | None:
        """Find a scene by ID."""
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

    def get_scenes_by_status(self, status: AssetStatus, asset_type: str = "image") -> list[Scene]:
        """Get all scenes where a specific asset has the given status."""
        results = []
        for scene in self.scenes:
            asset_status = getattr(scene.assets, f"{asset_type}_status", None)
            if asset_status == status:
                results.append(scene)
        return results

    def total_cost(self) -> float:
        """Sum all cost entries."""
        return sum(c.amount for c in self.costs)

    def add_cost(self, category: str, amount: float, details: str = "") -> None:
        """Record a cost event."""
        self.costs.append(CostEntry(category=category, amount=amount, details=details))

    def progress_summary(self) -> dict[str, dict[str, int]]:
        """Get count of each asset status per asset type."""
        summary: dict[str, dict[str, int]] = {}
        for asset_type in ("image", "audio", "subtitle"):
            counts: dict[str, int] = {}
            for scene in self.scenes:
                status = getattr(scene.assets, f"{asset_type}_status").value
                counts[status] = counts.get(status, 0) + 1
            summary[asset_type] = counts
        return summary
