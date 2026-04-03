"""Base interface for image generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageResult:
    """Result of generating a single image."""
    image_path: str = ""
    prompt: str = ""
    seed: int = 0
    width: int = 0
    height: int = 0


class BaseImageGenerator(ABC):
    """Abstract base for image generation backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1792,
        seed: int = -1,
    ) -> ImageResult:
        """Generate a single image from a prompt."""

    @abstractmethod
    def generate_character_sheet(
        self,
        character_description: str,
        output_path: str,
        style: str = "anime",
    ) -> ImageResult:
        """Generate a character reference sheet."""

    def generate_batch(
        self,
        prompts: list[dict],
        output_dir: str,
    ) -> list[ImageResult]:
        """
        Generate multiple images.

        Args:
            prompts: List of dicts with 'prompt', 'filename', and optional 'seed'.
            output_dir: Directory to save images.
        """
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        for item in prompts:
            path = str(Path(output_dir) / item["filename"])
            result = self.generate(
                prompt=item["prompt"],
                output_path=path,
                seed=item.get("seed", -1),
            )
            results.append(result)
        return results
