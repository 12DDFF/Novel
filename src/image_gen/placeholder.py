"""
Placeholder image generator for testing the full pipeline without a GPU.

Generates colored images with the prompt text overlaid so you can verify
the pipeline works end-to-end before connecting real image generation.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .base import BaseImageGenerator, ImageResult


class PlaceholderGenerator(BaseImageGenerator):
    """Generates placeholder images with prompt text for pipeline testing."""

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1792,
        seed: int = -1,
    ) -> ImageResult:
        if seed == -1:
            seed = random.randint(0, 2**32)

        # Deterministic color from prompt
        color_hash = hashlib.md5(prompt.encode()).hexdigest()
        bg_color = (
            int(color_hash[:2], 16),
            int(color_hash[2:4], 16),
            int(color_hash[4:6], 16),
        )

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw prompt text (wrapped)
        text_color = (255, 255, 255) if sum(bg_color) < 384 else (0, 0, 0)
        margin = 40
        y = margin
        max_chars = max(1, (width - margin * 2) // 16)

        lines = []
        for i in range(0, len(prompt), max_chars):
            lines.append(prompt[i:i + max_chars])

        for line in lines[:20]:  # max 20 lines
            draw.text((margin, y), line, fill=text_color)
            y += 24

        # Draw a border
        draw.rectangle(
            [10, 10, width - 10, height - 10],
            outline=text_color,
            width=2,
        )

        # Label
        draw.text((margin, height - 60), f"PLACEHOLDER | {width}x{height}", fill=text_color)
        draw.text((margin, height - 36), f"seed: {seed}", fill=text_color)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")

        return ImageResult(
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
        )

    def generate_character_sheet(
        self,
        character_description: str,
        output_path: str,
        style: str = "anime",
    ) -> ImageResult:
        prompt = f"[CHARACTER SHEET] {style} style | {character_description}"
        return self.generate(prompt, output_path, width=1024, height=1024)
