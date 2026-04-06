"""
Replicate API backend for Flux Kontext image generation.

Two-phase approach:
1. Character design: Flux Dev (text-to-image) to create master reference
2. Scene generation: Flux Kontext Dev (image-to-image) to place character in scenes

No GPU needed — everything runs on Replicate's cloud.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
from pathlib import Path

import httpx
import replicate

from .base import BaseImageGenerator, ImageResult

logger = logging.getLogger(__name__)

# Model IDs on Replicate
FLUX_DEV_MODEL = "black-forest-labs/flux-dev"
FLUX_KONTEXT_DEV_MODEL = "black-forest-labs/flux-kontext-dev"
FLUX_KONTEXT_PRO_MODEL = "black-forest-labs/flux-kontext-pro"


class ReplicateGenerator(BaseImageGenerator):
    """
    Image generator using Replicate API with Flux models.

    - Character design: Flux Dev (text-to-image)
    - Scene generation: Flux Kontext Dev (image-to-image with character reference)
    """

    def __init__(
        self,
        api_token: str | None = None,
        kontext_model: str = FLUX_KONTEXT_PRO_MODEL,
        txt2img_model: str = FLUX_DEV_MODEL,
    ):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN", "")
        if self.api_token:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

        self.kontext_model = kontext_model
        self.txt2img_model = txt2img_model

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1792,
        seed: int = -1,
        reference_image: str | None = None,
    ) -> ImageResult:
        """
        Generate an image. If reference_image is provided, uses Flux Kontext
        (character-consistent image-to-image). Otherwise uses Flux Dev (text-to-image).
        """
        if seed == -1:
            seed = random.randint(0, 2**32)

        if reference_image:
            return self._generate_kontext(prompt, output_path, reference_image, seed)
        else:
            return self._generate_txt2img(prompt, output_path, width, height, seed)

    def generate_scene(
        self,
        prompt: str,
        output_path: str,
        character_refs: list[str],
        seed: int = -1,
    ) -> ImageResult:
        """
        Generate a scene image with character consistency.

        For single character: uses the reference directly.
        For multiple characters: stitches references and uses grouped prompt.
        """
        if seed == -1:
            seed = random.randint(0, 2**32)

        if len(character_refs) == 1:
            return self._generate_kontext(prompt, output_path, character_refs[0], seed)
        elif len(character_refs) > 1:
            # For multiple characters, stitch references into one image
            stitched = self._stitch_references(character_refs, output_path + ".refs.png")
            return self._generate_kontext(prompt, output_path, stitched, seed)
        else:
            # No character refs, just text-to-image
            return self._generate_txt2img(prompt, output_path, 1024, 1792, seed)

    def generate_character_sheet(
        self,
        character_description: str,
        output_path: str,
        style: str = "anime",
    ) -> ImageResult:
        """Generate a character master reference using Flux Dev (text-to-image)."""
        prompt = (
            f"anime illustration, front-facing character portrait with three-quarter body framing, "
            f"{character_description}, "
            f"arms relaxed at sides with hands visible and empty, "
            f"NOT holding any weapons or objects or tools, "
            f"neutral solid color background, soft even studio lighting from the front, "
            f"detailed face with sharp focus on facial features, clean linework, "
            f"original character design, NOT resembling any existing anime or manga character, "
            f"natural hand proportions, correct human anatomy"
        )
        return self._generate_txt2img(prompt, output_path, 1024, 1024, random.randint(0, 2**32))

    def generate_character_turnaround(
        self,
        reference_image: str,
        output_dir: str,
        angles: list[str] | None = None,
    ) -> list[ImageResult]:
        """
        Generate multi-angle views of a character from a reference image.
        Uses Flux Kontext to maintain consistency.
        """
        if angles is None:
            angles = [
                "same character, 3/4 view from the left, same outfit and features",
                "same character, side profile view, same outfit and features",
                "same character, 3/4 view from the right, same outfit and features",
                "same character, view from behind, same outfit and features",
            ]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for i, angle_prompt in enumerate(angles):
            output_path = str(Path(output_dir) / f"angle_{i}.png")
            result = self._generate_kontext(angle_prompt, output_path, reference_image)
            results.append(result)

        return results

    def _generate_txt2img(
        self,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        seed: int,
    ) -> ImageResult:
        """Text-to-image using Flux Dev."""
        logger.info(f"Generating txt2img: {prompt[:60]}...")

        output = replicate.run(
            self.txt2img_model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "num_outputs": 1,
                "output_format": "png",
            },
        )

        # Output is a list of FileOutput objects or URLs
        image_url = self._extract_url(output)
        self._download_image(image_url, output_path)

        return ImageResult(
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
        )

    def _generate_kontext(
        self,
        prompt: str,
        output_path: str,
        reference_image: str,
        seed: int = -1,
    ) -> ImageResult:
        """Image-to-image using Flux Kontext Dev (character-consistent generation)."""
        if seed == -1:
            seed = random.randint(0, 2**32)

        logger.info(f"Generating kontext: {prompt[:60]}...")

        # Prepare the reference image input
        image_input = self._prepare_image_input(reference_image)

        try:
            output = replicate.run(
                self.kontext_model,
                input={
                    "prompt": prompt,
                    "input_image": image_input,
                    "seed": seed,
                    "output_format": "png",
                },
            )
        except Exception as e:
            if "sensitive" in str(e).lower() or "E005" in str(e):
                # Retry with safer prompt
                logger.warning(f"Content flagged, retrying with safer prompt...")
                safe_prompt = prompt + ", safe for work, fully clothed, appropriate content"
                image_input = self._prepare_image_input(reference_image)
                output = replicate.run(
                    self.kontext_model,
                    input={
                        "prompt": safe_prompt,
                        "input_image": image_input,
                        "seed": seed + 1,
                        "output_format": "png",
                    },
                )
            else:
                raise

        image_url = self._extract_url(output)
        self._download_image(image_url, output_path)

        # Get dimensions from downloaded image
        from PIL import Image
        img = Image.open(output_path)
        w, h = img.size

        return ImageResult(
            image_path=output_path,
            prompt=prompt,
            seed=seed,
            width=w,
            height=h,
        )

    @staticmethod
    def _prepare_image_input(image_path: str):
        """Prepare image for Replicate API — returns file handle or URI."""
        path = Path(image_path)
        if path.exists():
            return open(image_path, "rb")
        elif image_path.startswith("http"):
            return image_path
        else:
            raise FileNotFoundError(f"Reference image not found: {image_path}")

    @staticmethod
    def _extract_url(output) -> str:
        """Extract image URL from Replicate output (handles various formats)."""
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            item = output[0]
            return str(item)
        # FileOutput object
        return str(output)

    @staticmethod
    def _download_image(url: str, output_path: str) -> None:
        """Download image from URL to local path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)

    @staticmethod
    def _stitch_references(image_paths: list[str], output_path: str) -> str:
        """Stitch multiple reference images side by side for multi-character input."""
        from PIL import Image

        images = [Image.open(p) for p in image_paths]

        # Resize all to same height
        target_h = min(img.height for img in images)
        resized = []
        for img in images:
            ratio = target_h / img.height
            new_w = int(img.width * ratio)
            resized.append(img.resize((new_w, target_h), Image.LANCZOS))

        total_w = sum(img.width for img in resized)
        stitched = Image.new("RGB", (total_w, target_h))

        x = 0
        for img in resized:
            stitched.paste(img, (x, 0))
            x += img.width

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stitched.save(output_path, "PNG")
        return output_path
