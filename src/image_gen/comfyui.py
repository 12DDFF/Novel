"""
ComfyUI client for Flux Dev image generation.

Connects to a ComfyUI server (local or cloud) via its REST API,
submits Flux Dev workflows, and downloads results.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from pathlib import Path

import httpx

from .base import BaseImageGenerator, ImageResult

logger = logging.getLogger(__name__)

# Flux Dev workflow template for ComfyUI
# This is a minimal txt2img workflow using the Flux Dev checkpoint
_FLUX_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "flux1-dev-fp8.safetensors",
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 1024,
            "height": 1792,
            "batch_size": 1,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["4", 1],
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["4", 1],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "novel2video",
            "images": ["8", 0],
        },
    },
}


class ComfyUIGenerator(BaseImageGenerator):
    """
    Image generator using ComfyUI's REST API with Flux Dev.

    Requires a running ComfyUI server (local or remote).
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8188",
        checkpoint: str = "flux1-dev-fp8.safetensors",
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.checkpoint = checkpoint
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = uuid.uuid4().hex[:8]

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

        # Build workflow
        workflow = json.loads(json.dumps(_FLUX_WORKFLOW))
        workflow["3"]["inputs"]["seed"] = seed
        workflow["4"]["inputs"]["ckpt_name"] = self.checkpoint
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["6"]["inputs"]["text"] = prompt

        # Submit
        prompt_id = self._queue_prompt(workflow)

        # Wait for completion
        output_images = self._wait_for_result(prompt_id)

        if not output_images:
            raise RuntimeError(f"No images generated for prompt: {prompt[:50]}...")

        # Download first image
        image_data = self._download_image(output_images[0])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)

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
        prompt = (
            f"character design sheet, front view, {character_description}, "
            f"{style} style, plain white background, full body, clear features, "
            f"high detail, reference sheet"
        )
        return self.generate(prompt, output_path, width=1024, height=1024)

    def _queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        response = httpx.post(
            f"{self.server_url}/prompt",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["prompt_id"]

    def _wait_for_result(self, prompt_id: str) -> list[dict]:
        """Poll until the prompt completes. Returns list of output image info."""
        start = time.monotonic()

        while time.monotonic() - start < self.timeout:
            response = httpx.get(
                f"{self.server_url}/history/{prompt_id}",
                timeout=10.0,
            )
            response.raise_for_status()
            history = response.json()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                images = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        images.extend(node_output["images"])
                return images

            time.sleep(self.poll_interval)

        raise TimeoutError(f"ComfyUI generation timed out after {self.timeout}s")

    def _download_image(self, image_info: dict) -> bytes:
        """Download a generated image from ComfyUI."""
        params = {
            "filename": image_info["filename"],
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        response = httpx.get(
            f"{self.server_url}/view",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.content

    def health_check(self) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            response = httpx.get(f"{self.server_url}/system_stats", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
