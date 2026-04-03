"""
Ken Burns effect: slow pan/zoom on static images to create motion.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image
from moviepy import VideoClip


def ken_burns_clip(
    image_path: str,
    duration: float,
    target_resolution: tuple[int, int] = (1080, 1920),
    effect: str = "random",
    zoom_ratio: float = 0.08,
    fps: int = 30,
) -> VideoClip:
    """
    Create a Ken Burns effect clip from a static image.

    Args:
        image_path: Path to the source image.
        duration: Clip duration in seconds.
        target_resolution: Output (width, height).
        effect: "zoom_in", "zoom_out", "pan_left", "pan_right",
                "pan_up", "pan_down", "random".
        zoom_ratio: How much to zoom (0.08 = 8%).
        fps: Frames per second.
    """
    if effect == "random":
        effect = random.choice([
            "zoom_in", "zoom_out", "pan_left", "pan_right",
            "pan_up", "pan_down",
        ])

    target_w, target_h = target_resolution

    # Load and scale image
    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.size

    scale_w = (target_w * (1 + zoom_ratio * 2)) / img_w
    scale_h = (target_h * (1 + zoom_ratio * 2)) / img_h
    scale = max(scale_w, scale_h)

    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    pil_img = pil_img.resize((scaled_w, scaled_h), Image.LANCZOS)
    base_frame = np.array(pil_img)

    def make_frame(t):
        progress = t / duration if duration > 0 else 0

        if effect == "zoom_in":
            z = 1.0 + zoom_ratio * progress
            crop_w = int(target_w / z)
            crop_h = int(target_h / z)
            x = (scaled_w - crop_w) // 2
            y = (scaled_h - crop_h) // 2
        elif effect == "zoom_out":
            z = 1.0 + zoom_ratio * (1 - progress)
            crop_w = int(target_w / z)
            crop_h = int(target_h / z)
            x = (scaled_w - crop_w) // 2
            y = (scaled_h - crop_h) // 2
        elif effect == "pan_left":
            crop_w, crop_h = target_w, target_h
            x = int((scaled_w - target_w) * (1 - progress))
            y = (scaled_h - target_h) // 2
        elif effect == "pan_right":
            crop_w, crop_h = target_w, target_h
            x = int((scaled_w - target_w) * progress)
            y = (scaled_h - target_h) // 2
        elif effect == "pan_up":
            crop_w, crop_h = target_w, target_h
            x = (scaled_w - target_w) // 2
            y = int((scaled_h - target_h) * (1 - progress))
        elif effect == "pan_down":
            crop_w, crop_h = target_w, target_h
            x = (scaled_w - target_w) // 2
            y = int((scaled_h - target_h) * progress)
        else:
            crop_w, crop_h = target_w, target_h
            x = (scaled_w - target_w) // 2
            y = (scaled_h - target_h) // 2

        x = max(0, min(x, scaled_w - crop_w))
        y = max(0, min(y, scaled_h - crop_h))

        cropped = base_frame[y:y + crop_h, x:x + crop_w]

        if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
            img = Image.fromarray(cropped)
            img = img.resize((target_w, target_h), Image.LANCZOS)
            cropped = np.array(img)

        return cropped

    return VideoClip(make_frame, duration=duration).with_fps(fps)
