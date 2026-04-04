"""
Reference image selector for Flux Kontext Pro.

Picks which master reference image(s) to use for each scene,
enabling character/creature visual consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.image_pipeline.scene_analyzer import SceneAnalysis
from src.image_pipeline.visual_sheet import VisualSheet

logger = logging.getLogger(__name__)


def select_reference(
    analysis: SceneAnalysis,
    visual_sheet: VisualSheet,
) -> str | None:
    """
    Select the best reference image for this scene.

    Priority:
    1. Single character → that character's reference
    2. Multiple characters → first character with a reference (Kontext works best with one)
    3. Creature only → creature reference
    4. No entities → None (text-to-image fallback)

    Args:
        analysis: Scene analysis with characters_present and creatures_present.
        visual_sheet: Visual sheet with reference image paths.

    Returns:
        Path to reference image, or None for text-to-image.
    """
    # Try characters first (prioritize protagonist/main characters)
    for name in analysis.characters_present:
        ref = visual_sheet.get_reference(name)
        if ref:
            return ref

    # Try creatures
    for name in analysis.creatures_present:
        ref = visual_sheet.get_reference(name)
        if ref:
            return ref

    return None


def select_references_multi(
    analysis: SceneAnalysis,
    visual_sheet: VisualSheet,
    max_refs: int = 3,
) -> list[str]:
    """
    Select multiple reference images for multi-character scenes.

    Args:
        analysis: Scene analysis.
        visual_sheet: Visual sheet.
        max_refs: Maximum number of references to return.

    Returns:
        List of reference image paths (may be empty).
    """
    refs = []

    # Characters first
    for name in analysis.characters_present:
        ref = visual_sheet.get_reference(name)
        if ref and ref not in refs:
            refs.append(ref)
        if len(refs) >= max_refs:
            break

    # Then creatures if room
    if len(refs) < max_refs:
        for name in analysis.creatures_present:
            ref = visual_sheet.get_reference(name)
            if ref and ref not in refs:
                refs.append(ref)
            if len(refs) >= max_refs:
                break

    return refs
