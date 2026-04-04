"""Tests for the reference image selector."""

import pytest

from src.image_pipeline.ref_selector import select_reference, select_references_multi
from src.image_pipeline.scene_analyzer import SceneAnalysis
from src.image_pipeline.visual_sheet import VisualEntity, VisualSheet


@pytest.fixture
def visual_sheet(tmp_path):
    # Create fake reference images
    char_ref = tmp_path / "char.png"
    char_ref.write_text("fake")
    creature_ref = tmp_path / "zombie.png"
    creature_ref.write_text("fake")

    return VisualSheet(entities={
        "顾杀": VisualEntity(
            name="顾杀", entity_type="character",
            reference_image_path=str(char_ref),
        ),
        "陈楠楠": VisualEntity(
            name="陈楠楠", entity_type="character",
            reference_image_path=None,  # no reference
        ),
        "丧尸": VisualEntity(
            name="丧尸", entity_type="creature",
            reference_image_path=str(creature_ref),
        ),
    })


class TestSelectReference:
    def test_single_character(self, visual_sheet):
        analysis = SceneAnalysis(characters_present=["顾杀"])
        ref = select_reference(analysis, visual_sheet)
        assert ref is not None
        assert "char.png" in ref

    def test_character_without_reference(self, visual_sheet):
        analysis = SceneAnalysis(characters_present=["陈楠楠"])
        ref = select_reference(analysis, visual_sheet)
        assert ref is None

    def test_creature_only(self, visual_sheet):
        analysis = SceneAnalysis(creatures_present=["丧尸"])
        ref = select_reference(analysis, visual_sheet)
        assert ref is not None
        assert "zombie.png" in ref

    def test_character_prioritized_over_creature(self, visual_sheet):
        analysis = SceneAnalysis(
            characters_present=["顾杀"],
            creatures_present=["丧尸"],
        )
        ref = select_reference(analysis, visual_sheet)
        assert "char.png" in ref  # character wins

    def test_no_entities(self, visual_sheet):
        analysis = SceneAnalysis()
        ref = select_reference(analysis, visual_sheet)
        assert ref is None

    def test_unknown_character(self, visual_sheet):
        analysis = SceneAnalysis(characters_present=["不存在的人"])
        ref = select_reference(analysis, visual_sheet)
        assert ref is None


class TestSelectReferencesMulti:
    def test_multiple_characters(self, visual_sheet):
        analysis = SceneAnalysis(
            characters_present=["顾杀", "陈楠楠"],
            creatures_present=["丧尸"],
        )
        refs = select_references_multi(analysis, visual_sheet, max_refs=3)
        # 顾杀 has ref, 陈楠楠 doesn't, 丧尸 has ref
        assert len(refs) == 2

    def test_max_refs_limit(self, visual_sheet):
        analysis = SceneAnalysis(
            characters_present=["顾杀"],
            creatures_present=["丧尸"],
        )
        refs = select_references_multi(analysis, visual_sheet, max_refs=1)
        assert len(refs) == 1

    def test_empty_scene(self, visual_sheet):
        analysis = SceneAnalysis()
        refs = select_references_multi(analysis, visual_sheet)
        assert refs == []
