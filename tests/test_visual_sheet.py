"""Tests for the visual character/creature sheet."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.image_gen.base import BaseImageGenerator, ImageResult
from src.image_pipeline.visual_sheet import (
    VisualEntity,
    VisualSheet,
    VisualSheetBuilder,
)
from src.narration.bible import CharacterBible, StoryBible, WorldFact


@pytest.fixture
def bible():
    b = StoryBible(novel_id="test")
    b.characters = {
        "顾杀": CharacterBible(
            name="顾杀", role="protagonist", tier="active",
            description="孤儿高中生，重生者，性格冷酷果断",
        ),
        "陈楠楠": CharacterBible(
            name="陈楠楠", role="antagonist", tier="active",
            description="班花，圣母心泛滥",
        ),
    }
    b.world = [
        WorldFact(fact="猩红阳光降临后人类变成丧尸", chapter=1, category="event"),
        WorldFact(fact="变异火猫王出现在废弃商场", chapter=10, category="creature"),
    ]
    return b


@pytest.fixture
def archetype_map():
    return {"顾杀": "小帅", "陈楠楠": "白莲花"}


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_json.return_value = {
        "characters": [
            {
                "name": "顾杀",
                "visual_description_en": "18-year-old Chinese male with short black hair and cold eyes, lean athletic build, torn school uniform, carrying fire axe",
            },
            {
                "name": "陈楠楠",
                "visual_description_en": "17-year-old Chinese girl with long straight black hair, innocent round face, neat school uniform, pink hairpin",
            },
        ],
        "creatures": [
            {
                "name": "丧尸",
                "visual_description_en": "humanoid zombie with grey rotting skin, glowing red eyes, torn clothing, hunched posture",
            },
        ],
    }
    return llm


@pytest.fixture
def mock_image_gen():
    gen = MagicMock(spec=BaseImageGenerator)
    gen.generate_character_sheet.return_value = ImageResult(
        image_path="test.png", prompt="test", seed=42, width=1024, height=1024,
    )
    return gen


class TestVisualEntity:
    def test_basic(self):
        entity = VisualEntity(
            name="顾杀", archetype="小帅", entity_type="character",
            visual_description_en="young man with axe",
        )
        assert entity.name == "顾杀"
        assert entity.entity_type == "character"

    def test_default_type(self):
        entity = VisualEntity(name="test")
        assert entity.entity_type == "character"
        assert entity.reference_image_path is None


class TestVisualSheet:
    def test_get_entity(self):
        sheet = VisualSheet(entities={
            "顾杀": VisualEntity(name="顾杀", visual_description_en="desc"),
        })
        assert sheet.get_entity("顾杀") is not None
        assert sheet.get_entity("不存在") is None

    def test_get_visual_description(self):
        sheet = VisualSheet(entities={
            "顾杀": VisualEntity(name="顾杀", visual_description_en="young man"),
        })
        assert sheet.get_visual_description("顾杀") == "young man"
        assert sheet.get_visual_description("不存在") == ""

    def test_all_characters_vs_creatures(self):
        sheet = VisualSheet(entities={
            "顾杀": VisualEntity(name="顾杀", entity_type="character"),
            "丧尸": VisualEntity(name="丧尸", entity_type="creature"),
        })
        assert "顾杀" in sheet.all_characters()
        assert "丧尸" not in sheet.all_characters()
        assert "丧尸" in sheet.all_creatures()

    def test_serialization_roundtrip(self, tmp_path):
        sheet = VisualSheet(
            novel_id="test",
            entities={
                "顾杀": VisualEntity(
                    name="顾杀", archetype="小帅", entity_type="character",
                    visual_description_en="young man with axe",
                ),
            },
        )
        path = tmp_path / "visual_sheet.json"
        sheet.save(path)
        loaded = VisualSheet.load(path)
        assert loaded.novel_id == "test"
        assert "顾杀" in loaded.entities
        assert loaded.entities["顾杀"].visual_description_en == "young man with axe"

    def test_get_reference_nonexistent_file(self):
        sheet = VisualSheet(entities={
            "顾杀": VisualEntity(
                name="顾杀",
                reference_image_path="/nonexistent/path.png",
            ),
        })
        assert sheet.get_reference("顾杀") is None


class TestVisualSheetBuilder:
    def test_build_creates_entities(self, bible, archetype_map, mock_llm, tmp_path):
        builder = VisualSheetBuilder(mock_llm)
        sheet = builder.build(bible, archetype_map, tmp_path, generate_images=False)

        assert "顾杀" in sheet.entities
        assert "陈楠楠" in sheet.entities
        assert "丧尸" in sheet.entities
        assert sheet.entities["顾杀"].entity_type == "character"
        assert sheet.entities["丧尸"].entity_type == "creature"

    def test_build_sets_archetype(self, bible, archetype_map, mock_llm, tmp_path):
        builder = VisualSheetBuilder(mock_llm)
        sheet = builder.build(bible, archetype_map, tmp_path, generate_images=False)

        assert sheet.entities["顾杀"].archetype == "小帅"

    def test_build_sets_visual_description(self, bible, archetype_map, mock_llm, tmp_path):
        builder = VisualSheetBuilder(mock_llm)
        sheet = builder.build(bible, archetype_map, tmp_path, generate_images=False)

        assert "black hair" in sheet.entities["顾杀"].visual_description_en
        assert "zombie" in sheet.entities["丧尸"].visual_description_en

    def test_build_with_image_gen(self, bible, archetype_map, mock_llm, mock_image_gen, tmp_path):
        builder = VisualSheetBuilder(mock_llm, mock_image_gen)
        sheet = builder.build(bible, archetype_map, tmp_path, generate_images=True)

        # Should have called generate_character_sheet for each entity
        assert mock_image_gen.generate_character_sheet.call_count >= 2

    def test_build_handles_llm_failure(self, bible, archetype_map, tmp_path):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.side_effect = ValueError("bad json")

        builder = VisualSheetBuilder(mock_llm)
        sheet = builder.build(bible, archetype_map, tmp_path, generate_images=False)

        # Should not crash, just return empty sheet
        assert len(sheet.entities) == 0
