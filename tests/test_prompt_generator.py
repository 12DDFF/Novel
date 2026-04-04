"""Tests for the image prompt generator."""

from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.image_pipeline.prompt_generator import ImagePromptGenerator
from src.image_pipeline.scene_analyzer import SceneAnalysis
from src.image_pipeline.visual_sheet import VisualEntity, VisualSheet


@pytest.fixture
def visual_sheet():
    return VisualSheet(entities={
        "顾杀": VisualEntity(
            name="顾杀", archetype="小帅", entity_type="character",
            visual_description_en="young Chinese man with short black hair and cold eyes, athletic build, torn school uniform, fire axe",
            reference_image_path="/fake/ref.png",
        ),
        "丧尸": VisualEntity(
            name="丧尸", entity_type="creature",
            visual_description_en="humanoid zombie with grey skin and red eyes",
        ),
    })


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_json.return_value = [
        {
            "sentence_index": 0,
            "image_prompt": "anime illustration, medium shot, the same character swinging a fire axe in a dark classroom, dramatic side lighting, natural hand proportions, correct human anatomy",
        },
        {
            "sentence_index": 1,
            "image_prompt": "anime illustration, wide shot, zombies rushing through broken door into classroom, horror underlighting, natural hand proportions, correct human anatomy",
        },
    ]
    return llm


class TestImagePromptGenerator:
    def test_generates_prompts(self, mock_llm, visual_sheet):
        gen = ImagePromptGenerator(mock_llm)
        analyses = [
            SceneAnalysis(sentence_index=0, sentence="顾杀挥斧砍向丧尸。",
                          characters_present=["顾杀"], mood="action", camera_suggestion="medium_shot"),
            SceneAnalysis(sentence_index=1, sentence="丧尸从门口涌入。",
                          creatures_present=["丧尸"], mood="horror", camera_suggestion="wide_shot"),
        ]

        prompts = gen.generate_batch(analyses, visual_sheet)

        assert len(prompts) == 2
        assert "anime illustration" in prompts[0]
        assert "anime illustration" in prompts[1]

    def test_fallback_on_failure(self, visual_sheet):
        bad_llm = MagicMock(spec=LLMClient)
        bad_llm.chat_json.side_effect = ValueError("fail")

        gen = ImagePromptGenerator(bad_llm)
        analyses = [
            SceneAnalysis(sentence_index=0, sentence="test", mood="action",
                          camera_suggestion="close_up", key_action="fighting"),
        ]

        prompts = gen.generate_batch(analyses, visual_sheet)

        assert len(prompts) == 1
        assert "anime illustration" in prompts[0]
        assert "close up" in prompts[0]

    def test_empty_input(self, mock_llm, visual_sheet):
        gen = ImagePromptGenerator(mock_llm)
        assert gen.generate_batch([], visual_sheet) == []
