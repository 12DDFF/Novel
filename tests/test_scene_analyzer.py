"""Tests for the scene context analyzer."""

from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.narration.bible import CharacterBible, StoryBible, WorldFact
from src.image_pipeline.scene_analyzer import SceneAnalysis, SceneAnalyzer


@pytest.fixture
def bible():
    b = StoryBible(novel_id="test")
    b.characters = {
        "顾杀": CharacterBible(name="顾杀", role="protagonist", tier="active", description="重生高中生"),
        "陈楠楠": CharacterBible(name="陈楠楠", role="antagonist", tier="active", description="班花圣母"),
    }
    b.world = [WorldFact(fact="猩红阳光降临后人类变成丧尸", chapter=1, category="event")]
    return b


@pytest.fixture
def archetype_map():
    return {"顾杀": "小帅", "陈楠楠": "白莲花"}


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_json.return_value = [
        {
            "sentence_index": 0,
            "characters_present": ["顾杀"],
            "location": "教室",
            "location_changed": False,
            "mood": "tense",
            "camera_suggestion": "medium_shot",
            "creatures_present": [],
            "key_action": "顾杀命令堵门",
            "background_description": "昏暗教室",
        },
        {
            "sentence_index": 1,
            "characters_present": ["顾杀", "陈楠楠"],
            "location": "教室",
            "location_changed": False,
            "mood": "action",
            "camera_suggestion": "close_up",
            "creatures_present": ["丧尸"],
            "key_action": "陈楠楠开门放丧尸进来",
            "background_description": "昏暗教室",
        },
    ]
    return llm


class TestSceneAnalysis:
    def test_default_values(self):
        a = SceneAnalysis()
        assert a.mood == "dramatic"
        assert a.camera_suggestion == "medium_shot"
        assert a.characters_present == []

    def test_from_dict(self):
        a = SceneAnalysis.model_validate({
            "sentence_index": 0,
            "sentence": "test",
            "characters_present": ["顾杀"],
            "mood": "action",
        })
        assert a.characters_present == ["顾杀"]
        assert a.mood == "action"


class TestSceneAnalyzer:
    def test_analyze_all_basic(self, mock_llm, bible, archetype_map):
        analyzer = SceneAnalyzer(mock_llm, bible, archetype_map)
        sentences = ["顾杀命令大家堵住门窗。", "陈楠楠非要开门，丧尸冲了进来。"]

        results = analyzer.analyze_all(sentences)

        assert len(results) == 2
        assert results[0].characters_present == ["顾杀"]
        assert results[1].characters_present == ["顾杀", "陈楠楠"]
        assert "丧尸" in results[1].creatures_present

    def test_fills_archetype_names(self, mock_llm, bible, archetype_map):
        analyzer = SceneAnalyzer(mock_llm, bible, archetype_map)
        results = analyzer.analyze_all(["顾杀站在那里。", "陈楠楠哭了。"])

        assert results[0].characters_archetype == ["小帅"]
        assert "白莲花" in results[1].characters_archetype

    def test_context_carryover(self, mock_llm, bible, archetype_map):
        analyzer = SceneAnalyzer(mock_llm, bible, archetype_map)
        prev = ["上一集的最后一句话。", "上一集倒数第二句。"]

        results = analyzer.analyze_all(
            ["新一集开始。", "继续讲故事。"],
            previous_sentences=prev,
        )

        # Should have been called with backward context from previous episode
        call_args = mock_llm.chat_json.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "上一集" in prompt

    def test_handles_llm_failure(self, bible, archetype_map):
        bad_llm = MagicMock(spec=LLMClient)
        bad_llm.chat_json.side_effect = ValueError("bad json")

        analyzer = SceneAnalyzer(bad_llm, bible, archetype_map)
        results = analyzer.analyze_all(["一句话。", "另一句。"])

        # Should return fallback analyses, not crash
        assert len(results) == 2
        assert results[0].mood == "dramatic"  # default

    def test_backward_window_from_current(self, mock_llm, bible, archetype_map):
        analyzer = SceneAnalyzer(mock_llm, bible, archetype_map)
        sentences = ["s1。", "s2。", "s3。", "s4。", "s5。", "s6。", "s7。", "s8。", "s9。", "s10。"]

        # With batch_size=5, second batch should have backward from first batch
        analyzer.analyze_all(sentences, batch_size=5)
        assert mock_llm.chat_json.call_count == 2
