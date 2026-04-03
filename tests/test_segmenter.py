"""Tests for the scene segmenter module."""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.llm_client import LLMClient
from src.models import Character, Mood, Scene, TransitionType
from src.segmenter.character_extractor import extract_characters
from src.segmenter.segmenter import SceneSegmenter


# ── Mock LLM responses ──────────────────────────────────────────────────────

MOCK_CHARACTERS_RESPONSE = [
    {
        "name": "王林",
        "aliases": ["小林子"],
        "description": "年轻男子，面容冷峻，穿着灰色长袍",
        "role": "protagonist",
    },
    {
        "name": "司徒南",
        "aliases": [],
        "description": "老者，白发白须，仙风道骨",
        "role": "supporting",
    },
]

MOCK_SCENES_RESPONSE = [
    {
        "sequence": 1,
        "narration_text": "王林站在山顶，俯瞰着整个村庄。寒风呼啸而过，吹动他灰色的长袍。",
        "visual_description": "Young man in grey robes on mountain peak overlooking village",
        "characters_present": ["王林"],
        "mood": "dramatic",
        "setting": "Mountain peak, windy, overcast",
        "image_prompt": "cinematic anime style, wide shot, young Chinese man on mountain peak",
        "transition": "fade_to_black",
    },
    {
        "sequence": 2,
        "narration_text": "他转身看向身后的老者。司徒南微微一笑，点了点头。",
        "visual_description": "Young man turning to face an elderly sage, both on mountain peak",
        "characters_present": ["王林", "司徒南"],
        "mood": "peaceful",
        "setting": "Mountain peak, calmer wind",
        "image_prompt": "cinematic anime style, medium shot, two figures on mountain",
        "transition": "crossfade",
    },
    {
        "sequence": 3,
        "narration_text": "一道剑光划过天际，直奔村庄而去。王林的眼中闪过一丝寒意。",
        "visual_description": "Sword light streaking across sky toward village, man watching with cold eyes",
        "characters_present": ["王林"],
        "mood": "tense",
        "setting": "Mountain peak, sword light in sky",
        "image_prompt": "cinematic anime style, dramatic shot, sword light in sky",
        "transition": "cut",
    },
]


# ── LLM Client Tests ────────────────────────────────────────────────────────


class TestLLMClient:
    def test_extract_json_direct(self):
        raw = '[{"name": "test"}]'
        result = LLMClient._extract_json(raw)
        assert result == [{"name": "test"}]

    def test_extract_json_from_code_block(self):
        raw = '```json\n[{"name": "test"}]\n```'
        result = LLMClient._extract_json(raw)
        assert result == [{"name": "test"}]

    def test_extract_json_with_surrounding_text(self):
        raw = 'Here is the result:\n[{"name": "test"}]\nDone!'
        result = LLMClient._extract_json(raw)
        assert result == [{"name": "test"}]

    def test_extract_json_object(self):
        raw = '{"key": "value"}'
        result = LLMClient._extract_json(raw)
        assert result == {"key": "value"}

    def test_extract_json_invalid(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            LLMClient._extract_json("this is not json at all")


# ── Character Extractor Tests ────────────────────────────────────────────────


class TestCharacterExtractor:
    def test_extract_characters(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.return_value = MOCK_CHARACTERS_RESPONSE

        chars = extract_characters(mock_llm, "fake chapter text")

        assert len(chars) == 2
        assert chars[0].name == "王林"
        assert chars[0].aliases == ["小林子"]
        assert chars[0].role == "protagonist"
        assert chars[1].name == "司徒南"

    def test_extract_characters_empty(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.return_value = []

        chars = extract_characters(mock_llm, "no characters here")
        assert chars == []

    def test_extract_characters_filters_empty_names(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.return_value = [
            {"name": "王林", "aliases": [], "description": "test", "role": "protagonist"},
            {"name": "", "aliases": [], "description": "unnamed", "role": "minor"},
        ]

        chars = extract_characters(mock_llm, "text")
        assert len(chars) == 1
        assert chars[0].name == "王林"

    def test_extract_characters_invalid_response(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.return_value = {"not": "a list"}

        with pytest.raises(ValueError, match="Expected list"):
            extract_characters(mock_llm, "text")


# ── Scene Segmenter Tests ───────────────────────────────────────────────────


class TestSceneSegmenter:
    def _make_segmenter(self, scenes_response=None, chars_response=None):
        mock_llm = MagicMock(spec=LLMClient)
        responses = []
        if chars_response is not None:
            responses.append(chars_response)
        if scenes_response is not None:
            responses.append(scenes_response)
        mock_llm.chat_json.side_effect = responses if responses else [MOCK_SCENES_RESPONSE]
        return SceneSegmenter(mock_llm, art_style="cinematic anime style"), mock_llm

    def test_segment_scenes(self):
        segmenter, _ = self._make_segmenter(MOCK_SCENES_RESPONSE)
        characters = [Character(name="王林"), Character(name="司徒南")]

        scenes = segmenter.segment_scenes("fake chapter text", characters)

        assert len(scenes) == 3
        assert scenes[0].sequence == 1
        assert scenes[0].mood == Mood.DRAMATIC
        assert scenes[0].transition == TransitionType.FADE_TO_BLACK
        assert "王林" in scenes[0].characters_present
        assert scenes[1].mood == Mood.PEACEFUL
        assert scenes[2].mood == Mood.TENSE
        assert scenes[2].transition == TransitionType.CUT

    def test_segment_scenes_duration_estimate(self):
        segmenter, _ = self._make_segmenter(MOCK_SCENES_RESPONSE)
        characters = [Character(name="王林")]

        scenes = segmenter.segment_scenes("fake text", characters)

        # Duration should be based on narration length (~4 chars/sec), min 5s
        for scene in scenes:
            expected = max(5.0, len(scene.narration_text) / 4.0)
            assert scene.duration_estimate_seconds == expected

    def test_process_chapter_full_pipeline(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.side_effect = [
            MOCK_CHARACTERS_RESPONSE,
            MOCK_SCENES_RESPONSE,
        ]

        segmenter = SceneSegmenter(mock_llm, art_style="anime")
        characters, scenes = segmenter.process_chapter("fake chapter text")

        assert len(characters) == 2
        assert len(scenes) == 3
        assert mock_llm.chat_json.call_count == 2

    def test_process_chapter_with_existing_characters(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.side_effect = [
            [{"name": "王林", "aliases": ["小林子", "王前辈"], "description": "更详细的描述更详细的描述", "role": "protagonist"}],
            MOCK_SCENES_RESPONSE,
        ]

        existing = [Character(name="王林", aliases=["小林子"], description="短描述")]
        segmenter = SceneSegmenter(mock_llm)
        characters, scenes = segmenter.process_chapter("text", existing_characters=existing)

        # Should merge: keep existing character but update description and add new alias
        wang_lin = next(c for c in characters if c.name == "王林")
        assert "王前辈" in wang_lin.aliases
        assert "小林子" in wang_lin.aliases
        assert wang_lin.description == "更详细的描述更详细的描述"  # longer = updated

    def test_format_character_profiles(self):
        characters = [
            Character(name="王林", aliases=["小林子"], description="年轻修士", role="protagonist"),
            Character(name="司徒南", aliases=[], description="老者", role="supporting"),
        ]
        result = SceneSegmenter._format_character_profiles(characters)
        assert "王林" in result
        assert "小林子" in result
        assert "protagonist" in result
        assert "司徒南" in result

    def test_format_character_profiles_empty(self):
        result = SceneSegmenter._format_character_profiles([])
        assert "No character" in result

    def test_split_into_chunks_short_text(self):
        text = "这是一段短文本。"
        chunks = SceneSegmenter._split_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_into_chunks_long_text(self):
        # Create text longer than _MAX_CHUNK_SIZE (6000)
        text = "这是一个句子。" * 2000  # ~14000 chars
        chunks = SceneSegmenter._split_into_chunks(text)
        assert len(chunks) > 1
        # Each chunk should be <= 6000 chars (approximately)
        for chunk in chunks:
            assert len(chunk) <= 6500  # some slack for sentence boundary finding

    def test_split_into_chunks_preserves_all_content(self):
        text = "这是一个句子。" * 2000
        chunks = SceneSegmenter._split_into_chunks(text)
        # Due to overlap, concatenation won't equal original, but all content should be present
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined += chunk[500:]  # skip overlap
        # At minimum, no content lost from start and end
        assert combined.startswith(text[:100])
        assert combined.endswith(text[-100:])

    def test_merge_characters_no_overlap(self):
        existing = [Character(name="A", description="desc A")]
        new = [Character(name="B", description="desc B")]
        merged = SceneSegmenter._merge_characters(existing, new)
        assert len(merged) == 2

    def test_merge_characters_with_overlap(self):
        existing = [Character(name="王林", description="短", aliases=["小林"])]
        new = [Character(name="王林", description="更长的描述信息", aliases=["林弟"])]
        merged = SceneSegmenter._merge_characters(existing, new)
        assert len(merged) == 1
        assert merged[0].description == "更长的描述信息"
        assert "小林" in merged[0].aliases
        assert "林弟" in merged[0].aliases

    def test_segment_invalid_response(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat_json.return_value = {"not": "a list"}
        segmenter = SceneSegmenter(mock_llm)
        with pytest.raises(ValueError, match="Expected list"):
            segmenter.segment_scenes("text", [])


# ── Integration Test (requires DeepSeek API key) ────────────────────────────


@pytest.mark.skipif(
    not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")),
    reason="No LLM API key set (DEEPSEEK_API_KEY or OPENROUTER_API_KEY)",
)
class TestSegmenterIntegration:
    """Live tests that hit the real DeepSeek API. Run with: pytest -k Integration"""

    SAMPLE_CHAPTER = """
    王林坐在山洞中，盘膝修炼。洞外风雪交加，寒气透骨。他已经在这里修炼了三天三夜，体内的灵力终于有了突破的迹象。

    突然，一道脚步声从洞外传来。王林猛地睁开双眼，目光如电。

    "谁？"他沉声问道。

    一个白发老者走了进来，脸上带着温和的笑容。他身穿青色长袍，手持一柄拂尘，仙风道骨。

    "小友不必紧张，老夫司徒南，路过此地，见洞中有人修炼，特来一观。"老者微微一笑。

    王林仔细打量着老者，感受到对方身上深不可测的气息，心中一凛。这是一个实力远在自己之上的强者。

    "前辈请坐。"王林站起身来，恭敬地说道。

    司徒南走到洞中的一块石头前坐下，目光在王林身上停留了片刻，眼中闪过一丝赞赏。

    "你的修为虽然不高，但根基扎实，心性坚韧。难得，难得。"司徒南点头说道。
    """

    def _make_llm(self):
        from src.core.config import LLMConfig
        if os.getenv("OPENROUTER_API_KEY"):
            config = LLMConfig(
                provider="openrouter",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model="deepseek/deepseek-chat",
            )
        else:
            config = LLMConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
            )
        return LLMClient(config)

    def test_extract_characters_live(self):
        llm = self._make_llm()
        chars = extract_characters(llm, self.SAMPLE_CHAPTER)

        names = [c.name for c in chars]
        assert any("王林" in n for n in names), f"Expected 王林 in {names}"
        assert any("司徒南" in n for n in names), f"Expected 司徒南 in {names}"

    def test_segment_scenes_live(self):
        llm = self._make_llm()
        segmenter = SceneSegmenter(llm, art_style="cinematic anime style")

        chars = [
            Character(name="王林", description="年轻修士", role="protagonist"),
            Character(name="司徒南", description="白发老者", role="supporting"),
        ]
        scenes = segmenter.segment_scenes(self.SAMPLE_CHAPTER, chars)

        assert len(scenes) >= 2, f"Expected at least 2 scenes, got {len(scenes)}"
        for scene in scenes:
            assert scene.narration_text, "Scene should have narration text"
            assert scene.image_prompt, "Scene should have image prompt"
            assert scene.visual_description, "Scene should have visual description"

    def test_full_pipeline_live(self):
        llm = self._make_llm()
        segmenter = SceneSegmenter(llm, art_style="cinematic anime style")

        characters, scenes = segmenter.process_chapter(self.SAMPLE_CHAPTER)

        assert len(characters) >= 2
        assert len(scenes) >= 2
        # Verify structure (avoid printing Chinese on Windows cp1252 terminal)
        for c in characters:
            assert c.name, "Character must have a name"
        for s in scenes:
            assert s.narration_text, "Scene must have narration text"
            assert s.image_prompt, "Scene must have image prompt"
            assert s.visual_description, "Scene must have visual description"
