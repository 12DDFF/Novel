"""Tests for the sentence splitter."""

import pytest

from src.image_pipeline.sentence_splitter import (
    _normalize_lengths,
    _split_sentences,
    _strip_markers,
    split_narration,
)


# ── Real narration samples ────────────────────────────────────────────────────

SAMPLE_EPISODE = """末世降临的第一天，猩红色的阳光笼罩大地，百分之七十的人类瞬间变成了丧尸。顾杀，一个孤儿高中生，意外重生回到了这一天。前世他错过了机缘，只勉强成为五阶战士，最终战死。这一世，他决心抢占先机。
[画面：教室里，顾杀冷静地指挥同学们堵住门窗，外面传来丧尸的咆哮声]

---SCENE---
就在这时，班花圣母心泛滥，非要开门救外面的同学。顾杀坚决反对，但班花不听，带着舔狗们强行开门。下一秒，门外的女生突然变成丧尸，扑向了他们。

---SCENE---
顾杀当机立断，一把将班花推出去，迅速关门。所有人都惊呆了，没人敢再质疑顾杀的决定。"""


# ── Strip markers ─────────────────────────────────────────────────────────────


class TestStripMarkers:
    def test_removes_visual_notes(self):
        text = "一些文字\n[画面：城市夜景]\n更多文字"
        result = _strip_markers(text)
        assert "[画面" not in result
        assert "一些文字" in result
        assert "更多文字" in result

    def test_removes_scene_markers(self):
        text = "文字1\n---SCENE---\n文字2\n---\n文字3"
        result = _strip_markers(text)
        assert "SCENE" not in result
        assert "---" not in result

    def test_removes_episode_headers(self):
        text = "第1集 (章节 1-50)\n故事开始了。"
        result = _strip_markers(text)
        assert "第1集" not in result
        assert "故事开始了" in result

    def test_removes_markdown(self):
        text = "**设定**\n故事背景。\n### 章节概要\n内容。"
        result = _strip_markers(text)
        assert "**" not in result
        assert "###" not in result

    def test_removes_stage_directions(self):
        text = "（开场镜头）故事开始了。(收尾)结束。"
        result = _strip_markers(text)
        assert "开场镜头" not in result
        assert "故事开始了" in result

    def test_handles_empty_text(self):
        assert _strip_markers("") == ""

    def test_preserves_narration_text(self):
        text = "顾杀站在天台上，看着远方的丧尸潮。他冷冷一笑。"
        result = _strip_markers(text)
        assert result == text


# ── Split sentences ───────────────────────────────────────────────────────────


class TestSplitSentences:
    def test_splits_on_period(self):
        text = "第一句话。第二句话。第三句话。"
        result = _split_sentences(text)
        assert len(result) == 3

    def test_splits_on_exclamation(self):
        text = "快跑！他来了！"
        result = _split_sentences(text)
        assert len(result) == 2

    def test_splits_on_question(self):
        text = "你是谁？为什么在这里？"
        result = _split_sentences(text)
        assert len(result) == 2

    def test_splits_on_newline(self):
        text = "第一段\n第二段"
        result = _split_sentences(text)
        assert len(result) == 2

    def test_keeps_punctuation_with_sentence(self):
        text = "他说了一句话。"
        result = _split_sentences(text)
        assert result[0].endswith("。")

    def test_empty_text(self):
        assert _split_sentences("") == []


# ── Normalize lengths ─────────────────────────────────────────────────────────


class TestNormalizeLengths:
    def test_merges_short_sentences(self):
        sentences = ["他来了。", "啊！", "所有人都惊呆了。"]
        result = _normalize_lengths(sentences, min_chars=5, max_chars=200)
        # "啊！" is too short, merges with previous
        assert len(result) < len(sentences)
        assert any("啊" in s for s in result)

    def test_splits_long_sentences(self):
        long_sent = "这是一个很长的句子，包含了很多信息，讲述了主角的经历，他从小就是孤儿，在末世中独自求生，" * 3
        result = _normalize_lengths([long_sent], min_chars=5, max_chars=80)
        assert len(result) > 1

    def test_keeps_normal_sentences(self):
        sentences = ["这是正常长度的句子。", "另一句正常的话。"]
        result = _normalize_lengths(sentences, min_chars=5, max_chars=200)
        assert len(result) == 2

    def test_empty_list(self):
        assert _normalize_lengths([], min_chars=5, max_chars=200) == []


# ── Full pipeline ─────────────────────────────────────────────────────────────


class TestSplitNarration:
    def test_real_episode(self):
        sentences = split_narration(SAMPLE_EPISODE)
        # Should produce multiple sentences
        assert len(sentences) >= 5
        # No markers in output
        for s in sentences:
            assert "[画面" not in s
            assert "---" not in s
            assert "SCENE" not in s

    def test_all_sentences_nonempty(self):
        sentences = split_narration(SAMPLE_EPISODE)
        for s in sentences:
            assert len(s.strip()) > 0

    def test_no_very_short_sentences(self):
        sentences = split_narration(SAMPLE_EPISODE, min_chars=8)
        for s in sentences:
            assert len(s) >= 8

    def test_preserves_story_content(self):
        sentences = split_narration(SAMPLE_EPISODE)
        combined = "".join(sentences)
        assert "顾杀" in combined
        assert "丧尸" in combined
        assert "班花" in combined

    def test_empty_input(self):
        assert split_narration("") == []

    def test_only_markers(self):
        text = "---SCENE---\n[画面：描述]\n---"
        assert split_narration(text) == []
