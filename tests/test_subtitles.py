"""Tests for subtitles module."""
from pathlib import Path

import pytest

from src.subtitles.chunker import chunk_text, split_into_subtitle_groups
from src.subtitles.generator import SubtitleGenerator
from src.tts.base import SubtitleCue, TTSResult


class TestChunker:
    def test_short_text(self):
        assert chunk_text("短文本") == ["短文本"]

    def test_break_at_punctuation(self):
        result = chunk_text("这是第一句话，这是第二句话。", max_chars=10)
        assert len(result) >= 2
        assert result[0].endswith("，")

    def test_hard_break(self):
        text = "没有任何标点符号的一段很长的文本内容"
        result = chunk_text(text, max_chars=10)
        assert all(len(c) <= 10 for c in result)

    def test_empty(self):
        assert chunk_text("") == []

    def test_subtitle_groups(self):
        result = split_into_subtitle_groups(
            "第一句。第二句。第三句。第四句。",
            max_chars_per_line=8,
            max_lines=2,
        )
        assert len(result) >= 1
        for group in result:
            assert group.count("\n") <= 1  # max 2 lines


class TestSubtitleGenerator:
    def test_generate_from_text(self, tmp_path):
        gen = SubtitleGenerator()
        output = str(tmp_path / "test.srt")
        gen.generate_from_text_and_duration("测试字幕。第二句。", 5000, output, format="srt")
        content = Path(output).read_text(encoding="utf-8")
        assert "测试" in content
        assert "-->" in content

    def test_generate_ass(self, tmp_path):
        gen = SubtitleGenerator()
        output = str(tmp_path / "test.ass")
        gen.generate_from_text_and_duration("测试字幕内容。", 3000, output, format="ass")
        content = Path(output).read_text(encoding="utf-8")
        assert "[V4+ Styles]" in content

    def test_generate_from_cues(self, tmp_path):
        gen = SubtitleGenerator()
        result = TTSResult(
            audio_path="fake.mp3",
            duration_ms=5000,
            cues=[
                SubtitleCue(text="第一句话。", start_ms=0, end_ms=2000),
                SubtitleCue(text="第二句话。", start_ms=2200, end_ms=4500),
            ],
        )
        output = str(tmp_path / "cues.srt")
        gen.generate_for_scene(result, output, format="srt")
        content = Path(output).read_text(encoding="utf-8")
        assert "第一句" in content
        assert "第二句" in content

    def test_generate_from_multiple_results(self, tmp_path):
        gen = SubtitleGenerator()
        results = [
            TTSResult(duration_ms=3000, cues=[
                SubtitleCue(text="场景一。", start_ms=0, end_ms=2500),
            ]),
            TTSResult(duration_ms=3000, cues=[
                SubtitleCue(text="场景二。", start_ms=0, end_ms=2500),
            ]),
        ]
        output = str(tmp_path / "full.ass")
        gen.generate_from_results(results, output)
        content = Path(output).read_text(encoding="utf-8")
        assert "场景一" in content
        assert "场景二" in content

    def test_empty_text(self, tmp_path):
        gen = SubtitleGenerator()
        output = str(tmp_path / "empty.srt")
        gen.generate_from_text_and_duration("", 1000, output, format="srt")
        assert Path(output).exists()
