"""Tests for the TTS module."""
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tts.base import SubtitleCue, TTSResult
from src.tts.edge_tts_backend import EdgeTTSNarrator, VOICES, _split_sentences


# ── Sentence Splitting Tests ─────────────────────────────────────────────────


class TestSplitSentences:
    def test_basic_chinese(self):
        text = "王林走进了山洞。他看到了一个老人。"
        result = _split_sentences(text)
        assert len(result) == 2
        assert result[0] == "王林走进了山洞。"
        assert result[1] == "他看到了一个老人。"

    def test_mixed_punctuation(self):
        text = "真的吗？当然！那就好。"
        result = _split_sentences(text)
        assert len(result) == 3

    def test_semicolons(self):
        text = "白天修炼；夜晚休息。"
        result = _split_sentences(text)
        assert len(result) == 2

    def test_no_punctuation(self):
        text = "没有标点符号的一段话"
        result = _split_sentences(text)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_text(self):
        assert _split_sentences("") == []

    def test_newlines(self):
        text = "第一行\n第二行\n第三行"
        result = _split_sentences(text)
        assert len(result) == 3

    def test_consecutive_punctuation(self):
        text = "什么？！好吧。"
        result = _split_sentences(text)
        # Should handle gracefully (may produce empty segments that get filtered)
        assert all(s.strip() for s in result)


# ── EdgeTTSNarrator Unit Tests ───────────────────────────────────────────────


class TestEdgeTTSNarrator:
    def test_voice_shortname(self):
        narrator = EdgeTTSNarrator(voice="yunxi")
        assert narrator.voice == "zh-CN-YunxiNeural"

    def test_voice_full_name(self):
        narrator = EdgeTTSNarrator(voice="zh-CN-XiaoxiaoNeural")
        assert narrator.voice == "zh-CN-XiaoxiaoNeural"

    def test_voices_dict(self):
        assert "yunxi" in VOICES
        assert "xiaoxiao" in VOICES
        assert all(v.endswith("Neural") for v in VOICES.values())

    def test_build_cues_from_boundaries(self):
        narrator = EdgeTTSNarrator()
        boundaries = [
            {"offset_ms": 100, "duration_ms": 2000, "text": "第一句话。"},
            {"offset_ms": 2200, "duration_ms": 1800, "text": "第二句话。"},
        ]
        cues = narrator._build_cues("第一句话。第二句话。", boundaries)
        assert len(cues) == 2
        assert cues[0].start_ms == 100
        assert cues[0].end_ms == 2100
        assert cues[0].text == "第一句话。"
        assert cues[1].start_ms == 2200

    def test_build_cues_fallback(self):
        narrator = EdgeTTSNarrator()
        cues = narrator._build_cues("第一句。第二句。", [])
        assert len(cues) == 2
        assert cues[0].start_ms == 0
        assert cues[0].end_ms > 0
        assert cues[1].start_ms == cues[0].end_ms

    def test_build_cues_empty_text(self):
        narrator = EdgeTTSNarrator()
        cues = narrator._build_cues("", [])
        assert cues == []


# ── Audio Processing Tests ───────────────────────────────────────────────────


class TestAudioProcessing:
    def _make_audio(self, duration_ms=1000):
        """Create a simple test audio segment."""
        from pydub import AudioSegment
        from pydub.generators import Sine
        return Sine(440).to_audio_segment(duration=duration_ms).apply_gain(-20)

    def test_normalize_volume(self):
        from src.tts.audio_processing import normalize_volume
        audio = self._make_audio()
        normalized = normalize_volume(audio, target_dbfs=-16.0)
        assert abs(normalized.dBFS - (-16.0)) < 0.5

    def test_add_silence_padding(self):
        from src.tts.audio_processing import add_silence_padding
        audio = self._make_audio(1000)
        padded = add_silence_padding(audio, start_ms=200, end_ms=300)
        assert len(padded) == 1000 + 200 + 300

    def test_process_scene_audio(self, tmp_path):
        from src.tts.audio_processing import process_scene_audio
        audio = self._make_audio(1000)
        input_path = str(tmp_path / "input.mp3")
        output_path = str(tmp_path / "output.mp3")
        audio.export(input_path, format="mp3")

        result = process_scene_audio(input_path, output_path, pad_start_ms=200, pad_end_ms=300)
        assert Path(result).exists()

        from pydub import AudioSegment
        processed = AudioSegment.from_file(result)
        # Should be longer due to padding
        assert len(processed) > 1000

    def test_concatenate_audio(self, tmp_path):
        from src.tts.audio_processing import concatenate_audio
        audio1 = self._make_audio(500)
        audio2 = self._make_audio(500)
        p1 = str(tmp_path / "a1.mp3")
        p2 = str(tmp_path / "a2.mp3")
        audio1.export(p1, format="mp3")
        audio2.export(p2, format="mp3")

        output = str(tmp_path / "combined.mp3")
        result = concatenate_audio([p1, p2], output)
        assert Path(result).exists()

        from pydub import AudioSegment
        combined = AudioSegment.from_file(result)
        # Should be approximately 1000ms (some variance from mp3 encoding)
        assert 900 < len(combined) < 1200

    def test_concatenate_empty_raises(self):
        from src.tts.audio_processing import concatenate_audio
        with pytest.raises(ValueError, match="No audio"):
            concatenate_audio([], "output.mp3")

    def test_get_duration_ms(self, tmp_path):
        from src.tts.audio_processing import get_duration_ms
        audio = self._make_audio(2000)
        path = str(tmp_path / "test.mp3")
        audio.export(path, format="mp3")
        duration = get_duration_ms(path)
        assert 1900 < duration < 2200  # some mp3 encoding variance


# ── Integration Test (requires internet) ─────────────────────────────────────


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "0") == "1",
    reason="Network tests disabled",
)
class TestTTSIntegration:
    """Live tests that call Edge TTS servers."""

    def test_synthesize_chinese(self, tmp_path):
        narrator = EdgeTTSNarrator(voice="yunxi")
        output = str(tmp_path / "test.mp3")
        result = narrator.synthesize("王林走进了山洞，他看到了一个老人。", output)

        assert Path(result.audio_path).exists()
        assert result.duration_ms > 0
        assert Path(result.audio_path).stat().st_size > 100

    def test_synthesize_long_text(self, tmp_path):
        narrator = EdgeTTSNarrator(voice="yunxi")
        text = "王林站在山顶，俯瞰着整个村庄。寒风呼啸而过，吹动他灰色的长袍。他的目光深邃而坚定，仿佛能穿透这漫天的风雪。在他身后，是一条蜿蜒曲折的山路。"
        output = str(tmp_path / "long.mp3")
        result = narrator.synthesize(text, output)

        assert result.duration_ms > 2000  # should be several seconds
        assert len(result.cues) >= 1

    def test_synthesize_multiple_scenes(self, tmp_path):
        narrator = EdgeTTSNarrator(voice="yunxi")
        scenes = [
            {"id": "scene_001", "narration_text": "他走进了山洞。"},
            {"id": "scene_002", "narration_text": "洞内一片漆黑。"},
            {"id": "scene_003", "narration_text": "远处传来一声低吼。"},
        ]
        output_dir = str(tmp_path / "audio")
        results = narrator.synthesize_scenes(scenes, output_dir)

        assert len(results) == 3
        for r in results:
            assert Path(r.audio_path).exists()
            assert r.duration_ms > 0

    def test_synthesize_with_rate(self, tmp_path):
        narrator_fast = EdgeTTSNarrator(voice="yunxi", rate="+20%")
        narrator_normal = EdgeTTSNarrator(voice="yunxi", rate="+0%")
        text = "这是一段测试文本，用来测试语速调节功能。"

        fast = narrator_fast.synthesize(text, str(tmp_path / "fast.mp3"))
        normal = narrator_normal.synthesize(text, str(tmp_path / "normal.mp3"))

        # Fast should be shorter than normal
        assert fast.duration_ms < normal.duration_ms
