"""Tests for Narrator v2 with audience state tracking."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.narration.bible import CharacterBible, RelationshipEntry, StoryBible, WorldFact
from src.narration.narrator_v2 import (
    NarrationManifest,
    NarratorV2,
    VideoNarrationRecord,
)
from src.narration.prompts import BRIDGE_FIRST_VIDEO


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bible():
    b = StoryBible(novel_id="test", last_processed_chapter=5)
    b.characters = {
        "沈辰": CharacterBible(
            name="沈辰", surname="沈", role="protagonist",
            description="重生的豪门大少", arc_status="暗中布局",
            first_appeared=1, last_appeared=5, tier="active",
            relationships={
                "蓝溪": [RelationshipEntry(chapter=1, state="love_interest", detail="贴身保镖")],
            },
        ),
        "蓝溪": CharacterBible(
            name="蓝溪", surname="蓝", role="ally",
            description="黑裙少女保镖", arc_status="忠心保护",
            first_appeared=1, last_appeared=5, tier="active",
        ),
    }
    b.world = [WorldFact(fact="末日即将降临", chapter=1, category="event")]
    return b


@pytest.fixture
def archetype_map():
    return {"沈辰": "小帅", "蓝溪": "小美"}


@pytest.fixture
def empty_manifest():
    return NarrationManifest(novel_id="test")


@pytest.fixture
def manifest_with_video():
    m = NarrationManifest(novel_id="test")
    m.add_video(VideoNarrationRecord(
        video_id="v1",
        chapters_covered=[1, 2, 3],
        archetypes_used={"沈辰": "小帅", "蓝溪": "小美"},
        audience_knows=["小帅是重生的", "末日即将降临", "小美是小帅的保镖"],
        audience_does_not_know=["蓝溪的真实身份", "沈斐的暗中计划"],
        cliffhanger="小帅发现有人在暗中跟踪他",
        last_scene_summary="小帅站在仓库里看着物资",
    ))
    return m


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


# ── Narration Manifest Tests ─────────────────────────────────────────────────


class TestNarrationManifest:
    """Test manifest state tracking."""

    def test_empty_manifest(self, empty_manifest):
        assert empty_manifest.get_previous_video() is None
        state = empty_manifest.get_audience_state()
        assert state["audience_knows"] == []
        assert state["audience_does_not_know"] == []

    def test_add_video(self, empty_manifest):
        record = VideoNarrationRecord(
            chapters_covered=[1, 2],
            archetypes_used={"沈辰": "小帅"},
            audience_knows=["fact1"],
            cliffhanger="cliffhanger1",
        )
        empty_manifest.add_video(record)
        assert len(empty_manifest.videos) == 1
        assert empty_manifest.locked_archetypes["沈辰"] == "小帅"

    def test_get_previous_video(self, manifest_with_video):
        prev = manifest_with_video.get_previous_video()
        assert prev is not None
        assert prev.video_id == "v1"
        assert prev.cliffhanger == "小帅发现有人在暗中跟踪他"

    def test_audience_state_aggregation(self, manifest_with_video):
        state = manifest_with_video.get_audience_state()
        assert "小帅是重生的" in state["audience_knows"]
        assert "蓝溪的真实身份" in state["audience_does_not_know"]

    def test_audience_state_deduplication(self):
        m = NarrationManifest(novel_id="test")
        m.add_video(VideoNarrationRecord(
            audience_knows=["fact1"],
            audience_does_not_know=["secret1"],
        ))
        m.add_video(VideoNarrationRecord(
            audience_knows=["secret1", "fact2"],  # secret1 is now revealed
            audience_does_not_know=["secret2"],
        ))
        state = m.get_audience_state()
        # secret1 was revealed in video 2, so it's no longer unknown
        assert "secret1" in state["audience_knows"]
        assert "secret1" not in state["audience_does_not_know"]

    def test_lock_archetype(self, empty_manifest):
        empty_manifest.lock_archetype("沈辰", "小帅")
        assert empty_manifest.locked_archetypes["沈辰"] == "小帅"

    def test_serialization_roundtrip(self, manifest_with_video, tmp_path):
        path = tmp_path / "manifest.json"
        manifest_with_video.save(path)
        loaded = NarrationManifest.load(path)

        assert loaded.novel_id == "test"
        assert len(loaded.videos) == 1
        assert loaded.locked_archetypes["沈辰"] == "小帅"
        assert loaded.videos[0].cliffhanger == "小帅发现有人在暗中跟踪他"


# ── Bridge Instructions Tests ─────────────────────────────────────────────────


class TestBridgeInstructions:
    """Test bridge generation between videos."""

    def test_first_video_bridge(self, empty_manifest, bible, archetype_map):
        narrator = NarratorV2(MagicMock(spec=LLMClient))
        bridge = narrator._build_bridge_instructions(empty_manifest)
        assert bridge == BRIDGE_FIRST_VIDEO
        assert "FIRST" in bridge

    def test_continuation_bridge(self, manifest_with_video, bible, archetype_map):
        narrator = NarratorV2(MagicMock(spec=LLMClient))
        bridge = narrator._build_bridge_instructions(manifest_with_video)
        assert "小帅发现有人在暗中跟踪他" in bridge
        assert "小帅是重生的" in bridge
        assert "蓝溪的真实身份" in bridge

    def test_continuation_bridge_has_cliffhanger(self, manifest_with_video):
        narrator = NarratorV2(MagicMock(spec=LLMClient))
        bridge = narrator._build_bridge_instructions(manifest_with_video)
        assert "跟踪" in bridge


# ── Character Sheet Tests ─────────────────────────────────────────────────────


class TestCharacterSheet:
    """Test character sheet construction for prompts."""

    def test_includes_archetype_and_description(self, bible, archetype_map):
        sheet = NarratorV2._build_character_sheet(archetype_map, bible)
        assert "小帅" in sheet
        assert "沈辰" in sheet
        assert "重生" in sheet
        assert "小美" in sheet
        assert "蓝溪" in sheet

    def test_includes_arc_status(self, bible, archetype_map):
        sheet = NarratorV2._build_character_sheet(archetype_map, bible)
        assert "暗中布局" in sheet

    def test_empty_map(self, bible):
        sheet = NarratorV2._build_character_sheet({}, bible)
        assert sheet == "No characters defined."


# ── Scene Parsing Tests ──────────────────────────────────────────────────────


class TestScriptDedup:
    """Test removal of repeated/looping scenes."""

    def test_removes_exact_duplicate_scenes(self):
        scenes = [
            {"narration": "第一段不同的内容，讲述故事开头。", "visual_note": "城市夜景", "index": 0},
            {"narration": "范闲站在山顶，看着远方的风景，心中感慨万千。", "visual_note": "山顶远眺", "index": 1},
            {"narration": "范闲站在山顶，看着远方的风景，心中感慨万千。", "visual_note": "山顶远眺", "index": 2},
            {"narration": "第三段不同的内容，讲述新的情节。", "visual_note": "战斗场面", "index": 3},
        ]
        result = NarratorV2._dedup_scenes(scenes)
        narrations = [s["narration"] for s in result]
        assert sum(1 for n in narrations if "范闲站在山顶" in n) == 1
        assert any("第一段" in n for n in narrations)
        assert any("第三段" in n for n in narrations)

    def test_removes_near_duplicate_scenes(self):
        scenes = [
            {"narration": "开头内容讲述整个故事的背景和起因。", "visual_note": "", "index": 0},
            {"narration": "范闲站在山庄石坪前端，看着脚下不远处竟然就有云雾轻飘，远处的瘦山青林也是格外清晰，不由发出一声感叹。", "visual_note": "", "index": 1},
            {"narration": "范闲站在山庄石坪前端，看着脚下不远处竟然就有云雾轻飘，远处的瘦山青林也是格外清晰，不由发出一声感叹。林婉儿轻轻靠在他的身边。", "visual_note": "", "index": 2},
            {"narration": "完全不同的结尾内容，讲述了后续的发展和转折。", "visual_note": "", "index": 3},
        ]
        result = NarratorV2._dedup_scenes(scenes)
        assert sum(1 for s in result if "山庄石坪前端" in s["narration"]) == 1

    def test_keeps_unique_scenes(self):
        scenes = [
            {"narration": "第一段独特的内容关于童年", "visual_note": "", "index": 0},
            {"narration": "第二段完全不同的内容关于战斗", "visual_note": "", "index": 1},
            {"narration": "第三段也是不同的讲述结局", "visual_note": "", "index": 2},
        ]
        result = NarratorV2._dedup_scenes(scenes)
        assert len(result) == 3

    def test_handles_short_list(self):
        scenes = [{"narration": "只有一段", "visual_note": "", "index": 0}]
        result = NarratorV2._dedup_scenes(scenes)
        assert len(result) == 1


class TestScriptValidation:
    """Test that original names are replaced in the script."""

    def test_replaces_leaked_names(self):
        script = "沈辰站在那里，蓝溪跟在身后。"
        archetype_map = {"沈辰": "小帅", "蓝溪": "小美"}
        result = NarratorV2._validate_script(script, archetype_map)
        assert "沈辰" not in result
        assert "蓝溪" not in result
        assert "小帅" in result
        assert "小美" in result

    def test_longer_names_replaced_first(self):
        script = "苏梦柠和苏梦都在。"
        archetype_map = {"苏梦柠": "白莲花", "苏梦": "小妹"}
        result = NarratorV2._validate_script(script, archetype_map)
        # 苏梦柠 should be replaced first, not partially matched
        assert "白莲花" in result

    def test_no_changes_when_clean(self):
        script = "小帅站在那里，小美跟在身后。"
        archetype_map = {"沈辰": "小帅", "蓝溪": "小美"}
        result = NarratorV2._validate_script(script, archetype_map)
        assert result == script


class TestSceneParsing:
    """Test parsing of narration scripts into scenes."""

    def test_basic_scene_split(self):
        script = """话说有这么一个小帅，三年前被人陷害。
[画面：一个男子站在高楼窗前，俯瞰城市夜景]
---SCENE---
没想到，就在这时，一个黄毛出现了。
[画面：一个金发青年走进酒吧]
---SCENE---
小帅淡淡一笑，直接把黄毛打趴下了。
[画面：男子一拳击倒对手]"""

        scenes = NarratorV2._parse_scenes(script)
        assert len(scenes) == 3
        assert "小帅" in scenes[0]["narration"]
        assert "高楼" in scenes[0]["visual_note"]
        assert "黄毛" in scenes[1]["narration"]

    def test_visual_notes_extracted(self):
        script = """一些文字
[画面：夕阳下两人对视]
---
另一些文字
[画面：暴雨中的追逐戏]"""

        scenes = NarratorV2._parse_scenes(script)
        assert scenes[0]["visual_note"] == "夕阳下两人对视"
        assert scenes[1]["visual_note"] == "暴雨中的追逐戏"

    def test_empty_scenes_filtered(self):
        # Realistic script with empty gap between two scene markers
        script = "第一段内容\n---SCENE---\n\n\n---SCENE---\n有内容的场景\n[画面：描述]"

        scenes = NarratorV2._parse_scenes(script)
        # Should have 2 scenes: first content + last content, empty middle filtered
        assert len(scenes) >= 2
        # All scenes should have non-empty narration
        for scene in scenes:
            assert scene["narration"].strip()

    def test_visual_notes_removed_from_narration(self):
        script = "小帅站在那里\n[画面：男子站立]"
        scenes = NarratorV2._parse_scenes(script)
        assert "[画面" not in scenes[0]["narration"]


# ── Full Pipeline Tests ──────────────────────────────────────────────────────


class TestNarratorV2Pipeline:
    """Test the full generate_script pipeline."""

    def test_generate_script_returns_structure(self, bible, archetype_map, empty_manifest, mock_llm):
        # Mock the narration generation (each scene must be distinct to survive dedup)
        mock_llm.chat.return_value = """话说有这么一个小帅，他从前世重生到了这个陌生的世界，成为了一个婴儿。
[画面：男子站在高楼窗前俯瞰城市]
---SCENE---
小美是他的贴身保镖，武功高强，对小帅忠心耿耿，誓死保护他的安全。
[画面：黑裙少女手持利刃站在月光下]
---SCENE---
突然一个神秘黑衣人从天而降，直奔小帅而来，小美拔刀迎战，火花四溅。
[画面：夜色中两人激烈交锋的剪影]"""

        # Mock the audience extraction
        mock_llm.chat_json.return_value = {
            "audience_knows": ["小帅重生了", "小美是他的保镖"],
            "audience_does_not_know": ["小帅为什么重生"],
            "cliffhanger": "小帅感觉有人在跟踪",
            "last_scene_summary": "夜景全景",
        }

        narrator = NarratorV2(mock_llm)
        result = narrator.generate_script(
            chapters_text=["chapter 1 text", "chapter 2 text"],
            chapter_numbers=[1, 2],
            bible=bible,
            archetype_map=archetype_map,
            manifest=empty_manifest,
        )

        assert "script" in result
        assert "scenes" in result
        assert "video_record" in result
        assert len(result["scenes"]) == 3
        assert result["video_record"].cliffhanger == "小帅感觉有人在跟踪"
        assert result["video_record"].chapters_covered == [1, 2]

    def test_video_record_tracks_archetypes(self, bible, archetype_map, empty_manifest, mock_llm):
        mock_llm.chat.return_value = "一些文字"
        mock_llm.chat_json.return_value = {
            "audience_knows": [], "audience_does_not_know": [],
            "cliffhanger": "", "last_scene_summary": "",
        }

        narrator = NarratorV2(mock_llm)
        result = narrator.generate_script(
            chapters_text=["text"],
            chapter_numbers=[1],
            bible=bible,
            archetype_map=archetype_map,
            manifest=empty_manifest,
        )

        assert result["video_record"].archetypes_used == archetype_map

    def test_audience_extraction_failure_handled(self, bible, archetype_map, empty_manifest, mock_llm):
        mock_llm.chat.return_value = "一些文字"
        mock_llm.chat_json.side_effect = ValueError("Bad JSON")

        narrator = NarratorV2(mock_llm)
        result = narrator.generate_script(
            chapters_text=["text"],
            chapter_numbers=[1],
            bible=bible,
            archetype_map=archetype_map,
            manifest=empty_manifest,
        )

        # Should not crash, just return empty audience state
        assert result["video_record"].audience_knows == []
        assert result["video_record"].cliffhanger == ""
