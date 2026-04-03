"""Tests for the Story Bible schema and builder."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.narration.bible import (
    BibleBuilder,
    BibleUpdate,
    CharacterBible,
    LooseThread,
    PlotEvent,
    RelationshipEntry,
    StoryBible,
    WorldFact,
)
from src.narration.harvester import CharacterHarvester, HarvestedCharacter


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_bible():
    return StoryBible(novel_id="test_novel")


@pytest.fixture
def populated_bible():
    bible = StoryBible(novel_id="test_novel", last_processed_chapter=5)
    bible.characters = {
        "沈辰": CharacterBible(
            name="沈辰",
            surname="沈",
            aliases=["沈少"],
            role="protagonist",
            description="二十岁，身材瘦高，面如白玉",
            arc_status="重生归来，暗中布局",
            first_appeared=1,
            last_appeared=5,
            tier="active",
            relationships={
                "蓝溪": [
                    RelationshipEntry(chapter=1, state="subordinate", detail="贴身保镖", evidence="蓝溪是沈辰的贴身保镖"),
                ],
                "沈斐": [
                    RelationshipEntry(chapter=1, state="family", detail="父子关系", evidence="沈斐之子"),
                ],
            },
        ),
        "蓝溪": CharacterBible(
            name="蓝溪",
            surname="蓝",
            role="ally",
            description="黑裙少女，淡蓝色眼瞳",
            arc_status="忠心保护少爷",
            first_appeared=1,
            last_appeared=5,
            tier="active",
        ),
        "苏明硕": CharacterBible(
            name="苏明硕",
            surname="苏",
            role="ally",
            description="江城实际掌控者",
            arc_status="与沈家合作",
            first_appeared=2,
            last_appeared=3,
            tier="active",
        ),
        "路人甲": CharacterBible(
            name="路人甲",
            surname="",
            role="minor",
            description="酒店服务员",
            first_appeared=1,
            last_appeared=1,
            tier="active",
        ),
    }
    bible.world = [
        WorldFact(fact="末日即将降临", chapter=1, category="event"),
        WorldFact(fact="沈家是帝国最大的权贵门阀", chapter=1, category="faction"),
    ]
    bible.loose_threads = [
        LooseThread(chapter=1, detail="蓝溪手腕上有奇怪的纹身", characters=["蓝溪"]),
    ]
    bible.timeline = [
        PlotEvent(chapter=1, summary="沈辰在江城酒店俯瞰夜景，感慨末日将至", characters_involved=["沈辰", "蓝溪"]),
    ]
    return bible


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


@pytest.fixture
def sample_harvested():
    return [
        HarvestedCharacter(name="沈辰", surname="沈", given_name="辰", frequency=10, chapter_appearances={1: 10}),
        HarvestedCharacter(name="蓝溪", surname="蓝", given_name="溪", frequency=8, chapter_appearances={1: 8}),
        HarvestedCharacter(name="沈斐", surname="沈", given_name="斐", frequency=5, chapter_appearances={1: 5}),
    ]


# ── Schema Tests ──────────────────────────────────────────────────────────────


class TestStoryBibleSchema:
    """Test Bible Pydantic model behavior."""

    def test_empty_bible(self, empty_bible):
        assert empty_bible.novel_id == "test_novel"
        assert len(empty_bible.characters) == 0
        assert empty_bible.last_processed_chapter == 0

    def test_active_characters(self, populated_bible):
        active = populated_bible.active_characters()
        assert "沈辰" in active
        assert "蓝溪" in active

    def test_dormant_characters(self, populated_bible):
        populated_bible.characters["路人甲"].tier = "dormant"
        dormant = populated_bible.dormant_characters()
        assert "路人甲" in dormant
        assert "沈辰" not in dormant

    def test_promote_character(self, populated_bible):
        populated_bible.characters["路人甲"].tier = "dormant"
        populated_bible.promote_character("路人甲")
        assert populated_bible.characters["路人甲"].tier == "active"

    def test_demote_character(self, populated_bible):
        populated_bible.demote_character("路人甲")
        assert populated_bible.characters["路人甲"].tier == "dormant"

    def test_auto_manage_tiers(self, populated_bible):
        # 路人甲 last appeared ch1, current is ch100 → should become dormant
        populated_bible.auto_manage_tiers(current_chapter=100, dormant_threshold=50)
        assert populated_bible.characters["路人甲"].tier == "dormant"
        # 苏明硕 last appeared ch3, 100-3=97 > 50 → dormant
        assert populated_bible.characters["苏明硕"].tier == "dormant"
        # 沈辰 last appeared ch5, 100-5=95 > 50 → dormant
        assert populated_bible.characters["沈辰"].tier == "dormant"

    def test_current_relationship(self, populated_bible):
        shen = populated_bible.characters["沈辰"]
        assert shen.current_relationship("蓝溪") == "subordinate"
        assert shen.current_relationship("沈斐") == "family"
        assert shen.current_relationship("unknown") is None


class TestBibleSerialization:
    """Test save/load roundtrip."""

    def test_roundtrip(self, populated_bible, tmp_path):
        path = tmp_path / "bible.json"
        populated_bible.save(path)
        loaded = StoryBible.load(path)

        assert loaded.novel_id == populated_bible.novel_id
        assert loaded.last_processed_chapter == populated_bible.last_processed_chapter
        assert "沈辰" in loaded.characters
        assert loaded.characters["沈辰"].role == "protagonist"
        assert len(loaded.world) == 2
        assert len(loaded.loose_threads) == 1
        assert len(loaded.timeline) == 1

    def test_roundtrip_preserves_relationships(self, populated_bible, tmp_path):
        path = tmp_path / "bible.json"
        populated_bible.save(path)
        loaded = StoryBible.load(path)

        shen = loaded.characters["沈辰"]
        assert "蓝溪" in shen.relationships
        assert shen.relationships["蓝溪"][0].state == "subordinate"
        assert shen.relationships["蓝溪"][0].evidence == "蓝溪是沈辰的贴身保镖"

    def test_save_creates_directories(self, populated_bible, tmp_path):
        path = tmp_path / "deep" / "nested" / "bible.json"
        populated_bible.save(path)
        assert path.exists()

    def test_json_is_readable(self, populated_bible, tmp_path):
        path = tmp_path / "bible.json"
        populated_bible.save(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["novel_id"] == "test_novel"
        assert "updated_at" in data


class TestBibleContext:
    """Test context generation for LLM injection."""

    def test_context_includes_active_chars(self, populated_bible):
        ctx = populated_bible.get_context_for_chapter()
        assert "沈辰" in ctx
        assert "protagonist" in ctx

    def test_context_includes_relationships(self, populated_bible):
        ctx = populated_bible.get_context_for_chapter()
        assert "蓝溪" in ctx
        assert "subordinate" in ctx

    def test_context_includes_world(self, populated_bible):
        ctx = populated_bible.get_context_for_chapter()
        assert "末日" in ctx

    def test_context_includes_threads(self, populated_bible):
        ctx = populated_bible.get_context_for_chapter()
        assert "纹身" in ctx

    def test_empty_bible_context(self, empty_bible):
        ctx = empty_bible.get_context_for_chapter()
        assert ctx == ""  # no content to show


# ── Bible Update Schema Tests ─────────────────────────────────────────────────


class TestBibleUpdate:
    """Test the update schema."""

    def test_empty_update(self):
        update = BibleUpdate()
        assert update.new_characters == []
        assert update.relationship_changes == []

    def test_parse_from_dict(self):
        data = {
            "new_characters": [{"name": "张三", "role": "antagonist"}],
            "updated_characters": [],
            "relationship_changes": [
                {"source": "沈辰", "target": "张三", "state": "enemy", "detail": "confrontation"}
            ],
            "new_world_facts": [{"fact": "东都有防御系统", "category": "setting"}],
            "new_loose_threads": [],
            "resolved_threads": [],
            "plot_events": [{"summary": "沈辰与张三对峙", "characters_involved": ["沈辰", "张三"]}],
        }
        update = BibleUpdate.model_validate(data)
        assert len(update.new_characters) == 1
        assert update.new_characters[0]["name"] == "张三"
        assert len(update.relationship_changes) == 1


# ── Bible Builder Tests ───────────────────────────────────────────────────────


MOCK_LLM_UPDATE_RESPONSE = {
    "new_characters": [
        {
            "name": "苏清影",
            "aliases": [],
            "surname": "苏",
            "role": "neutral",
            "description": "苏家大小姐，性格冷傲",
            "arc_status": "初次出场，对沈辰态度冷淡",
        }
    ],
    "updated_characters": [
        {
            "name": "沈辰",
            "arc_status": "抵达江城，开始接触苏家",
        }
    ],
    "relationship_changes": [
        {
            "source": "沈辰",
            "target": "苏清影",
            "state": "neutral",
            "detail": "初次见面",
            "evidence": "苏清影冷冷道：你就是沈辰？",
        }
    ],
    "new_world_facts": [
        {"fact": "苏家掌控江城", "category": "faction"},
    ],
    "new_loose_threads": [],
    "resolved_threads": [],
    "plot_events": [
        {"summary": "沈辰抵达江城与苏家会面", "characters_involved": ["沈辰", "苏清影"]},
    ],
}


class TestBibleBuilder:
    """Test the incremental Bible builder."""

    def test_build_chapter_adds_character(self, populated_bible, mock_llm, sample_harvested):
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        # Add 苏清影 to harvested names so validation passes
        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5, chapter_appearances={6: 5}),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "chapter text...", 6, harvested)

        assert "苏清影" in result.characters
        assert result.characters["苏清影"].role == "neutral"
        assert result.last_processed_chapter == 6

    def test_build_chapter_updates_existing(self, populated_bible, mock_llm, sample_harvested):
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5, chapter_appearances={6: 5}),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "chapter text...", 6, harvested)

        # 沈辰's arc_status should be updated
        assert result.characters["沈辰"].arc_status == "抵达江城，开始接触苏家"

    def test_build_chapter_adds_relationship(self, populated_bible, mock_llm, sample_harvested):
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5, chapter_appearances={6: 5}),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "chapter text...", 6, harvested)

        shen = result.characters["沈辰"]
        assert "苏清影" in shen.relationships
        assert shen.relationships["苏清影"][-1].state == "neutral"

    def test_validates_rejects_hallucinated_names(self, populated_bible, mock_llm, sample_harvested):
        # LLM returns a character not in harvested names
        response = {
            "new_characters": [
                {"name": "幻觉角色", "role": "antagonist", "description": "不存在的人"},
            ],
            "updated_characters": [],
            "relationship_changes": [],
            "new_world_facts": [],
            "new_loose_threads": [],
            "resolved_threads": [],
            "plot_events": [],
        }
        mock_llm.chat_json.return_value = response

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "text...", 6, sample_harvested)

        # Hallucinated character should NOT be in Bible
        assert "幻觉角色" not in result.characters

    def test_adds_world_facts(self, populated_bible, mock_llm, sample_harvested):
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5, chapter_appearances={6: 5}),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "text...", 6, harvested)

        facts = [f.fact for f in result.world]
        assert "苏家掌控江城" in facts

    def test_adds_plot_events(self, populated_bible, mock_llm, sample_harvested):
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5, chapter_appearances={6: 5}),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "text...", 6, harvested)

        assert len(result.timeline) == 2  # original + new
        assert "苏家会面" in result.timeline[-1].summary

    def test_model_override_passed(self, empty_bible, mock_llm):
        mock_llm.chat_json.return_value = {
            "new_characters": [],
            "updated_characters": [],
            "relationship_changes": [],
            "new_world_facts": [],
            "new_loose_threads": [],
            "resolved_threads": [],
            "plot_events": [],
        }

        builder = BibleBuilder(mock_llm, model_override="deepseek-reasoner")
        harvested = [HarvestedCharacter(name="沈辰", surname="沈", given_name="辰", frequency=5)]
        builder.build_chapter(empty_bible, "text", 1, harvested)

        # Verify model override was passed to LLM
        call_kwargs = mock_llm.chat_json.call_args
        assert call_kwargs.kwargs.get("model") == "deepseek-reasoner"

    def test_consolidation(self, populated_bible, mock_llm):
        mock_llm.chat_json.return_value = {
            "corrections": [
                {"character": "路人甲", "role": "minor", "arc_status": "已离场"},
            ],
            "merge_suggestions": [],
        }

        builder = BibleBuilder(mock_llm)
        result = builder.consolidate(populated_bible)

        assert result.characters["路人甲"].arc_status == "已离场"

    def test_resolved_threads(self, populated_bible, mock_llm, sample_harvested):
        response = {
            "new_characters": [],
            "updated_characters": [],
            "relationship_changes": [],
            "new_world_facts": [],
            "new_loose_threads": [],
            "resolved_threads": ["蓝溪手腕上有奇怪的纹身"],
            "plot_events": [],
        }
        mock_llm.chat_json.return_value = response

        builder = BibleBuilder(mock_llm)
        result = builder.build_chapter(populated_bible, "text...", 10, sample_harvested)

        thread = result.loose_threads[0]
        assert thread.status == "resolved"
        assert thread.resolved_chapter == 10

    def test_build_batch(self, populated_bible, mock_llm, sample_harvested):
        """Test batch processing of multiple chapters in one LLM call."""
        mock_llm.chat_json.return_value = MOCK_LLM_UPDATE_RESPONSE

        harvested = sample_harvested + [
            HarvestedCharacter(name="苏清影", surname="苏", given_name="清影", frequency=5,
                               chapter_appearances={6: 2, 7: 3}),
        ]

        chapters = [
            (6, "chapter 6 text..."),
            (7, "chapter 7 text..."),
            (8, "chapter 8 text..."),
        ]

        builder = BibleBuilder(mock_llm)
        result = builder.build_batch(populated_bible, chapters, harvested=harvested)

        # Should have processed all chapters
        assert result.last_processed_chapter == 8
        # Should have added the new character
        assert "苏清影" in result.characters
        # Only 1 LLM call for the whole batch
        assert mock_llm.chat_json.call_count == 1

    def test_build_batch_empty(self, populated_bible, mock_llm):
        """Batch with no chapters returns bible unchanged."""
        builder = BibleBuilder(mock_llm)
        result = builder.build_batch(populated_bible, [])
        assert result.last_processed_chapter == populated_bible.last_processed_chapter
        mock_llm.chat_json.assert_not_called()
