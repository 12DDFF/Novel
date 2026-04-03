"""Tests for the archetype taxonomy and assignment system."""

from unittest.mock import MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.narration.archetype import (
    ARCHETYPE_REGISTRY,
    ArchetypeAssigner,
    ArchetypeCategory,
    ArchetypeDefinition,
    _generate_descriptive_name,
    _infer_gender,
    validate_assignment,
)
from src.narration.bible import CharacterBible, RelationshipEntry, StoryBible
from src.narration.harvester import HarvestedCharacter


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bible_with_characters():
    bible = StoryBible(novel_id="test")
    bible.characters = {
        "沈辰": CharacterBible(
            name="沈辰", surname="沈", role="protagonist",
            description="二十岁男子，重生归来",
            arc_status="布局中", first_appeared=1, last_appeared=5, tier="active",
        ),
        "蓝溪": CharacterBible(
            name="蓝溪", surname="蓝", role="ally",
            description="黑裙少女，贴身保镖",
            arc_status="保护少爷", first_appeared=1, last_appeared=5, tier="active",
            relationships={
                "沈辰": [RelationshipEntry(chapter=1, state="love_interest", detail="暗恋沈辰")],
            },
        ),
        "黄邵": CharacterBible(
            name="黄邵", surname="黄", role="antagonist",
            description="黄家家主，四十岁男子",
            first_appeared=2, last_appeared=3, tier="active",
            relationships={
                "沈辰": [RelationshipEntry(chapter=2, state="enemy", detail="与沈家对立")],
            },
        ),
        "沈斐": CharacterBible(
            name="沈斐", surname="沈", role="father",
            description="沈辰的父亲，沈家家主",
            first_appeared=1, last_appeared=5, tier="active",
            relationships={
                "沈辰": [RelationshipEntry(chapter=1, state="family", detail="父子关系")],
            },
        ),
        "苏梦柠": CharacterBible(
            name="苏梦柠", surname="苏", role="supporting",
            description="苏家小女儿，十八岁",
            first_appeared=3, last_appeared=3, tier="active",
        ),
    }
    return bible


@pytest.fixture
def harvested():
    return [
        HarvestedCharacter(name="沈辰", surname="沈", frequency=50),
        HarvestedCharacter(name="蓝溪", surname="蓝", frequency=30),
        HarvestedCharacter(name="黄邵", surname="黄", frequency=15),
        HarvestedCharacter(name="沈斐", surname="沈", frequency=20),
        HarvestedCharacter(name="苏梦柠", surname="苏", frequency=8),
        HarvestedCharacter(name="路人甲", surname="", frequency=1),
    ]


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


# ── Taxonomy Tests ────────────────────────────────────────────────────────────


class TestArchetypeTaxonomy:
    """Test the archetype registry is complete and consistent."""

    def test_registry_not_empty(self):
        assert len(ARCHETYPE_REGISTRY) > 20

    def test_key_archetypes_present(self):
        expected = ["小帅", "小美", "渣男", "渣女", "黄毛", "老爷子",
                     "白莲花", "绿茶", "闺蜜", "兄弟", "小弟",
                     "败家爹", "继父", "岳父", "亲妈", "后妈", "恶婆婆"]
        for name in expected:
            assert name in ARCHETYPE_REGISTRY, f"Missing archetype: {name}"

    def test_all_have_descriptions(self):
        for name, defn in ARCHETYPE_REGISTRY.items():
            assert defn.description, f"{name} has no description"

    def test_all_have_categories(self):
        for name, defn in ARCHETYPE_REGISTRY.items():
            assert isinstance(defn.category, ArchetypeCategory)

    def test_渣男_has_forbidden_roles(self):
        defn = ARCHETYPE_REGISTRY["渣男"]
        assert "father" in defn.forbidden_roles
        assert "stepfather" in defn.forbidden_roles
        assert "elder" in defn.forbidden_roles

    def test_渣女_has_forbidden_roles(self):
        defn = ARCHETYPE_REGISTRY["渣女"]
        assert "mother" in defn.forbidden_roles
        assert "stepmother" in defn.forbidden_roles

    def test_小帅_is_male_protagonist(self):
        defn = ARCHETYPE_REGISTRY["小帅"]
        assert defn.gender == "male"
        assert defn.category == ArchetypeCategory.PROTAGONIST

    def test_小美_is_female_love_interest(self):
        defn = ARCHETYPE_REGISTRY["小美"]
        assert defn.gender == "female"
        assert defn.category == ArchetypeCategory.LOVE_INTEREST

    def test_小弟_is_unique(self):
        defn = ARCHETYPE_REGISTRY["小弟"]
        assert defn.unique is True


# ── Validation Tests ──────────────────────────────────────────────────────────


class TestValidation:
    """Test semantic validation rules."""

    def test_valid_assignment(self, bible_with_characters):
        valid, reason = validate_assignment("沈辰", "小帅", bible_with_characters)
        assert valid

    def test_gender_mismatch_rejected(self, bible_with_characters):
        # 蓝溪 is female, 渣男 requires male
        valid, reason = validate_assignment("蓝溪", "渣男", bible_with_characters)
        assert not valid
        assert "gender" in reason

    def test_forbidden_role_rejected(self, bible_with_characters):
        # 沈斐 is a father, 渣男 forbids father role
        valid, reason = validate_assignment("沈斐", "渣男", bible_with_characters)
        assert not valid
        assert "father" in reason or "cannot" in reason

    def test_unknown_archetype_rejected(self, bible_with_characters):
        valid, reason = validate_assignment("沈辰", "不存在的类型", bible_with_characters)
        assert not valid
        assert "Unknown" in reason

    def test_valid_family_archetype(self, bible_with_characters):
        valid, reason = validate_assignment("沈斐", "老爷子", bible_with_characters)
        assert valid

    def test_female_can_be_小美(self, bible_with_characters):
        valid, reason = validate_assignment("蓝溪", "小美", bible_with_characters)
        assert valid

    def test_male_cannot_be_小美(self, bible_with_characters):
        valid, reason = validate_assignment("沈辰", "小美", bible_with_characters)
        assert not valid

    def test_unknown_character_passes(self, bible_with_characters):
        # Character not in Bible → can't validate, default to valid
        valid, reason = validate_assignment("未知人物", "黄毛", bible_with_characters)
        assert valid


# ── Assigner Tests ────────────────────────────────────────────────────────────


class TestArchetypeAssigner:
    """Test the LLM-based archetype assignment."""

    def test_assign_with_mock_llm(self, bible_with_characters, harvested, mock_llm):
        mock_llm.chat_json.return_value = {
            "assignments": [
                {"original_name": "沈辰", "archetype": "小帅", "reasoning": "male protagonist"},
                {"original_name": "蓝溪", "archetype": "小美", "reasoning": "female love interest"},
                {"original_name": "黄邵", "archetype": "心机男", "reasoning": "male antagonist"},
                {"original_name": "沈斐", "archetype": "老爷子", "reasoning": "father figure"},
                {"original_name": "苏梦柠", "archetype": "千金", "reasoning": "rich girl"},
            ]
        }

        assigner = ArchetypeAssigner(mock_llm)
        result = assigner.assign(bible_with_characters, harvested)

        assert result["沈辰"] == "小帅"
        assert result["蓝溪"] == "小美"
        assert result["黄邵"] == "心机男"
        assert result["沈斐"] == "老爷子"

    def test_locked_assignments_preserved(self, bible_with_characters, harvested, mock_llm):
        mock_llm.chat_json.return_value = {
            "assignments": [
                {"original_name": "黄邵", "archetype": "黄毛", "reasoning": "bully"},
                {"original_name": "沈斐", "archetype": "老爷子", "reasoning": "elder"},
                {"original_name": "苏梦柠", "archetype": "千金", "reasoning": "rich girl"},
            ]
        }

        locked = {"沈辰": "小帅", "蓝溪": "小美"}
        assigner = ArchetypeAssigner(mock_llm)
        result = assigner.assign(bible_with_characters, harvested, locked=locked)

        # Locked assignments stay
        assert result["沈辰"] == "小帅"
        assert result["蓝溪"] == "小美"
        # New assignments added
        assert result["黄邵"] == "黄毛"

    def test_low_frequency_gets_descriptive(self, bible_with_characters, harvested, mock_llm):
        mock_llm.chat_json.return_value = {"assignments": []}

        assigner = ArchetypeAssigner(mock_llm)
        # 路人甲 has frequency=1, below min_frequency=3
        result = assigner.assign(bible_with_characters, harvested, min_frequency=3)

        # 路人甲 is not in bible_with_characters, so it won't appear
        # But characters with low frequency get descriptive roles

    def test_invalid_assignment_rejected_and_fallback(self, bible_with_characters, harvested, mock_llm):
        # LLM tries to assign 渣男 to 沈斐 (father figure) — should be rejected
        mock_llm.chat_json.return_value = {
            "assignments": [
                {"original_name": "沈辰", "archetype": "小帅", "reasoning": "protagonist"},
                {"original_name": "蓝溪", "archetype": "小美", "reasoning": "love interest"},
                {"original_name": "黄邵", "archetype": "黄毛", "reasoning": "antagonist"},
                {"original_name": "沈斐", "archetype": "渣男", "reasoning": "bad guy"},
                {"original_name": "苏梦柠", "archetype": "千金", "reasoning": "rich girl"},
            ]
        }

        assigner = ArchetypeAssigner(mock_llm)
        result = assigner.assign(bible_with_characters, harvested)

        # 沈斐 should NOT be 渣男 (forbidden: father role)
        assert result.get("沈斐") != "渣男"

    def test_model_override_passed(self, bible_with_characters, harvested, mock_llm):
        mock_llm.chat_json.return_value = {"assignments": []}

        assigner = ArchetypeAssigner(mock_llm, model_override="deepseek-reasoner")
        assigner.assign(bible_with_characters, harvested)

        call_kwargs = mock_llm.chat_json.call_args
        assert call_kwargs.kwargs.get("model") == "deepseek-reasoner"

    def test_archetype_menu_marks_taken(self, bible_with_characters, harvested, mock_llm):
        mock_llm.chat_json.return_value = {"assignments": []}

        locked = {"沈辰": "小帅"}
        assigner = ArchetypeAssigner(mock_llm)
        assigner.assign(bible_with_characters, harvested, locked=locked)

        # Check that the prompt mentions 小帅 is taken
        call_args = mock_llm.chat_json.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "[TAKEN]" in prompt


# ── Gender Inference Tests ────────────────────────────────────────────────────


class TestGenderInference:
    """Test improved gender detection."""

    def test_feminine_description_detected(self):
        char = CharacterBible(
            name="苏梦柠", role="minor",
            description="容貌纯净无暇，脸蛋略带幼态，能让任何男子升起无尽的保护欲",
        )
        assert _infer_gender(char) == "female"

    def test_masculine_description_detected(self):
        char = CharacterBible(
            name="沈辰", role="protagonist",
            description="二十岁男子，身材瘦高，面如白玉，气质矜贵",
        )
        assert _infer_gender(char) == "male"

    def test_dress_indicates_female(self):
        char = CharacterBible(
            name="蓝溪", role="ally",
            description="黑裙少女，淡蓝色眼瞳，皮肤细若白瓷",
        )
        assert _infer_gender(char) == "female"

    def test_elder_male(self):
        char = CharacterBible(
            name="沈斐", role="father",
            description="五十岁身材魁梧，面容威严",
        )
        assert _infer_gender(char) == "male"

    def test_no_signals_returns_none(self):
        char = CharacterBible(name="路人", role="minor", description="站在旁边")
        assert _infer_gender(char) is None

    def test_name_chars_help(self):
        char = CharacterBible(name="林婉儿", description="")
        assert _infer_gender(char) == "female"


# ── Descriptive Name Tests ────────────────────────────────────────────────────


class TestDescriptiveName:
    """Test Chinese descriptive nickname generation."""

    def test_female_with_surname(self):
        char = CharacterBible(
            name="苏梦柠", surname="苏",
            description="容貌纯净", role="minor",
        )
        name = _generate_descriptive_name(char)
        assert "苏" in name
        assert "姑娘" in name or "姐" in name or "小姐" in name

    def test_male_with_surname(self):
        char = CharacterBible(
            name="李觅", surname="李",
            description="三十岁身材健壮", role="minor",
        )
        name = _generate_descriptive_name(char)
        assert "李" in name

    def test_no_surname_female(self):
        char = CharacterBible(
            name="小红", description="少女", role="minor",
        )
        name = _generate_descriptive_name(char)
        assert name != "路人"
        assert "那个" in name or "姑娘" in name

    def test_never_returns_english(self):
        char = CharacterBible(
            name="test", role="ally", description="some person",
        )
        name = _generate_descriptive_name(char)
        # Should be Chinese, no English role names
        assert "ally" not in name
        assert "minor" not in name
