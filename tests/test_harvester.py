"""Tests for the regex character harvester."""

from collections import Counter

import pytest

from src.narration.harvester import (
    DIALOGUE_REGEX,
    NOISE_WORDS,
    CharacterHarvester,
    HarvestedCharacter,
    _split_name,
)


# ── Real novel text samples (from 末世财阀 chapters) ─────────────────────────

CHAPTER_1_SAMPLE = """江城。
水晶之夜国际酒店。
七十八楼。
一位二十岁左右的男子站在巨大的玻璃窗前，静静看着江城繁华的夜景。
男子身材瘦高，面如白玉，眼瞳如黑曜石般耀眼，气质矜贵。
在他身后，一位黑裙少女同样安静地站着。
"少爷，你在看什么？"
少女忍不住发问。
其名沈辰，乃是帝国枢机局大长老沈斐之子。
"蓝溪，你看看这江城的夜景，难道不美吗？"
沈辰淡笑一声。
"美。"
蓝溪答了一句。
沈辰继续欣赏着江城的夜景。
"可惜，不久后，这一切——都要改变了……"
沈辰喃喃一句。
蓝溪看着沈辰。
"少爷，难道说……家族中那些传闻……都是真的吗？"
沈辰看着蓝溪那张足以祸世的少女颜。
"放心吧蓝溪，家族己经准备好了一切。"
沈辰露出一个和煦的微笑。"""

CHAPTER_2_SAMPLE = """沈辰接通了来自沈斐的电话。
"父亲，我己经抵达了江城。"
"呵呵，辰儿，一周前我便己经给苏明硕通过电话。"
沈斐笑道。
沈辰同样淡淡一笑。
"我们给出的条件，苏家绝对不会拒绝。"
听着沈辰自信而从容的声音，沈斐心中更是满意。
苏明硕是江城的实际掌控者，苏家在江城的地位举足轻重。
苏清影冷冷道："你就是沈辰？"
苏梦柠微笑道："姐姐别这样，人家刚来。"
黄悦涵问道："沈少，你真的是从东都来的？"
苏长沙沉声道："既然沈家开出这个条件，我们没理由拒绝。"
苏肖叹了口气："这年头，谁不想抱大腿啊。"
李觅点了点头。"""

CHAPTER_3_DIALOGUE_HEAVY = """沈辰说："我需要你们的配合。"
蓝溪道："少爷，一切听您的。"
苏明硕问："沈少爷，你需要什么？"
苏清影冷笑道："凭什么信你？"
黄邵怒道："你们苏家要和沈家合作？"
苏梦柠叫道："黄叔叔别生气！"
沈斐大声道："辰儿做得好！"
李觅喊道："有情况！"
苏肖嘀咕一声。"""


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def harvester():
    return CharacterHarvester()


# ── Pass 1: Dialogue verb extraction ──────────────────────────────────────────


class TestDialogueVerbExtraction:
    """Test extraction of names before dialogue verbs."""

    def test_basic_dialogue_verbs(self, harvester):
        text = '沈辰说："我知道了。"\n蓝溪道："好的。"'
        names = harvester._extract_dialogue_names(text)
        assert "沈辰" in names
        assert "蓝溪" in names

    def test_compound_dialogue_verbs(self, harvester):
        text = "苏清影冷笑道：你算什么东西\n黄邵怒道：放肆！"
        names = harvester._extract_dialogue_names(text)
        assert "苏清影" in names
        assert "黄邵" in names

    def test_action_verbs(self, harvester):
        text = "李觅点了点头。\n苏肖叹了口气。\n沈辰摇了摇头。"
        names = harvester._extract_dialogue_names(text)
        assert "李觅" in names
        assert "苏肖" in names
        assert "沈辰" in names

    def test_noise_words_filtered(self, harvester):
        text = '少女忍不住发问。\n男子淡笑一声。\n众人说："走吧。"'
        names = harvester._extract_dialogue_names(text)
        assert "少女" not in names
        assert "男子" not in names
        # 众人 is in noise words
        assert "众人" not in names

    def test_false_positive_prefixes_filtered(self, harvester):
        text = '随即说道："快走！"\n紧接着说："来不及了。"'
        names = harvester._extract_dialogue_names(text)
        assert "随即" not in names

    def test_frequency_counting(self, harvester):
        text = '沈辰说："好。"\n沈辰道："嗯。"\n沈辰笑道："哈哈。"'
        names = harvester._extract_dialogue_names(text)
        assert names["沈辰"] == 3

    def test_real_chapter_1(self, harvester):
        names = harvester._extract_dialogue_names(CHAPTER_1_SAMPLE)
        assert "沈辰" in names
        assert "蓝溪" in names

    def test_real_chapter_2(self, harvester):
        names = harvester._extract_dialogue_names(CHAPTER_2_SAMPLE)
        assert "沈斐" in names
        assert "苏清影" in names
        assert "苏梦柠" in names
        assert "黄悦涵" in names
        assert "苏长沙" in names
        assert "苏肖" in names
        assert "李觅" in names


# ── Pass 2: Surname-based narrative extraction ────────────────────────────────


class TestNarrativeExtraction:
    """Test extraction of names in narrative context (not dialogue)."""

    def test_finds_known_names_in_narration(self, harvester):
        known_surnames = {"沈", "蓝"}
        known_names = {"沈辰", "蓝溪"}
        text = "沈辰继续欣赏着江城的夜景。蓝溪看着沈辰。"
        names = harvester._extract_narrative_names(text, known_surnames, known_names)
        assert "沈辰" in names
        assert "蓝溪" in names
        # Should NOT produce garbage like 沈辰继
        assert "沈辰继" not in names

    def test_counts_known_three_char_names(self, harvester):
        known_surnames = {"苏"}
        known_names = {"苏明硕", "苏清影"}
        text = "苏明硕是江城的实际掌控者。苏清影也在场。"
        names = harvester._extract_narrative_names(text, known_surnames, known_names)
        assert "苏明硕" in names
        assert "苏清影" in names

    def test_no_surnames_returns_empty(self, harvester):
        names = harvester._extract_narrative_names("一些文字", set())
        assert len(names) == 0

    def test_doesnt_match_noise(self, harvester):
        known = {"男"}  # not a real surname but testing the filter
        text = "男子站在那里。"
        names = harvester._extract_narrative_names(text, known)
        # "男子" should be filtered as noise
        assert "男子" not in names or names.get("男子", 0) == 0


# ── Pass 3: Title reference extraction ────────────────────────────────────────


class TestTitleExtraction:
    """Test extraction of surname + title suffix patterns."""

    def test_elder_titles(self, harvester):
        text = "沈老爷子坐在主位上。苏夫人端着茶杯。"
        names = harvester._extract_title_references(text)
        assert "沈老爷子" in names
        assert "苏夫人" in names

    def test_young_titles(self, harvester):
        text = "沈少爷请进。苏小姐好久不见。"
        names = harvester._extract_title_references(text)
        assert "沈少爷" in names
        assert "苏小姐" in names

    def test_martial_titles(self, harvester):
        text = "李师兄在前面带路。张长老已经等候多时。"
        names = harvester._extract_title_references(text)
        assert "李师兄" in names
        assert "张长老" in names

    def test_authority_titles(self, harvester):
        text = "王将军带兵前来。陈总裁同意了这笔交易。"
        names = harvester._extract_title_references(text)
        assert "王将军" in names
        assert "陈总裁" in names


# ── Surname clustering ────────────────────────────────────────────────────────


class TestSurnameClustering:
    """Test grouping names by shared surname."""

    def test_basic_clustering(self, harvester):
        names = Counter({"沈辰": 10, "沈斐": 5, "蓝溪": 8})
        clusters = harvester._cluster_by_surname(names)
        assert "沈" in clusters
        assert "沈辰" in clusters["沈"]
        assert "沈斐" in clusters["沈"]
        assert "蓝" in clusters
        assert "蓝溪" in clusters["蓝"]

    def test_title_refs_in_cluster(self, harvester):
        names = Counter({"沈辰": 10, "沈斐": 5, "沈老爷子": 3})
        clusters = harvester._cluster_by_surname(names)
        assert "沈老爷子" in clusters["沈"]

    def test_compound_surname(self, harvester):
        names = Counter({"欧阳锋": 5, "欧阳克": 3})
        clusters = harvester._cluster_by_surname(names)
        assert "欧阳" in clusters
        assert "欧阳锋" in clusters["欧阳"]
        assert "欧阳克" in clusters["欧阳"]

    def test_su_family_cluster(self, harvester):
        names = Counter({
            "苏明硕": 8, "苏清影": 6, "苏梦柠": 5, "苏长沙": 4, "苏肖": 3,
        })
        clusters = harvester._cluster_by_surname(names)
        assert "苏" in clusters
        assert len(clusters["苏"]) == 5


# ── Alias resolution ─────────────────────────────────────────────────────────


class TestAliasResolution:
    """Test mapping aliases to primary character names."""

    def test_title_maps_to_primary(self, harvester):
        names = Counter({"沈辰": 50, "沈斐": 20, "沈老爷子": 5})
        clusters = harvester._cluster_by_surname(names)
        alias_map = harvester._resolve_aliases(clusters, names)
        # 沈老爷子 should map to either 沈辰 or 沈斐 (both are proper names)
        # It maps to whichever proper name is most frequent
        assert "沈老爷子" in alias_map

    def test_title_maps_to_most_frequent(self, harvester):
        names = Counter({"苏清影": 20, "苏明硕": 50, "苏小姐": 3})
        clusters = harvester._cluster_by_surname(names)
        alias_map = harvester._resolve_aliases(clusters, names)
        # 苏小姐 (title ref) maps to most frequent proper name
        assert alias_map.get("苏小姐") == "苏明硕"
        # But 苏清影 stays separate — different person!
        assert "苏清影" not in alias_map

    def test_no_aliases_when_single_name(self, harvester):
        names = Counter({"蓝溪": 10})
        clusters = harvester._cluster_by_surname(names)
        alias_map = harvester._resolve_aliases(clusters, names)
        assert "蓝溪" not in alias_map  # primary names don't appear in alias_map


# ── Full harvest pipeline ─────────────────────────────────────────────────────


class TestHarvestChapter:
    """Test the full single-chapter harvest pipeline."""

    def test_chapter_1(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_1_SAMPLE, chapter_num=1)
        names = {c.name for c in chars}
        assert "沈辰" in names
        assert "蓝溪" in names

    def test_chapter_2_finds_all_characters(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_2_SAMPLE, chapter_num=2)
        names = {c.name for c in chars}
        # Main characters from chapter 2
        assert "沈辰" in names
        assert "沈斐" in names
        assert "苏清影" in names
        assert "苏梦柠" in names
        assert "苏长沙" in names
        assert "苏肖" in names
        assert "李觅" in names

    def test_sorted_by_frequency(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_1_SAMPLE, chapter_num=1)
        frequencies = [c.frequency for c in chars]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_chapter_appearances_set(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_1_SAMPLE, chapter_num=5)
        for char in chars:
            assert 5 in char.chapter_appearances

    def test_surnames_extracted(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_2_SAMPLE, chapter_num=1)
        shen_chen = next((c for c in chars if c.name == "沈辰"), None)
        assert shen_chen is not None
        assert shen_chen.surname == "沈"
        assert shen_chen.given_name == "辰"

    def test_three_char_given_name(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_2_SAMPLE, chapter_num=1)
        su_qingying = next((c for c in chars if c.name == "苏清影"), None)
        assert su_qingying is not None
        assert su_qingying.surname == "苏"
        assert su_qingying.given_name == "清影"

    def test_dialogue_heavy_chapter(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_3_DIALOGUE_HEAVY, chapter_num=3)
        names = {c.name for c in chars}
        assert "沈辰" in names
        assert "蓝溪" in names
        assert "苏清影" in names
        assert "苏梦柠" in names
        assert "黄邵" in names
        assert "李觅" in names
        assert "沈斐" in names

    def test_noise_not_in_results(self, harvester):
        chars = harvester.harvest_chapter(CHAPTER_1_SAMPLE, chapter_num=1)
        names = {c.name for c in chars}
        assert "少女" not in names
        assert "男子" not in names
        assert "少爷" not in names  # standalone 少爷 is not a character


class TestHarvestNovel:
    """Test multi-chapter harvesting."""

    def test_merges_across_chapters(self, harvester):
        chapters = [
            (1, CHAPTER_1_SAMPLE),
            (2, CHAPTER_2_SAMPLE),
            (3, CHAPTER_3_DIALOGUE_HEAVY),
        ]
        chars = harvester.harvest_novel(chapters)
        shen_chen = next((c for c in chars if c.name == "沈辰"), None)
        assert shen_chen is not None
        # Should appear in all 3 chapters
        assert len(shen_chen.chapter_appearances) == 3
        assert 1 in shen_chen.chapter_appearances
        assert 2 in shen_chen.chapter_appearances
        assert 3 in shen_chen.chapter_appearances

    def test_frequency_accumulated(self, harvester):
        chapters = [
            (1, CHAPTER_1_SAMPLE),
            (2, CHAPTER_2_SAMPLE),
        ]
        chars = harvester.harvest_novel(chapters)
        shen_chen = next((c for c in chars if c.name == "沈辰"), None)
        assert shen_chen is not None
        assert shen_chen.frequency > 5  # appears many times across both chapters

    def test_sorted_by_total_frequency(self, harvester):
        chapters = [
            (1, CHAPTER_1_SAMPLE),
            (2, CHAPTER_2_SAMPLE),
            (3, CHAPTER_3_DIALOGUE_HEAVY),
        ]
        chars = harvester.harvest_novel(chapters)
        frequencies = [c.frequency for c in chars]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_all_characters_found(self, harvester):
        chapters = [
            (1, CHAPTER_1_SAMPLE),
            (2, CHAPTER_2_SAMPLE),
            (3, CHAPTER_3_DIALOGUE_HEAVY),
        ]
        chars = harvester.harvest_novel(chapters)
        names = {c.name for c in chars}
        expected = {"沈辰", "蓝溪", "沈斐", "苏清影", "苏梦柠", "苏长沙", "苏肖", "李觅"}
        assert expected.issubset(names)


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_text(self, harvester):
        chars = harvester.harvest_chapter("", chapter_num=1)
        assert chars == []

    def test_text_with_no_dialogue(self, harvester):
        text = "天空很蓝。风吹过树梢。远处传来鸟鸣。"
        chars = harvester.harvest_chapter(text, chapter_num=1)
        assert chars == []

    def test_text_with_no_chinese(self, harvester):
        text = "Hello world. This is English text."
        chars = harvester.harvest_chapter(text, chapter_num=1)
        assert chars == []

    def test_compound_surname_character(self, harvester):
        text = '欧阳锋冷笑道："你们谁敢上来？"\n司徒南说："我来。"'
        chars = harvester.harvest_chapter(text, chapter_num=1)
        names = {c.name for c in chars}
        assert "欧阳锋" in names
        assert "司徒南" in names
        ouyang = next(c for c in chars if c.name == "欧阳锋")
        assert ouyang.surname == "欧阳"
        assert ouyang.given_name == "锋"

    def test_single_chapter_novel(self, harvester):
        chapters = [(1, CHAPTER_1_SAMPLE)]
        chars = harvester.harvest_novel(chapters)
        assert len(chars) > 0

    def test_harvested_character_dataclass(self):
        char = HarvestedCharacter(
            name="沈辰",
            surname="沈",
            given_name="辰",
            aliases=["沈少"],
            frequency=42,
            chapter_appearances={1: 20, 2: 22},
        )
        assert char.name == "沈辰"
        assert char.frequency == 42
        assert len(char.chapter_appearances) == 2


# ── Helper function tests ─────────────────────────────────────────────────────


class TestSplitName:
    """Test the _split_name helper."""

    def test_single_surname(self):
        assert _split_name("沈辰") == ("沈", "辰")

    def test_three_char_name(self):
        assert _split_name("苏清影") == ("苏", "清影")

    def test_compound_surname(self):
        assert _split_name("欧阳锋") == ("欧阳", "锋")

    def test_unknown_surname(self):
        assert _split_name("鑫鑫") == ("", "鑫鑫")

    def test_title_with_surname(self):
        assert _split_name("沈老爷子") == ("沈", "老爷子")


# ── Regex pattern tests ──────────────────────────────────────────────────────


class TestDialogueRegex:
    """Test the compiled dialogue regex directly."""

    def test_simple_verb(self):
        match = DIALOGUE_REGEX.search("沈辰说")
        assert match is not None
        assert match.group(1) == "沈辰"

    def test_compound_verb(self):
        match = DIALOGUE_REGEX.search("苏清影冷笑道")
        assert match is not None
        assert match.group(1) == "苏清影"

    def test_action_phrase(self):
        match = DIALOGUE_REGEX.search("李觅点了点头")
        assert match is not None
        assert match.group(1) == "李觅"

    def test_no_match_on_plain_text(self):
        match = DIALOGUE_REGEX.search("天空很蓝")
        assert match is None
