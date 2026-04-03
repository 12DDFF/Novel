"""
Regex-based character name harvester for Chinese web novels.

Extracts character names deterministically from raw novel text using:
1. Dialogue verb patterns (highest confidence)
2. Surname + structure matching (medium confidence)
3. Alias resolution (rule-based clustering)

No LLM needed. Can scan 1000 chapters in seconds.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class HarvestedCharacter:
    """A character discovered by regex analysis."""

    name: str  # primary name e.g. "沈辰"
    surname: str = ""  # e.g. "沈"
    given_name: str = ""  # e.g. "辰"
    aliases: list[str] = field(default_factory=list)
    frequency: int = 0  # total mentions across all chapters
    chapter_appearances: dict[int, int] = field(default_factory=dict)  # {ch_num: count}


# ── Dialogue verbs: words that follow a speaker's name in Chinese novels ──────

# Ordered longest-first so regex alternation matches greedily
DIALOGUE_VERBS = [
    # 3+ char verb phrases (dialogue/reaction markers only)
    "叹了口气", "点了点头", "摇了摇头", "答了一句", "嘀咕一声",
    "低喃一声", "喃喃一句", "冷哼一声", "忍不住发问",
    # 2-char compound verbs
    "冷笑道", "沉声道", "低声道", "大声道", "怒声道", "厉声道",
    "淡淡道", "冷冷道", "缓缓道", "轻声道", "高声道", "急声道",
    "微笑道", "苦笑道", "皱眉道", "嗤笑道", "冷哼道", "嘲讽道",
    "开口道", "接着道", "继续道", "补充道", "插嘴道", "反驳道",
    # Simple verbs
    "答道", "回道", "怒道", "惊道", "喝道", "叹道", "哼道", "笑道",
    "说道", "问道", "喊道", "叫道", "吼道", "骂道",
    "说", "道", "问", "喊", "叫", "吼", "骂",
]

# Build regex patterns: split into compound verbs (safe) and single-char verbs (need validation)
SINGLE_CHAR_VERBS = {"说", "道", "问", "喊", "叫", "吼", "骂"}
COMPOUND_VERBS = [v for v in DIALOGUE_VERBS if v not in SINGLE_CHAR_VERBS]

_COMPOUND_VERB_PATTERN = "|".join(re.escape(v) for v in COMPOUND_VERBS)
_SINGLE_VERB_PATTERN = "|".join(re.escape(v) for v in SINGLE_CHAR_VERBS)

# Compound verbs are specific enough to accept any 2-4 char name before them
COMPOUND_VERB_REGEX = re.compile(
    rf"([\u4e00-\u9fff]{{2,4}}?)({_COMPOUND_VERB_PATTERN})"
)
# Single-char verbs need stricter validation (checked in _extract_dialogue_names)
SINGLE_VERB_REGEX = re.compile(
    rf"([\u4e00-\u9fff]{{2,4}}?)({_SINGLE_VERB_PATTERN})"
)

# Combined for backwards compatibility in tests
_VERB_PATTERN = "|".join(re.escape(v) for v in DIALOGUE_VERBS)
DIALOGUE_REGEX = re.compile(
    rf"([\u4e00-\u9fff]{{2,4}}?)({_VERB_PATTERN})"
)

# ── Common Chinese surnames ───────────────────────────────────────────────────

# Compound surnames (2 chars) — must check before single-char surnames
COMPOUND_SURNAMES = {
    "欧阳", "司马", "上官", "诸葛", "司徒", "令狐", "皇甫", "尉迟",
    "公孙", "慕容", "端木", "宇文", "长孙", "独孤", "南宫", "东方",
    "西门", "百里", "轩辕", "夏侯",
}

# Top ~100 single-char surnames (covers 99%+ of Chinese names)
SINGLE_SURNAMES = {
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈",
    "褚", "卫", "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许",
    "何", "吕", "施", "张", "孔", "曹", "严", "华", "金", "魏",
    "陶", "姜", "戚", "谢", "邹", "喻", "柏", "窦", "章", "云",
    "苏", "潘", "葛", "奚", "范", "彭", "郎", "鲁", "韦", "昌",
    "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳", "唐",
    "罗", "薛", "雷", "贺", "倪", "汤", "滕", "殷", "罗", "毕",
    "郝", "邬", "安", "常", "乐", "于", "时", "傅", "皮", "齐",
    "康", "伍", "余", "元", "卜", "顾", "孟", "平", "黄", "和",
    "穆", "萧", "尹", "姚", "邵", "湛", "汪", "祁", "毛", "禹",
    "狄", "米", "贝", "明", "臧", "计", "伏", "成", "戴", "谈",
    "宋", "茅", "庞", "熊", "纪", "舒", "屈", "项", "祝", "董",
    "梁", "杜", "阮", "蓝", "闵", "席", "季", "麻", "强", "贾",
    "路", "娄", "危", "江", "童", "颜", "郭", "梅", "盛", "林",
    "刁", "钟", "丘", "骆", "高", "夏", "蔡", "田", "樊", "胡",
    "凌", "霍", "虞", "万", "支", "柯", "管", "卢", "莫", "经",
    "房", "裘", "缪", "干", "解", "应", "宗", "丁", "宣", "贲",
    "邓", "郁", "单", "杭", "洪", "包", "诸", "左", "石", "崔",
    "吉", "钮", "龚", "程", "嵇", "邢", "裴", "陆", "荣", "翁",
    "荀", "羊", "甄", "白", "叶", "温", "秋", "厉", "阎", "连",
}

ALL_SURNAMES = COMPOUND_SURNAMES | SINGLE_SURNAMES

# ── Title suffixes that follow surnames to form character references ───────────

TITLE_SUFFIXES = [
    # Elder / family
    "老爷子", "老太太", "老爷", "夫人", "太太", "老太",
    "少爷", "小姐", "姑娘", "公子",
    # Kinship
    "爷爷", "奶奶", "姥姥", "姥爷",
    "叔叔", "婶婶", "伯伯", "伯母",
    "哥哥", "姐姐", "弟弟", "妹妹",
    "大哥", "大姐", "大嫂",
    # Short kinship / honorific
    "哥", "姐", "弟", "妹", "叔", "婶", "伯", "爸", "妈",
    "爷", "嫂", "兄",
    # Sect / martial arts
    "师兄", "师姐", "师弟", "师妹", "师父", "师傅",
    "掌门", "长老", "护法",
    # Rank / authority
    "大人", "将军", "总裁", "董事长", "老板", "主任", "队长",
    "教授", "医生", "老师",
    # Generic
    "先生", "女士",
]

# ── Alias prefixes ────────────────────────────────────────────────────────────

ALIAS_PREFIXES = ["阿", "小", "老"]

# ── Noise words: common descriptors that look like names but aren't ───────────

NOISE_WORDS = {
    # Generic people references
    "少女", "男子", "女子", "老者", "老人", "众人", "对方", "此人",
    "那人", "这人", "旁人", "他人", "本人", "何人", "有人", "无人",
    "二人", "两人", "三人", "几人", "一人", "某人",
    # Descriptive nouns
    "少年", "青年", "男人", "女人", "孩子", "小子", "丫头",
    "世人", "常人", "凡人", "敌人", "主人", "仙人", "高人",
    "夫人", "王爷",
    # Time/manner words starting with surname chars
    "时候", "时间", "时代", "时期", "时刻",
    "明白", "明天", "明显", "明知", "明日", "明明",
    "应该", "应当", "应是",
    "经常", "经过", "经络", "经脉", "经历", "经验",
    "许多", "许久", "许是", "许有",
    "安静", "安全", "安心", "安排", "安慰",
    "温和", "温柔", "温暖",
    "任何", "任由", "任凭",
    "毕竟", "方便", "方向", "方才", "方最",
    "诸位", "诸多",
    # Common objects/nouns that aren't people
    "马车", "心里", "经络", "经脉",
    # Emotions/expressions (verb/adj phrases)
    "笑着", "微笑", "冷笑", "苦笑", "轻声", "低声", "细声", "柔声",
    "淡淡", "冷冷", "缓缓", "轻轻", "嘻嘻", "喃喃", "讷讷",
    "皱眉", "挠头", "苦脸", "冷漠", "冷淡", "好奇", "正色",
    "哆嗦", "抽泣", "自嘲", "开口", "转而", "改而", "继续",
    "仔细", "恭谨",
    # Pronoun fragments (他X, 她X, 只X, 不X)
    "他冷", "她忽", "只听",
    # Preposition/conjunction fragments
    "于他", "于我", "于这", "于露", "于死", "于将", "于修",
    "和图书", "和自己", "和一些", "和谐", "和一股", "和丫环",
    "和这种",
    # Place names
    "江城", "东都", "帝都", "帝国", "京都",
    # X家 patterns (family references)
    "沈家", "苏家", "陈家", "王家", "李家", "张家", "范家",
    "黄家", "周家", "林家", "马家", "赵家", "杨家", "叶家",
    # Common surname + filler patterns
    "难道", "高兴", "高大", "白天", "白色", "金色", "金钱",
    # Fragments that are clearly not names
    "不由大", "恭谨回", "和丫环去", "时间之内",
    "于我们来", "于修行武", "经听路人", "许有些小",
    "安后准备", "经中毒", "经举了", "经处理", "经躲到",
    "经准备", "经顺着", "毕竟京都", "毕竟对方", "毕竟在", "毕竟末",
    "方成大", "程化作一", "莫非这个", "方最开始", "温柔叹息",
    # Website watermarks
    "和图书",  # hetushu.com literally means 和图书
}

# ── Compound verb starters that could be false-positive names ─────────────────

# E.g., "随即说" — 随即 is not a name
FALSE_POSITIVE_PREFIXES = {
    "随即", "紧接", "于是", "然后", "接着", "忽然", "突然",
    "顿时", "立刻", "马上", "终于", "果然", "竟然", "居然",
    "不禁", "不由", "连忙", "赶紧", "急忙", "当即", "随后",
}


class CharacterHarvester:
    """
    Extracts character names from Chinese novel text using regex patterns.

    No LLM calls. Deterministic and fast.
    """

    def __init__(self) -> None:
        # Pre-compile surname+title patterns
        suffix_pattern = "|".join(re.escape(s) for s in TITLE_SUFFIXES)
        # Match compound surnames first, then single
        compound_pattern = "|".join(re.escape(s) for s in COMPOUND_SURNAMES)
        single_pattern = "|".join(re.escape(s) for s in SINGLE_SURNAMES)
        self._title_regex = re.compile(
            rf"({compound_pattern}|{single_pattern})({suffix_pattern})"
        )

    def harvest_chapter(self, text: str, chapter_num: int = 1) -> list[HarvestedCharacter]:
        """Extract characters from a single chapter. Returns sorted by frequency (descending)."""
        # Pass 1: Dialogue verb extraction
        dialogue_names = self._extract_dialogue_names(text)

        # Pass 2: Count known names in narrative + discover names without dialogue
        known_surnames = self._collect_surnames(dialogue_names)
        # Also scan text for all possible surnames (catches characters who never speak)
        text_surnames = self._scan_all_surnames(text)
        all_surnames = known_surnames | text_surnames

        narrative_names = self._extract_narrative_names(
            text, all_surnames, known_names=set(dialogue_names.keys())
        )
        # Pass 2b: Discover names that only appear in narrative (no dialogue)
        discovered = self._discover_narrative_only_names(text, all_surnames, dialogue_names)
        narrative_names.update(discovered)

        # Pass 3: Title-based references
        title_refs = self._extract_title_references(text)

        # Merge all counts
        all_names: Counter[str] = Counter()
        all_names.update(dialogue_names)
        all_names.update(narrative_names)
        all_names.update(title_refs)

        # Filter noise and deduplicate substrings
        all_names = self._filter_noise(all_names)
        all_names = self._deduplicate_substrings(all_names)

        # Cluster by surname and resolve aliases
        clusters = self._cluster_by_surname(all_names)
        alias_map = self._resolve_aliases(clusters, all_names)

        # Build HarvestedCharacter objects
        characters = self._build_characters(all_names, alias_map, chapter_num)

        # Sort by frequency descending
        characters.sort(key=lambda c: c.frequency, reverse=True)
        return characters

    def harvest_novel(
        self, chapters: list[tuple[int, str]]
    ) -> list[HarvestedCharacter]:
        """
        Extract characters across multiple chapters.

        Args:
            chapters: List of (chapter_number, chapter_text) tuples.

        Returns:
            Merged character list sorted by total frequency.
        """
        # Collect per-chapter results
        all_names_global: Counter[str] = Counter()
        chapter_appearances: dict[str, dict[int, int]] = {}
        per_chapter_surnames: set[str] = set()

        for ch_num, text in chapters:
            dialogue_names = self._extract_dialogue_names(text)
            known_surnames = self._collect_surnames(dialogue_names)
            per_chapter_surnames.update(known_surnames)
            text_surnames = self._scan_all_surnames(text)
            combined_surnames = known_surnames | per_chapter_surnames | text_surnames

            narrative_names = self._extract_narrative_names(
                text, combined_surnames,
                known_names=set(dialogue_names.keys()),
            )
            discovered = self._discover_narrative_only_names(
                text, combined_surnames, dialogue_names
            )
            narrative_names.update(discovered)
            title_refs = self._extract_title_references(text)

            chapter_names: Counter[str] = Counter()
            chapter_names.update(dialogue_names)
            chapter_names.update(narrative_names)
            chapter_names.update(title_refs)
            chapter_names = self._filter_noise(chapter_names)
            chapter_names = self._deduplicate_substrings(chapter_names)

            all_names_global.update(chapter_names)
            for name, count in chapter_names.items():
                if name not in chapter_appearances:
                    chapter_appearances[name] = {}
                chapter_appearances[name][ch_num] = count

        # Global dedup and noise filter
        all_names_global = self._filter_noise(all_names_global)
        all_names_global = self._deduplicate_substrings(all_names_global)

        # Cluster across all chapters
        clusters = self._cluster_by_surname(all_names_global)
        alias_map = self._resolve_aliases(clusters, all_names_global)

        # Build characters with cross-chapter data
        characters = self._build_characters_multi(
            all_names_global, alias_map, chapter_appearances
        )
        characters.sort(key=lambda c: c.frequency, reverse=True)
        return characters

    # ── Pass 1: Dialogue verb extraction ──────────────────────────────────────

    def _extract_dialogue_names(self, text: str) -> Counter[str]:
        """Find names immediately before dialogue verbs.

        Compound verbs (笑道, 冷笑道, etc.) are specific enough to accept any name.
        Single-char verbs (说, 道, 问, etc.) are ambiguous in Chinese, so candidates
        must start with a known surname to be accepted.
        """
        counts: Counter[str] = Counter()

        # Pass A: Compound verbs — high confidence, accept any candidate
        for match in COMPOUND_VERB_REGEX.finditer(text):
            candidate = match.group(1)
            if candidate not in FALSE_POSITIVE_PREFIXES and candidate not in NOISE_WORDS:
                counts[candidate] += 1

        # Pass B: Single-char verbs — require surname validation
        for match in SINGLE_VERB_REGEX.finditer(text):
            candidate = match.group(1)
            if candidate in FALSE_POSITIVE_PREFIXES or candidate in NOISE_WORDS:
                continue
            # Must start with a known surname
            has_surname = False
            for cs in COMPOUND_SURNAMES:
                if candidate.startswith(cs):
                    has_surname = True
                    break
            if not has_surname and candidate and candidate[0] in SINGLE_SURNAMES:
                has_surname = True
            if has_surname:
                counts[candidate] += 1

        return counts

    # ── Pass 2: Surname-based narrative extraction ────────────────────────────

    @staticmethod
    def _scan_all_surnames(text: str) -> set[str]:
        """Scan text for surnames that likely belong to character names.

        Only returns a surname if it produces at least one plausible name candidate
        (surname + 1-2 chars that appears at a sentence start).
        """
        found: set[str] = set()
        # Sentence start pattern: after 。！？\n" or at start
        sentence_start_names: set[str] = set()
        for m in re.finditer(r'(?:^|[。！？\n\r""」])\s*([\u4e00-\u9fff]{2,4})', text):
            sentence_start_names.add(m.group(1))

        # Check compound surnames
        for cs in COMPOUND_SURNAMES:
            if cs in text:
                # Check if any sentence-start name starts with this surname
                for name in sentence_start_names:
                    if name.startswith(cs) and len(name) > len(cs):
                        found.add(cs)
                        break

        # Check single-char surnames
        for char in set(text):
            if char in SINGLE_SURNAMES:
                for name in sentence_start_names:
                    if name.startswith(char) and len(name) >= 2:
                        found.add(char)
                        break

        return found

    @staticmethod
    def _collect_surnames(names: Counter[str]) -> set[str]:
        """Extract unique surnames from a set of names."""
        surnames: set[str] = set()
        for name in names:
            for cs in COMPOUND_SURNAMES:
                if name.startswith(cs):
                    surnames.add(cs)
                    break
            else:
                if name and name[0] in SINGLE_SURNAMES:
                    surnames.add(name[0])
        return surnames

    def _extract_narrative_names(
        self, text: str, known_surnames: set[str],
        known_names: set[str] | None = None,
    ) -> Counter[str]:
        """Count occurrences of known character names in narrative text.

        If known_names is provided (from dialogue pass), only counts those exact names.
        Otherwise falls back to surname + 1-2 char scanning (less precise).
        """
        counts: Counter[str] = Counter()
        if not known_surnames:
            return counts

        if known_names:
            # Precise mode: only count names we already know from dialogue
            for name in known_names:
                # Count non-overlapping occurrences
                n = text.count(name)
                if n > 0:
                    counts[name] += n
            return counts

        # Fallback: scan for surname + 1-2 chars (only used when no dialogue names)
        for surname in known_surnames:
            pattern = re.compile(
                rf"{re.escape(surname)}([\u4e00-\u9fff]{{1,2}})"
            )
            for match in pattern.finditer(text):
                given = match.group(1)
                full_name = surname + given
                if full_name in NOISE_WORDS:
                    continue
                counts[full_name] += 1

        return counts

    @staticmethod
    def _discover_narrative_only_names(
        text: str, known_surnames: set[str], dialogue_names: Counter[str],
    ) -> Counter[str]:
        """Find character names that appear in narrative but never before dialogue verbs.

        Uses sentence-start heuristic: names often appear at the beginning of a
        sentence (after。！？\n or at position 0). Also requires 3+ occurrences
        to reduce false positives.
        """
        counts: Counter[str] = Counter()
        dialogue_name_set = set(dialogue_names.keys())
        title_suffix_set = set(TITLE_SUFFIXES)

        # Common chars that follow surnames but aren't given names
        # (common words, particles, verbs)
        # Characters that cannot be the FIRST char of a given name
        bad_given_first = set(
            "的了是在不有这那也都要还可就"
            "和与而但却又且或则因为所以"
            "到从把被让给向往对"
            "说道问叫喊吼骂笑哭"
            "知别想能会得着过来去"
            "很太最更真好大小多少"
            "家人们子女儿上下里外中前后"
            "正头内究并感必计便一将掌"
            "竟醒府城星总地"
        )

        # Sentence boundary pattern: matches start of sentence
        sentence_starts = set()
        for m in re.finditer(r'(?:^|[。！？\n\r""])([\u4e00-\u9fff]{2,4})', text):
            sentence_starts.add(m.group(1))

        for surname in known_surnames:
            # Scan for 2-char given names (e.g., 苏明硕, 苏清影)
            pattern_2 = re.compile(rf"{re.escape(surname)}([\u4e00-\u9fff]{{2}})")
            raw_2: Counter[str] = Counter()
            for match in pattern_2.finditer(text):
                given = match.group(1)
                full_name = surname + given
                if full_name in NOISE_WORDS or full_name in dialogue_name_set:
                    continue
                if given in title_suffix_set:
                    continue
                if given[0] in bad_given_first:
                    continue
                raw_2[full_name] += 1

            # Scan for 1-char given names (e.g., 沈辰)
            pattern_1 = re.compile(rf"{re.escape(surname)}([\u4e00-\u9fff])")
            raw_1: Counter[str] = Counter()
            for match in pattern_1.finditer(text):
                given = match.group(1)
                full_name = surname + given
                if full_name in NOISE_WORDS or full_name in dialogue_name_set:
                    continue
                if given in title_suffix_set or given in bad_given_first:
                    continue
                raw_1[full_name] += 1

            # Accept: 3+ occurrences, OR 2+ AND appears at sentence start
            for name, count in raw_2.items():
                if count >= 3 or (count >= 2 and name in sentence_starts):
                    counts[name] = count
            for name, count in raw_1.items():
                if count >= 3 or (count >= 2 and name in sentence_starts):
                    counts[name] = count

        return counts

    # ── Pass 3: Title reference extraction ────────────────────────────────────

    def _extract_title_references(self, text: str) -> Counter[str]:
        """Find surname + title suffix patterns (e.g., 沈老爷子, 苏夫人)."""
        counts: Counter[str] = Counter()
        for match in self._title_regex.finditer(text):
            full_ref = match.group(0)  # e.g., "沈老爷子"
            if full_ref not in NOISE_WORDS:
                counts[full_ref] += 1
        return counts

    # ── Filtering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_noise(names: Counter[str]) -> Counter[str]:
        """Remove known noise words and structurally implausible candidates."""
        family_place_chars = set("家城府")
        # Characters that never END a real name
        bad_ending_chars = set("着地了的过来去上下")
        # Characters that never START a real name (pronouns, prepositions, etc.)
        bad_start_chars = set("他她我你它只不这那很太最更")
        # Characters that should never appear in a given name
        # (particles, demonstratives, common function words)
        bad_given_any_chars = set("这那些个上下不没被把给让往")

        filtered: Counter[str] = Counter()
        for name, count in names.items():
            if name in NOISE_WORDS:
                continue
            if name in FALSE_POSITIVE_PREFIXES:
                continue
            if len(name) < 2:
                continue

            # Structural: names ending with verb particles (笑着, 冷淡地)
            if name[-1] in bad_ending_chars:
                continue

            # Structural: names starting with pronouns/prepositions
            if name[0] in bad_start_chars:
                continue

            # Structural: surname + 家/城/府 + X
            if len(name) >= 3:
                if name[0] in SINGLE_SURNAMES and name[1] in family_place_chars:
                    continue
                if any(name.startswith(cs) and len(name) > len(cs)
                       and name[len(cs)] in family_place_chars
                       for cs in COMPOUND_SURNAMES):
                    continue

            # 3+ char names without a valid surname are almost always noise
            # (e.g., 微笑着, 经听路人, 方成大)
            if len(name) >= 3:
                has_surname = name[0] in SINGLE_SURNAMES or any(
                    name.startswith(cs) for cs in COMPOUND_SURNAMES
                )
                if not has_surname:
                    continue

            # 4+ char names without compound surname are noise
            # (e.g., 明硕继续 — 明 is a surname but 明硕继续 is not a name)
            if len(name) >= 4 and name[0] not in SINGLE_SURNAMES:
                has_compound = any(name.startswith(cs) for cs in COMPOUND_SURNAMES)
                if not has_compound:
                    continue

            # Given name contains function words → phrase, not a name
            # (e.g., 于这些, 明白这个, 经常来信)
            surname, given = _split_name(name)
            if given and any(c in bad_given_any_chars for c in given):
                continue

            # Single-char "given name" that's a common word → noise
            # (e.g., 马车, 和图 — surname + common noun char)
            common_single_given = set("车图书白亮显常")
            if len(given) == 1 and given in common_single_given:
                continue

            # Ambiguous surnames: chars that are BOTH a surname AND a very common
            # function word. These need extra evidence — only keep if the full name
            # appears 5+ times (reduces false positives from 于/和/经/时/明)
            ambiguous_surnames = {"于", "和", "经", "时"}
            if surname in ambiguous_surnames and count < 5:
                continue

            filtered[name] = count
        return filtered

    @staticmethod
    def _deduplicate_substrings(names: Counter[str]) -> Counter[str]:
        """Remove names that are substrings of other names.

        Rules:
        - If shorter is a PREFIX of longer and they share a surname:
          keep LONGER (苏梦 vs 苏梦柠 → keep 苏梦柠, it's the full name)
        - If shorter is CONTAINED in longer but NOT a prefix:
          keep the more frequent one (蓝溪 vs X蓝溪X → keep 蓝溪)
        - If longer is a noise extension of shorter (蓝溪从, 蓝溪的):
          keep shorter if it has a known surname
        """
        name_list = list(names.keys())
        to_remove: set[str] = set()

        for short in name_list:
            for long in name_list:
                if short == long or short in to_remove or long in to_remove:
                    continue
                if len(short) >= len(long):
                    continue
                if short not in long:
                    continue

                # short is a substring of long
                short_surname = _split_name(short)[0]
                long_surname = _split_name(long)[0]

                if long.startswith(short) and short_surname and short_surname == long_surname:
                    # Prefix case: 苏梦 vs 苏梦柠, or 沈辰 vs 沈辰身
                    short_given_len = len(short) - len(short_surname)
                    long_given_len = len(long) - len(long_surname)

                    if names[short] > names[long] * 3:
                        # Short is WAY more frequent → it's the real name
                        # (沈辰×59 vs 沈辰身×4 → keep 沈辰)
                        to_remove.add(long)
                    elif short_given_len == 1 and long_given_len == 2 and names[long] >= 3:
                        # surname+1 vs surname+2, close frequencies:
                        # prefer 3-char name (苏梦 vs 苏梦柠 → keep 苏梦柠)
                        to_remove.add(short)
                    elif names[long] > names[short]:
                        to_remove.add(short)
                    else:
                        to_remove.add(long)
                elif short_surname and long_surname == short_surname and long.startswith(short_surname):
                    # Same surname, short is a real name, long is noise (蓝溪 vs 蓝溪从)
                    to_remove.add(long)
                else:
                    # Generic case: keep the more frequent
                    if names[short] >= names[long]:
                        to_remove.add(long)
                    else:
                        to_remove.add(short)

        return Counter({n: c for n, c in names.items() if n not in to_remove})

    # ── Surname clustering ────────────────────────────────────────────────────

    @staticmethod
    def _cluster_by_surname(names: Counter[str]) -> dict[str, list[str]]:
        """Group names by their surname."""
        clusters: dict[str, list[str]] = {}
        for name in names:
            surname = None
            # Check compound surnames first
            for cs in COMPOUND_SURNAMES:
                if name.startswith(cs):
                    surname = cs
                    break
            if surname is None and name and name[0] in SINGLE_SURNAMES:
                surname = name[0]

            if surname is not None:
                if surname not in clusters:
                    clusters[surname] = []
                if name not in clusters[surname]:
                    clusters[surname].append(name)
        return clusters

    # ── Alias resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_aliases(
        clusters: dict[str, list[str]], name_counts: Counter[str]
    ) -> dict[str, str]:
        """
        Build alias → primary_name mapping.

        Rules:
        - Different proper names in the same family are KEPT SEPARATE (沈辰 ≠ 沈斐)
        - surname+title (沈老爷子, 沈少爷) maps to most frequent proper name in cluster
        - Only title references become aliases, not proper names
        """
        alias_map: dict[str, str] = {}  # alias → primary

        for surname, names in clusters.items():
            if not names:
                continue

            # Separate proper names from title references
            proper_names = []
            title_names = []
            for n in names:
                rest = n[len(surname):]
                is_title = any(rest == sfx for sfx in TITLE_SUFFIXES)
                if is_title:
                    title_names.append(n)
                else:
                    proper_names.append(n)

            if not proper_names:
                # All are title references — keep them as separate characters
                continue

            # Only map TITLE references to the most frequent proper name
            # Do NOT merge different proper names (沈辰 and 沈斐 are different people)
            if title_names:
                primary = max(proper_names, key=lambda n: name_counts.get(n, 0))
                for t in title_names:
                    alias_map[t] = primary

        return alias_map

    # ── Build final character list ────────────────────────────────────────────

    @staticmethod
    def _build_characters(
        name_counts: Counter[str],
        alias_map: dict[str, str],
        chapter_num: int,
    ) -> list[HarvestedCharacter]:
        """Build character list for a single chapter."""
        # Group by primary name
        primary_data: dict[str, dict] = {}

        for name, count in name_counts.items():
            primary = alias_map.get(name, name)
            if primary not in primary_data:
                # Determine surname and given name
                surname, given = _split_name(primary)
                primary_data[primary] = {
                    "surname": surname,
                    "given_name": given,
                    "aliases": [],
                    "frequency": 0,
                    "chapter_appearances": {},
                }
            primary_data[primary]["frequency"] += count
            primary_data[primary]["chapter_appearances"][chapter_num] = (
                primary_data[primary]["chapter_appearances"].get(chapter_num, 0) + count
            )
            if name != primary and name not in primary_data[primary]["aliases"]:
                primary_data[primary]["aliases"].append(name)

        return [
            HarvestedCharacter(
                name=name,
                surname=data["surname"],
                given_name=data["given_name"],
                aliases=data["aliases"],
                frequency=data["frequency"],
                chapter_appearances=data["chapter_appearances"],
            )
            for name, data in primary_data.items()
        ]

    @staticmethod
    def _build_characters_multi(
        name_counts: Counter[str],
        alias_map: dict[str, str],
        chapter_appearances: dict[str, dict[int, int]],
    ) -> list[HarvestedCharacter]:
        """Build character list across multiple chapters."""
        primary_data: dict[str, dict] = {}

        for name, count in name_counts.items():
            primary = alias_map.get(name, name)
            if primary not in primary_data:
                surname, given = _split_name(primary)
                primary_data[primary] = {
                    "surname": surname,
                    "given_name": given,
                    "aliases": [],
                    "frequency": 0,
                    "chapter_appearances": {},
                }
            primary_data[primary]["frequency"] += count
            if name != primary and name not in primary_data[primary]["aliases"]:
                primary_data[primary]["aliases"].append(name)

            # Merge chapter appearances
            if name in chapter_appearances:
                for ch, ch_count in chapter_appearances[name].items():
                    primary_data[primary]["chapter_appearances"][ch] = (
                        primary_data[primary]["chapter_appearances"].get(ch, 0) + ch_count
                    )

        return [
            HarvestedCharacter(
                name=name,
                surname=data["surname"],
                given_name=data["given_name"],
                aliases=data["aliases"],
                frequency=data["frequency"],
                chapter_appearances=data["chapter_appearances"],
            )
            for name, data in primary_data.items()
        ]


def _split_name(name: str) -> tuple[str, str]:
    """Split a Chinese name into (surname, given_name)."""
    for cs in COMPOUND_SURNAMES:
        if name.startswith(cs):
            return cs, name[len(cs):]
    if name and name[0] in SINGLE_SURNAMES:
        return name[0], name[1:]
    return "", name
