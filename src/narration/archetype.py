"""
Archetype taxonomy and assignment for 视频解说 narration.

Defines the complete set of Chinese video narration archetypes with semantic
validation rules to prevent misassignment (e.g., 渣男 for a father figure).
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from src.core.llm_client import LLMClient
from src.narration.bible import CharacterBible, StoryBible
from src.narration.harvester import HarvestedCharacter
from src.narration.prompts import ARCHETYPE_PROMPT, ARCHETYPE_SYSTEM

logger = logging.getLogger(__name__)


# ── Archetype Category ────────────────────────────────────────────────────────


class ArchetypeCategory(str, Enum):
    PROTAGONIST = "protagonist"
    LOVE_INTEREST = "love_interest"
    FAMILY = "family"
    ANTAGONIST = "antagonist"
    ALLY = "ally"
    AUTHORITY = "authority"
    MINOR = "minor"


# ── Archetype Definition ──────────────────────────────────────────────────────


class ArchetypeDefinition(BaseModel):
    """Defines when and how an archetype name should be used."""

    name: str  # e.g. "小帅"
    category: ArchetypeCategory
    description: str  # when to use this archetype
    gender: str | None = None  # "male", "female", or None (any)
    forbidden_roles: list[str] = Field(default_factory=list)
    unique: bool = True  # only one character can have this archetype


# ── Archetype Registry ────────────────────────────────────────────────────────

ARCHETYPE_REGISTRY: dict[str, ArchetypeDefinition] = {
    # ── Protagonists ──
    "小帅": ArchetypeDefinition(
        name="小帅", category=ArchetypeCategory.PROTAGONIST,
        description="Male protagonist — the main character the audience roots for",
        gender="male",
    ),
    "小美": ArchetypeDefinition(
        name="小美", category=ArchetypeCategory.LOVE_INTEREST,
        description="Main female love interest — the primary romantic interest",
        gender="female",
    ),

    # ── Family ──
    "老爷子": ArchetypeDefinition(
        name="老爷子", category=ArchetypeCategory.FAMILY,
        description="Respected elder male — grandfather, sect elder, old master",
        gender="male",
    ),
    "败家爹": ArchetypeDefinition(
        name="败家爹", category=ArchetypeCategory.FAMILY,
        description="Deadbeat/useless father — gambler, irresponsible parent",
        gender="male",
    ),
    "恶婆婆": ArchetypeDefinition(
        name="恶婆婆", category=ArchetypeCategory.FAMILY,
        description="Evil/controlling mother-in-law or stepmother figure",
        gender="female",
    ),
    "后妈": ArchetypeDefinition(
        name="后妈", category=ArchetypeCategory.FAMILY,
        description="Stepmother — usually antagonistic",
        gender="female",
    ),
    "继父": ArchetypeDefinition(
        name="继父", category=ArchetypeCategory.FAMILY,
        description="Stepfather — usually antagonistic or distant",
        gender="male",
    ),
    "岳父": ArchetypeDefinition(
        name="岳父", category=ArchetypeCategory.FAMILY,
        description="Father-in-law — may be supportive or antagonistic",
        gender="male",
    ),
    "亲妈": ArchetypeDefinition(
        name="亲妈", category=ArchetypeCategory.FAMILY,
        description="Birth mother — usually supportive and caring",
        gender="female",
    ),

    # ── Antagonists ──
    "渣男": ArchetypeDefinition(
        name="渣男", category=ArchetypeCategory.ANTAGONIST,
        description="Romantic scumbag — cheater, manipulative in love relationships. "
                    "NEVER use for father/uncle/elder roles",
        gender="male",
        forbidden_roles=["father", "stepfather", "father_in_law", "grandfather",
                         "uncle", "elder", "family"],
    ),
    "渣女": ArchetypeDefinition(
        name="渣女", category=ArchetypeCategory.ANTAGONIST,
        description="Female romantic scumbag — betrayer in romantic context. "
                    "NEVER use for mother/aunt roles",
        gender="female",
        forbidden_roles=["mother", "stepmother", "mother_in_law", "grandmother",
                         "aunt", "family"],
    ),
    "黄毛": ArchetypeDefinition(
        name="黄毛", category=ArchetypeCategory.ANTAGONIST,
        description="Street thug/bully — physically threatening delinquent type, "
                    "NOT a sophisticated villain",
        gender="male",
    ),
    "白莲花": ArchetypeDefinition(
        name="白莲花", category=ArchetypeCategory.ANTAGONIST,
        description="Fake-innocent manipulator — acts pure/helpless but schemes behind the back",
        gender="female",
    ),
    "绿茶": ArchetypeDefinition(
        name="绿茶", category=ArchetypeCategory.ANTAGONIST,
        description="Passive-aggressive fake-nice woman — similar to 白莲花 but more subtle",
        gender="female",
    ),
    "心机男": ArchetypeDefinition(
        name="心机男", category=ArchetypeCategory.ANTAGONIST,
        description="Calculating male schemer — plots behind the scenes, different from 渣男",
        gender="male",
    ),
    "校霸": ArchetypeDefinition(
        name="校霸", category=ArchetypeCategory.ANTAGONIST,
        description="School bully/tyrant — campus setting antagonist",
        gender="male",
    ),

    # ── Authority / Power ──
    "老大": ArchetypeDefinition(
        name="老大", category=ArchetypeCategory.AUTHORITY,
        description="Powerful boss figure — could be good or bad",
        gender=None,
    ),
    "大佬": ArchetypeDefinition(
        name="大佬", category=ArchetypeCategory.AUTHORITY,
        description="Hidden big shot — secretly very powerful or wealthy",
        gender=None,
    ),
    "少爷": ArchetypeDefinition(
        name="少爷", category=ArchetypeCategory.AUTHORITY,
        description="Rich young master — wealthy family's son",
        gender="male",
    ),
    "千金": ArchetypeDefinition(
        name="千金", category=ArchetypeCategory.AUTHORITY,
        description="Rich young lady — wealthy family's daughter",
        gender="female",
    ),
    "校花": ArchetypeDefinition(
        name="校花", category=ArchetypeCategory.AUTHORITY,
        description="School beauty queen — admired by everyone on campus",
        gender="female",
    ),

    # ── Allies ──
    "闺蜜": ArchetypeDefinition(
        name="闺蜜", category=ArchetypeCategory.ALLY,
        description="Female best friend — loyal female companion",
        gender="female",
    ),
    "兄弟": ArchetypeDefinition(
        name="兄弟", category=ArchetypeCategory.ALLY,
        description="Male best friend/bro — loyal male companion",
        gender="male",
    ),
    "小弟": ArchetypeDefinition(
        name="小弟", category=ArchetypeCategory.ALLY,
        description="Loyal follower/henchman",
        gender="male",
    ),
}


# ── Assignment Models ─────────────────────────────────────────────────────────


class ArchetypeAssignment(BaseModel):
    """A single character → archetype mapping."""

    original_name: str
    archetype: str
    reasoning: str = ""


# ── Validation ────────────────────────────────────────────────────────────────


def validate_assignment(
    name: str,
    archetype: str,
    bible: StoryBible,
) -> tuple[bool, str]:
    """
    Check if an archetype assignment is valid.

    Returns (is_valid, reason).
    """
    if archetype not in ARCHETYPE_REGISTRY:
        return False, f"Unknown archetype: {archetype}"

    defn = ARCHETYPE_REGISTRY[archetype]
    char = bible.characters.get(name)

    if char is None:
        return True, ""  # can't validate without Bible data

    # Gender check
    if defn.gender is not None:
        # Infer gender from Bible description/role if possible
        char_gender = _infer_gender(char)
        if char_gender and char_gender != defn.gender:
            return False, f"{archetype} requires gender={defn.gender}, but {name} appears to be {char_gender}"

    # Forbidden roles check
    if defn.forbidden_roles:
        char_roles = _get_character_relationship_roles(char)
        for forbidden in defn.forbidden_roles:
            if forbidden in char_roles:
                return False, f"{archetype} cannot be used for {name} who has role '{forbidden}'"

    return True, ""


def _infer_gender(char: CharacterBible) -> str | None:
    """Try to infer gender from character description, role, relationships, and name.

    Priority: role/title signals (strongest) > description phrases > relationship context > name chars (weakest)
    """
    desc = char.description + " " + char.role + " " + char.arc_status

    # Also include relationship details for context
    rel_context = ""
    for other, entries in char.relationships.items():
        for e in entries:
            rel_context += " " + e.detail
    full_context = desc + " " + rel_context

    male_score = 0
    female_score = 0

    # ── Tier 1: Role/title signals (weight=5, very reliable) ──
    male_roles = ["院长", "国师", "宗师", "将军", "统领", "都督", "太守",
                  "守备", "尚书", "宰相", "侍郎", "主办", "之子", "的儿子",
                  "的独子", "少爷", "公子", "伯爵", "皇帝", "太子", "王爷",
                  "大师", "道士", "和尚", "僧人"]
    female_roles = ["丫环", "侍女", "婢女", "丫鬟", "奶娘", "嬷嬷",
                    "之女", "的女儿", "的独女", "公主", "皇后", "贵妃",
                    "妃子", "小姐", "夫人", "娘子", "女儿"]
    male_score += sum(5 for s in male_roles if s in full_context)
    female_score += sum(5 for s in female_roles if s in full_context)

    # ── Tier 2: Description phrases (weight=3) ──
    male_desc = ["男子", "男人", "先生", "大哥", "兄弟", "老爷",
                 "身材魁梧", "英俊", "帅气", "国字脸", "面容刚毅",
                 "小男孩", "男孩", "高手", "仆人", "盲人", "刺客",
                 "商人", "杀手"]
    female_desc = ["女子", "女人", "少女", "姑娘", "闺蜜",
                   "容颜", "容貌", "美女", "美人", "绝美", "倾城",
                   "黑裙", "长裙", "粉裙", "红裙", "连衣裙",
                   "脸蛋", "肌肤", "胜雪", "如瓷", "白皙",
                   "温婉", "冷艳", "清纯", "纯净", "惊艳",
                   "小女孩", "女孩", "妹妹"]
    male_score += sum(3 for s in male_desc if s in desc)
    female_score += sum(3 for s in female_desc if s in desc)

    # ── Tier 3: Relationship context (weight=3) ──
    if "其子" in full_context or "儿子" in full_context:
        male_score += 3
    if "其女" in full_context or "女儿" in full_context:
        female_score += 3
    # "父亲"/"母亲" in description — but ONLY if it describes THIS character,
    # not someone else's parent (e.g., "范闲母亲的仆人" ≠ female)
    # Check: the word should NOT be preceded by 的 or followed by 的
    import re
    if re.search(r"(?<!的)父亲(?!的)", desc) or "爸爸" in desc or "爷爷" in desc:
        male_score += 3
    if re.search(r"(?<!的)母亲(?!的)", desc) or "妈妈" in desc or "奶奶" in desc:
        female_score += 3

    # ── Tier 4: Name chars (weight=1, UNRELIABLE for Chinese novels) ──
    # Only use very strong gendered chars, and with low weight
    # Many male characters have names with 萍/荷/冰/玉 etc.
    very_female_chars = set("婉娘嫣媛")  # only the most unambiguous
    very_male_chars = set("刚雄")
    for c in char.name:
        if c in very_female_chars:
            female_score += 1
        if c in very_male_chars:
            male_score += 1

    # ── Tier 5: Alias hints (weight=3) ──
    for alias in char.aliases:
        if any(s in alias for s in ["少女", "姑娘", "小姐", "夫人", "公主"]):
            female_score += 3
        if any(s in alias for s in ["少爷", "公子", "先生", "大人", "大师"]):
            male_score += 3

    if male_score > female_score:
        return "male"
    if female_score > male_score:
        return "female"
    return None


def _generate_descriptive_name(char: CharacterBible) -> str:
    """Generate a short, usable Chinese descriptive nickname for characters.

    Uses the character's role, description, and relationships to create
    a meaningful and distinctive name like 监察院长, 毒师父, 蒙面刺客.
    """
    gender = _infer_gender(char)
    desc = char.description + " " + char.role

    # ── Priority 1: Role/title based (most distinctive) ──
    role_map = {
        "院长": "院长大人",
        "国师": "国师大人",
        "宗师": "武林宗师",
        "将军": "将军",
        "宰相": "宰相大人",
        "尚书": "尚书大人",
        "守备": "守备大人",
        "总督": "总督大人",
        "管家": "管家",
        "队长": "队长",
        "侍卫": "侍卫",
        "丫环": "小丫环",
        "侍女": "侍女",
        "公主": "公主",
        "太子": "太子",
        "医生": "医生",
        "道士": "道士",
        "商人": "商人",
        "刺客": "刺客",
        "老师": "老师",
        "才子": "才子",
    }
    for keyword, label in role_map.items():
        if keyword in desc:
            prefix = char.surname if char.surname else ""
            return f"{prefix}{label}" if prefix else label

    # ── Priority 2: Description-based nicknames ──
    desc_map = {
        "盲人": "盲侠",
        "高手": "高手",
        "仆人": "老仆",
        "密探": "密探",
        "护卫": "护卫头领",
        "妹妹": "小妹",
        "女儿": "千金",
        "儿子": "公子",
        "小男孩": "小男孩",
        "小女孩": "小女孩",
    }
    for keyword, label in desc_map.items():
        if keyword in desc:
            prefix = char.surname if char.surname else ""
            return f"{prefix}{label}" if prefix else label

    # ── Priority 3: Relationship-based ──
    for other_name, entries in char.relationships.items():
        if entries:
            latest = entries[-1]
            detail = latest.detail
            if "父" in detail:
                return f"{char.surname}老爹" if char.surname else "老爹"
            if "母" in detail:
                return f"{char.surname}大妈" if char.surname else "大妈"
            if "妹" in detail:
                return f"{char.surname}小妹" if char.surname else "小妹"

    # ── Priority 4: Gender + surname ──
    if char.surname:
        if gender == "female":
            return f"{char.surname}姑娘"
        elif gender == "male":
            return f"{char.surname}兄"
        return f"{char.surname}某"

    if gender == "female":
        return "那姑娘"
    elif gender == "male":
        return "那小子"
    return "路人"


def _get_character_relationship_roles(char: CharacterBible) -> set[str]:
    """Extract relationship roles from a character's Bible entry."""
    roles: set[str] = set()
    for other_name, entries in char.relationships.items():
        for entry in entries:
            state = entry.state.lower()
            detail = entry.detail.lower()
            # Check for family roles
            for role_keyword in ["father", "stepfather", "father_in_law", "grandfather",
                                 "uncle", "elder", "mother", "stepmother", "mother_in_law",
                                 "grandmother", "aunt", "family"]:
                if role_keyword in state or role_keyword in detail:
                    roles.add(role_keyword)
            # Chinese family keywords
            family_map = {
                "父": "father", "爸": "father", "爹": "father",
                "继父": "stepfather", "岳父": "father_in_law",
                "爷": "grandfather", "叔": "uncle", "伯": "uncle",
                "母": "mother", "妈": "mother",
                "继母": "stepmother", "后妈": "stepmother",
                "婆": "mother_in_law",
                "姑": "aunt", "婶": "aunt",
            }
            for cn_key, en_role in family_map.items():
                if cn_key in detail:
                    roles.add(en_role)

    # Also check the role field
    if char.role:
        role_lower = char.role.lower()
        for keyword in ["father", "母", "父", "爸", "妈", "爷", "奶"]:
            if keyword in role_lower:
                if "父" in role_lower or "爸" in role_lower or "father" in role_lower:
                    roles.add("father")
                if "母" in role_lower or "妈" in role_lower or "mother" in role_lower:
                    roles.add("mother")
    return roles


# ── Archetype Assigner ────────────────────────────────────────────────────────


class ArchetypeAssigner:
    """Assigns archetype nicknames to characters using LLM + validation rules."""

    def __init__(self, llm: LLMClient, model_override: str | None = None):
        self.llm = llm
        self.model_override = model_override

    def assign(
        self,
        bible: StoryBible,
        harvested: list[HarvestedCharacter],
        locked: dict[str, str] | None = None,
        min_frequency: int = 3,
    ) -> dict[str, str]:
        """
        Assign archetypes to all active characters.

        Args:
            bible: Current Story Bible.
            harvested: Character frequency data from harvester.
            locked: Already-locked assignments from Narration Manifest.
            min_frequency: Minimum mention count to get a named archetype.

        Returns:
            Dict of {original_name: archetype_name}.
        """
        locked = locked or {}
        result = dict(locked)  # start with locked assignments

        # Build profiles for unassigned characters
        # Use ALL characters in Bible (not just active), prioritized by frequency
        freq_map = {h.name: h.frequency for h in harvested}
        unassigned = []
        for name, char in bible.characters.items():
            if name in result:
                continue  # already locked
            freq = freq_map.get(name, 0)
            if freq < min_frequency:
                # Minor character → descriptive Chinese nickname
                result[name] = _generate_descriptive_name(char)
                continue
            unassigned.append((name, char, freq))

        if not unassigned:
            return result

        # Build LLM prompt
        profiles = self._build_profiles(unassigned, bible)
        menu = self._build_archetype_menu(result)

        prompt = ARCHETYPE_PROMPT.format(
            character_profiles=profiles,
            archetype_menu=menu,
        )

        raw = self.llm.chat_json(
            prompt=prompt,
            system=ARCHETYPE_SYSTEM,
            temperature=0.2,
            model=self.model_override,
        )

        # Parse and validate assignments with uniqueness enforcement
        used_archetypes: set[str] = set(result.values())
        assignments = raw.get("assignments", []) if isinstance(raw, dict) else raw
        for item in assignments:
            name = item.get("original_name", "")
            archetype = item.get("archetype", "")
            if not name or not archetype:
                continue

            # Validate: if archetype is not in registry, it might be a descriptive
            # name the LLM generated (e.g., "旁边的服务员") — accept those directly
            if archetype not in ARCHETYPE_REGISTRY:
                if any("\u4e00" <= c <= "\u9fff" for c in archetype):
                    result[name] = archetype
                    continue
                logger.warning("Rejected non-Chinese archetype for %s: %s", name, archetype)
                continue

            defn = ARCHETYPE_REGISTRY[archetype]

            # Uniqueness check: if archetype is unique and already used, reject
            if defn.unique and archetype in used_archetypes:
                logger.warning("Rejected %s → %s: archetype already assigned to another character", name, archetype)
                char = bible.characters.get(name)
                if char:
                    result[name] = _generate_descriptive_name(char)
                continue

            is_valid, reason = validate_assignment(name, archetype, bible)
            if is_valid:
                result[name] = archetype
                used_archetypes.add(archetype)
            else:
                logger.warning("Rejected assignment %s → %s: %s", name, archetype, reason)
                char = bible.characters.get(name)
                if char:
                    result[name] = _generate_descriptive_name(char)

        # Safety: ensure protagonist (highest frequency) gets 小帅
        if harvested and "小帅" not in used_archetypes:
            protagonist = harvested[0]  # highest frequency
            if protagonist.name not in result or result.get(protagonist.name) in ("路人", "那小子"):
                result[protagonist.name] = "小帅"
                logger.info("Force-assigned protagonist %s → 小帅", protagonist.name)

        return result

    @staticmethod
    def _build_profiles(
        characters: list[tuple[str, CharacterBible, int]],
        bible: StoryBible,
    ) -> str:
        """Build character profiles for the LLM prompt."""
        lines = []
        for name, char, freq in characters:
            alias_str = f" (aliases: {', '.join(char.aliases)})" if char.aliases else ""
            lines.append(f"- {name}{alias_str}")
            lines.append(f"  Role: {char.role} | Frequency: {freq} mentions")
            lines.append(f"  Description: {char.description}")
            lines.append(f"  Current status: {char.arc_status}")
            # Key relationships
            for other, rels in char.relationships.items():
                if rels:
                    latest = rels[-1]
                    lines.append(f"  Relationship with {other}: {latest.state} — {latest.detail}")
        return "\n".join(lines)

    @staticmethod
    def _build_archetype_menu(already_assigned: dict[str, str]) -> str:
        """Build the archetype menu, marking already-used ones."""
        used = set(already_assigned.values())
        lines = []
        for name, defn in ARCHETYPE_REGISTRY.items():
            status = ""
            if name in used and defn.unique:
                status = " [TAKEN]"
            gender_str = f" ({defn.gender})" if defn.gender else " (any gender)"
            lines.append(f"- {name}{gender_str}: {defn.description}{status}")
        return "\n".join(lines)
