"""
Story Bible: incremental, structured knowledge base for novel narration.

The Bible is built chapter-by-chapter via LLM extraction, validated against
deterministic harvester output. It stores characters, relationships, world
facts, plot events, and loose threads.

Design principles:
- LLM extracts, regex validates (no hallucinated characters)
- Evidence-based updates (every change cites the source text)
- Tiered characters (active/dormant/retired) to manage context size
- JSON-serializable for persistence between runs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.llm_client import LLMClient
from src.narration.harvester import CharacterHarvester, HarvestedCharacter
from src.narration.prompts import (
    BIBLE_BATCH_UPDATE_PROMPT,
    BIBLE_UPDATE_PROMPT,
    BIBLE_UPDATE_SYSTEM,
    CONSOLIDATION_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────


class RelationshipEntry(BaseModel):
    """A single relationship state change with evidence."""

    chapter: int
    state: str  # ally, enemy, love_interest, family, neutral, betrayed, complicated
    detail: str  # what happened
    evidence: str = ""  # quote from the chapter


class CharacterBible(BaseModel):
    """Accumulated knowledge about a single character."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    surname: str = ""
    role: str = ""  # protagonist, antagonist, ally, neutral, minor
    description: str = ""  # accumulated physical/personality description
    relationships: dict[str, list[RelationshipEntry]] = Field(default_factory=dict)
    arc_status: str = ""  # current state in story
    first_appeared: int = 0  # chapter number
    last_appeared: int = 0
    tier: str = "active"  # active, dormant, retired

    def current_relationship(self, other_name: str) -> str | None:
        """Get the latest relationship state with another character."""
        entries = self.relationships.get(other_name, [])
        return entries[-1].state if entries else None


class WorldFact(BaseModel):
    """A piece of world-building information."""

    fact: str
    chapter: int
    category: str = ""  # setting, power_system, faction, rule, event


class LooseThread(BaseModel):
    """An unresolved plot point or mystery."""

    chapter: int
    detail: str
    characters: list[str] = Field(default_factory=list)
    status: str = "unresolved"  # unresolved, resolved, red_herring
    resolved_chapter: int | None = None


class PlotEvent(BaseModel):
    """A significant plot event."""

    chapter: int
    summary: str
    characters_involved: list[str] = Field(default_factory=list)


class StoryBible(BaseModel):
    """The complete knowledge base for a novel."""

    novel_id: str
    last_processed_chapter: int = 0
    characters: dict[str, CharacterBible] = Field(default_factory=dict)
    world: list[WorldFact] = Field(default_factory=list)
    loose_threads: list[LooseThread] = Field(default_factory=list)
    timeline: list[PlotEvent] = Field(default_factory=list)
    updated_at: str = ""

    def active_characters(self) -> dict[str, CharacterBible]:
        """Get characters currently active in the story."""
        return {n: c for n, c in self.characters.items() if c.tier == "active"}

    def dormant_characters(self) -> dict[str, CharacterBible]:
        """Get characters not seen recently."""
        return {n: c for n, c in self.characters.items() if c.tier == "dormant"}

    def promote_character(self, name: str) -> None:
        """Move a character from dormant/retired to active."""
        if name in self.characters:
            self.characters[name].tier = "active"

    def demote_character(self, name: str) -> None:
        """Move a character from active to dormant."""
        if name in self.characters:
            self.characters[name].tier = "dormant"

    def auto_manage_tiers(self, current_chapter: int, dormant_threshold: int = 50) -> None:
        """Automatically demote characters not seen in dormant_threshold chapters."""
        for name, char in self.characters.items():
            if char.tier == "active" and (current_chapter - char.last_appeared) > dormant_threshold:
                char.tier = "dormant"
            elif char.tier == "retired":
                pass  # retired stays retired

    def get_context_for_chapter(self) -> str:
        """Serialize the relevant Bible subset for LLM context injection."""
        lines = []

        # Active characters
        active = self.active_characters()
        if active:
            lines.append("=== ACTIVE CHARACTERS ===")
            for name, char in active.items():
                alias_str = f" (aliases: {', '.join(char.aliases)})" if char.aliases else ""
                lines.append(f"- {name}{alias_str}: {char.description}")
                lines.append(f"  Role: {char.role} | Status: {char.arc_status}")
                for other, rels in char.relationships.items():
                    if rels:
                        latest = rels[-1]
                        lines.append(f"  → {other}: {latest.state} - {latest.detail}")

        # Dormant characters (summary only)
        dormant = self.dormant_characters()
        if dormant:
            lines.append("\n=== DORMANT CHARACTERS ===")
            for name, char in dormant.items():
                lines.append(f"- {name}: {char.role}, last seen ch.{char.last_appeared}")

        # World facts
        if self.world:
            lines.append("\n=== WORLD ===")
            for fact in self.world[-20:]:  # last 20 facts
                lines.append(f"- [{fact.category}] {fact.fact}")

        # Unresolved threads
        unresolved = [t for t in self.loose_threads if t.status == "unresolved"]
        if unresolved:
            lines.append("\n=== UNRESOLVED THREADS ===")
            for thread in unresolved[-10:]:  # last 10
                lines.append(f"- (ch.{thread.chapter}) {thread.detail}")

        # Recent timeline
        if self.timeline:
            lines.append("\n=== RECENT EVENTS ===")
            for event in self.timeline[-10:]:  # last 10
                lines.append(f"- (ch.{event.chapter}) {event.summary}")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save Bible to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        data["updated_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> StoryBible:
        """Load Bible from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


# ── Bible Update Schema ──────────────────────────────────────────────────────


class BibleUpdate(BaseModel):
    """Structured update returned by LLM for a single chapter."""

    new_characters: list[dict] = Field(default_factory=list)
    updated_characters: list[dict] = Field(default_factory=list)
    relationship_changes: list[dict] = Field(default_factory=list)
    new_world_facts: list[dict] = Field(default_factory=list)
    new_loose_threads: list[dict] = Field(default_factory=list)
    resolved_threads: list[str] = Field(default_factory=list)
    plot_events: list[dict] = Field(default_factory=list)


# ── Bible Builder ─────────────────────────────────────────────────────────────


class BibleBuilder:
    """Incrementally builds a Story Bible chapter by chapter."""

    def __init__(
        self,
        llm: LLMClient,
        harvester: CharacterHarvester | None = None,
        model_override: str | None = None,
    ):
        self.llm = llm
        self.harvester = harvester or CharacterHarvester()
        self.model_override = model_override  # e.g. "deepseek-reasoner"

    def build_chapter(
        self,
        bible: StoryBible,
        chapter_text: str,
        chapter_num: int,
        harvested: list[HarvestedCharacter] | None = None,
    ) -> StoryBible:
        """
        Process one chapter and update the Bible.

        1. Harvest character names (regex)
        2. Request structured update from LLM
        3. Validate update against harvester output
        4. Merge into Bible
        """
        # Step 1: Harvest if not provided
        if harvested is None:
            harvested = self.harvester.harvest_chapter(chapter_text, chapter_num)

        harvested_names = {h.name for h in harvested}
        for h in harvested:
            harvested_names.update(h.aliases)

        # Step 2: LLM extraction
        logger.info("Requesting Bible update for chapter %d...", chapter_num)
        update = self._request_update(bible, chapter_text, chapter_num, harvested)

        # Step 3: Validate
        update = self._validate_update(update, harvested_names, chapter_texts=chapter_text)

        # Step 4: Merge
        bible = self._merge_update(bible, update, chapter_num)
        bible.last_processed_chapter = chapter_num
        bible.auto_manage_tiers(chapter_num)

        return bible

    def build_batch(
        self,
        bible: StoryBible,
        chapters: list[tuple[int, str]],
        harvested: list[HarvestedCharacter] | None = None,
    ) -> StoryBible:
        """
        Process multiple chapters in a single LLM call (5-10x faster).

        Args:
            bible: Current Story Bible.
            chapters: List of (chapter_num, chapter_text) tuples.
            harvested: Pre-harvested characters across all chapters.

        Returns:
            Updated Bible.
        """
        if not chapters:
            return bible

        # Harvest if not provided
        if harvested is None:
            harvested = self.harvester.harvest_novel(chapters)

        harvested_names = {h.name for h in harvested}
        for h in harvested:
            harvested_names.update(h.aliases)

        start_ch = chapters[0][0]
        end_ch = chapters[-1][0]

        # Combine chapter texts with separators
        combined = "\n\n".join(
            f"=== 第{ch_num}章 ===\n{text}"
            for ch_num, text in chapters
        )
        # Cap total text to fit in context
        if len(combined) > 15000:
            combined = combined[:15000] + "\n\n[后续内容省略...]"

        harvested_info = "\n".join(
            f"- {h.name} (freq: {h.frequency}, aliases: {h.aliases})"
            for h in harvested[:20]  # top 20 characters
        )

        context = bible.get_context_for_chapter()

        prompt = BIBLE_BATCH_UPDATE_PROMPT.format(
            start_chapter=start_ch,
            end_chapter=end_ch,
            bible_context=context if context.strip() else "(Empty — first chapters)",
            harvested_characters=harvested_info,
            chapters_text=combined,
        )

        logger.info("Requesting batch Bible update for chapters %d-%d...", start_ch, end_ch)
        raw = self.llm.chat_json(
            prompt=prompt,
            system=BIBLE_UPDATE_SYSTEM,
            temperature=0.15,
            model=self.model_override,
        )

        raw = self._sanitize_raw_response(raw)
        update = BibleUpdate.model_validate(raw)
        update = self._validate_update(update, harvested_names, chapter_texts=combined)
        bible = self._merge_update(bible, update, end_ch)
        bible.last_processed_chapter = end_ch
        bible.auto_manage_tiers(end_ch)

        return bible

    def _request_update(
        self,
        bible: StoryBible,
        chapter_text: str,
        chapter_num: int,
        harvested: list[HarvestedCharacter],
    ) -> BibleUpdate:
        """Ask LLM to extract structured update from chapter."""
        context = bible.get_context_for_chapter()
        harvested_info = "\n".join(
            f"- {h.name} (freq: {h.frequency}, aliases: {h.aliases})"
            for h in harvested
        )

        prompt = BIBLE_UPDATE_PROMPT.format(
            chapter_num=chapter_num,
            bible_context=context if context.strip() else "(Empty — this is the first chapter)",
            harvested_characters=harvested_info,
            chapter_text=chapter_text[:8000],  # cap to avoid context overflow
        )

        raw = self.llm.chat_json(
            prompt=prompt,
            system=BIBLE_UPDATE_SYSTEM,
            temperature=0.15,
            model=self.model_override,
        )

        raw = self._sanitize_raw_response(raw)
        return BibleUpdate.model_validate(raw)

    @staticmethod
    def _sanitize_raw_response(raw: dict | list) -> dict:
        """Fix common LLM output issues before Pydantic validation.

        Handles:
        - world_facts as strings instead of dicts
        - loose_threads as strings instead of dicts
        - plot_events as strings instead of dicts
        - Missing keys
        """
        if not isinstance(raw, dict):
            return {"new_characters": [], "updated_characters": [],
                    "relationship_changes": [], "new_world_facts": [],
                    "new_loose_threads": [], "resolved_threads": [], "plot_events": []}

        # Fix world_facts: string → dict
        if "new_world_facts" in raw:
            fixed = []
            for item in raw["new_world_facts"]:
                if isinstance(item, str):
                    fixed.append({"fact": item, "category": ""})
                elif isinstance(item, dict):
                    fixed.append(item)
            raw["new_world_facts"] = fixed

        # Fix loose_threads: string → dict
        if "new_loose_threads" in raw:
            fixed = []
            for item in raw["new_loose_threads"]:
                if isinstance(item, str):
                    fixed.append({"detail": item, "characters": []})
                elif isinstance(item, dict):
                    fixed.append(item)
            raw["new_loose_threads"] = fixed

        # Fix plot_events: string → dict
        if "plot_events" in raw:
            fixed = []
            for item in raw["plot_events"]:
                if isinstance(item, str):
                    fixed.append({"summary": item, "characters_involved": []})
                elif isinstance(item, dict):
                    fixed.append(item)
            raw["plot_events"] = fixed

        # Fix resolved_threads: dict → string
        if "resolved_threads" in raw:
            fixed = []
            for item in raw["resolved_threads"]:
                if isinstance(item, dict):
                    fixed.append(item.get("detail", str(item)))
                elif isinstance(item, str):
                    fixed.append(item)
            raw["resolved_threads"] = fixed

        return raw

    @staticmethod
    def _validate_update(
        update: BibleUpdate, harvested_names: set[str],
        chapter_texts: str = "",
    ) -> BibleUpdate:
        """Cross-check LLM update against harvested names AND raw text.

        A character is accepted if:
        1. It's in the harvested names set (regex found it), OR
        2. Its name appears in the raw chapter text (LLM found a real name the regex missed)

        A character is rejected only if it doesn't appear in the text at all (hallucinated).
        """
        validated_new = []
        for char in update.new_characters:
            name = char.get("name", "")
            if not name:
                continue
            if name in harvested_names:
                validated_new.append(char)
            elif chapter_texts and name in chapter_texts:
                # Name exists in raw text — LLM found a real character the regex missed
                logger.info("Accepted LLM-found character: %s (in text but not harvested)", name)
                validated_new.append(char)
            else:
                logger.warning(
                    "Rejected hallucinated character: %s (not in text or harvested names)", name
                )
        update.new_characters = validated_new

        return update

    @staticmethod
    def _merge_update(
        bible: StoryBible, update: BibleUpdate, chapter_num: int
    ) -> StoryBible:
        """Apply a validated update to the Bible."""

        # New characters (with dedup: merge if name is a nickname of existing char)
        for char_data in update.new_characters:
            name = char_data.get("name", "")
            if not name or name in bible.characters:
                continue

            # Check for duplicates in both directions:
            # 1. New name is a nickname of existing char: 若若 → merge into 范若若
            # 2. Existing char is a nickname of new name: 若若 exists, 范若若 comes in → promote
            merged = False
            for existing_name, existing_char in list(bible.characters.items()):
                if name in existing_name or existing_name.endswith(name):
                    # New name is shorter, existing is the full name — merge into existing
                    if name not in existing_char.aliases:
                        existing_char.aliases.append(name)
                    existing_char.last_appeared = chapter_num
                    logger.info("Merged '%s' as alias of '%s'", name, existing_name)
                    merged = True
                    break
                if existing_name in name or name.endswith(existing_name):
                    # Existing is shorter, new name is the full name — promote
                    # Move existing data to new name, keep old as alias
                    existing_char.aliases.append(existing_name)
                    existing_char.name = name
                    existing_char.last_appeared = chapter_num
                    # Update description if new one is provided
                    new_desc = char_data.get("description", "")
                    if new_desc and len(new_desc) > len(existing_char.description):
                        existing_char.description = new_desc
                    bible.characters[name] = existing_char
                    del bible.characters[existing_name]
                    logger.info("Promoted '%s' to '%s' (full name)", existing_name, name)
                    merged = True
                    break
                if name in existing_char.aliases:
                    existing_char.last_appeared = chapter_num
                    merged = True
                    break

            if not merged:
                bible.characters[name] = CharacterBible(
                    name=name,
                    aliases=char_data.get("aliases", []),
                    surname=char_data.get("surname", ""),
                    role=char_data.get("role", ""),
                    description=char_data.get("description", ""),
                    arc_status=char_data.get("arc_status", ""),
                    first_appeared=chapter_num,
                    last_appeared=chapter_num,
                    tier="active",
                )

        # Updated characters
        for char_data in update.updated_characters:
            name = char_data.get("name", "")
            if name not in bible.characters:
                continue
            char = bible.characters[name]
            if "description" in char_data and char_data["description"]:
                char.description = char_data["description"]
            if "arc_status" in char_data and char_data["arc_status"]:
                char.arc_status = char_data["arc_status"]
            if "role" in char_data and char_data["role"]:
                char.role = char_data["role"]
            char.last_appeared = chapter_num
            char.tier = "active"  # re-appeared, promote if dormant

        # Relationship changes
        for rel in update.relationship_changes:
            source = rel.get("source", "")
            target = rel.get("target", "")
            state = rel.get("state", "")
            if not source or not target or not state:
                continue
            if source in bible.characters:
                char = bible.characters[source]
                if target not in char.relationships:
                    char.relationships[target] = []
                char.relationships[target].append(
                    RelationshipEntry(
                        chapter=chapter_num,
                        state=state,
                        detail=rel.get("detail", ""),
                        evidence=rel.get("evidence", ""),
                    )
                )

        # World facts
        for fact_data in update.new_world_facts:
            bible.world.append(
                WorldFact(
                    fact=fact_data.get("fact", ""),
                    chapter=chapter_num,
                    category=fact_data.get("category", ""),
                )
            )

        # Loose threads
        for thread_data in update.new_loose_threads:
            bible.loose_threads.append(
                LooseThread(
                    chapter=chapter_num,
                    detail=thread_data.get("detail", ""),
                    characters=thread_data.get("characters", []),
                )
            )

        # Resolved threads
        for thread_detail in update.resolved_threads:
            for thread in bible.loose_threads:
                if thread.detail == thread_detail and thread.status == "unresolved":
                    thread.status = "resolved"
                    thread.resolved_chapter = chapter_num
                    break

        # Plot events
        for event_data in update.plot_events:
            bible.timeline.append(
                PlotEvent(
                    chapter=chapter_num,
                    summary=event_data.get("summary", ""),
                    characters_involved=event_data.get("characters_involved", []),
                )
            )

        return bible

    def consolidate(self, bible: StoryBible) -> StoryBible:
        """Review the full Bible for contradictions (run every N chapters)."""
        context = bible.get_context_for_chapter()
        prompt = CONSOLIDATION_PROMPT.format(bible_context=context)

        result = self.llm.chat_json(
            prompt=prompt,
            system=BIBLE_UPDATE_SYSTEM,
            temperature=0.1,
            model=self.model_override,
        )

        # Apply corrections if any
        if isinstance(result, dict):
            corrections = result.get("corrections", [])
            for fix in corrections:
                name = fix.get("character", "")
                if name in bible.characters:
                    if "role" in fix:
                        bible.characters[name].role = fix["role"]
                    if "arc_status" in fix:
                        bible.characters[name].arc_status = fix["arc_status"]

        return bible
