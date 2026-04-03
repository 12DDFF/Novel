"""
LLM prompt templates for the narration pipeline v2.

All prompts used by bible.py, archetype.py, and narrator_v2.py.
"""

# ── Bible Update Prompts ──────────────────────────────────────────────────────

BIBLE_UPDATE_SYSTEM = """You are a precise literary analyst for Chinese web novels.
You extract character information, relationships, world-building facts, and plot events.
You ONLY report what is explicitly stated or strongly implied in the text.
You NEVER invent characters, events, or relationships that aren't in the source.
Always respond in valid JSON format. Use Chinese for character names and descriptions."""

BIBLE_UPDATE_PROMPT = """Analyze Chapter {chapter_num} and extract updates to the Story Bible.

CURRENT STORY BIBLE:
{bible_context}

CHARACTERS FOUND BY TEXT ANALYSIS (these names are confirmed to exist in the chapter):
{harvested_characters}

CHAPTER {chapter_num} TEXT:
---
{chapter_text}
---

Extract the following as JSON. IMPORTANT RULES:
- Only reference characters from the "CHARACTERS FOUND" list above
- For every relationship change, include an "evidence" field with a short quote from the chapter
- For "role", use one of: protagonist, antagonist, ally, neutral, minor
- For relationship "state", use one of: ally, enemy, love_interest, family, neutral, betrayed, complicated, subordinate, superior

Return this JSON structure:
{{
  "new_characters": [
    {{
      "name": "character name",
      "aliases": ["alias1"],
      "surname": "surname",
      "role": "protagonist",
      "description": "physical and personality description from text",
      "arc_status": "current state/situation"
    }}
  ],
  "updated_characters": [
    {{
      "name": "existing character name",
      "description": "updated description if new info revealed",
      "arc_status": "updated current state",
      "role": "updated role if changed"
    }}
  ],
  "relationship_changes": [
    {{
      "source": "character A",
      "target": "character B",
      "state": "relationship type",
      "detail": "what happened between them",
      "evidence": "quote from text"
    }}
  ],
  "new_world_facts": [
    {{
      "fact": "world-building detail",
      "category": "setting|power_system|faction|rule|event"
    }}
  ],
  "new_loose_threads": [
    {{
      "detail": "unexplained or foreshadowed element",
      "characters": ["involved characters"]
    }}
  ],
  "resolved_threads": ["detail of previously unresolved thread that is now explained"],
  "plot_events": [
    {{
      "summary": "what happened in this chapter",
      "characters_involved": ["character names"]
    }}
  ]
}}

Only include sections that have actual content. Empty arrays are fine."""

BIBLE_BATCH_UPDATE_PROMPT = """Analyze chapters {start_chapter}-{end_chapter} and extract ALL updates to the Story Bible.

CURRENT STORY BIBLE:
{bible_context}

CHARACTERS FOUND BY TEXT ANALYSIS (confirmed to exist across these chapters):
{harvested_characters}

CHAPTERS {start_chapter}-{end_chapter}:
---
{chapters_text}
---

Extract updates from ALL chapters above into a SINGLE combined JSON response.
Use the same format as a single-chapter update. Combine all changes across all chapters.
For relationship_changes, use the chapter number where the change happened.
For plot_events, include one event per chapter summarizing what happened.

IMPORTANT:
- Only reference characters from the "CHARACTERS FOUND" list above
- Include "evidence" quotes for relationship changes
- Track which chapter each change came from

Return the SAME JSON format as a single-chapter update:
{{
  "new_characters": [...],
  "updated_characters": [...],
  "relationship_changes": [...],
  "new_world_facts": [...],
  "new_loose_threads": [...],
  "resolved_threads": [...],
  "plot_events": [...]
}}"""

CONSOLIDATION_PROMPT = """Review this Story Bible for contradictions, duplicate characters, or inconsistencies.

STORY BIBLE:
{bible_context}

Check for:
1. Characters that might be the same person listed separately
2. Relationship states that contradict each other
3. Role classifications that don't match the character's actions
4. Arc statuses that are outdated

Return JSON:
{{
  "corrections": [
    {{
      "character": "name",
      "issue": "what's wrong",
      "role": "corrected role (if applicable)",
      "arc_status": "corrected status (if applicable)"
    }}
  ],
  "merge_suggestions": [
    {{
      "names": ["name1", "name2"],
      "reason": "why these might be the same character"
    }}
  ]
}}

If everything looks consistent, return {{"corrections": [], "merge_suggestions": []}}"""


# ── Archetype Assignment Prompts ──────────────────────────────────────────────

ARCHETYPE_SYSTEM = """You are a Chinese 视频解说 (Douyin novel narration) expert.
You assign archetype nicknames to characters for entertaining narration.
You understand Chinese internet slang and 视频解说 conventions deeply.
Always respond in valid JSON format."""

ARCHETYPE_PROMPT = """Assign 视频解说 archetype nicknames to these characters.

CHARACTER PROFILES (from Story Bible):
{character_profiles}

AVAILABLE ARCHETYPES:
{archetype_menu}

RULES:
1. The male protagonist MUST be 小帅
2. The main female love interest MUST be 小美
3. 渣男 is ONLY for romantic scumbags (cheaters, manipulative in love) — NEVER for fathers, uncles, or elders
4. 渣女 is ONLY for female romantic scumbags — NEVER for mothers or aunts
5. 黄毛 is for street-level bullies/thugs, NOT for sophisticated villains
6. Minor characters (freq < 5) should get descriptive roles like "旁边的服务员" instead of named archetypes
7. Each named archetype should only be assigned to ONE character (except 小弟 which can be reused)
8. Gender must match the archetype's gender requirement

For each character, provide:
{{
  "assignments": [
    {{
      "original_name": "沈辰",
      "archetype": "小帅",
      "reasoning": "male protagonist, reborn powerful heir"
    }}
  ]
}}"""


# ── Narrator v2 Prompts ──────────────────────────────────────────────────────

NARRATOR_V2_SYSTEM = """You are a Chinese 小说推文 narrator (视频解说).
You retell novels like you're telling a friend an amazing story you just read.
Keep it SIMPLE — a middle schooler should understand every sentence.
Use archetype nicknames, never real character names.
Write in natural spoken Chinese (口语化), third person only.
NEVER say 咱们. NEVER add 想知道xxx or 点赞关注.
Always respond in Chinese."""

NARRATOR_V2_PROMPT = """Write a 视频解说 narration script for these chapters.

CHARACTER SHEET (use ONLY these archetype names, NEVER the original names):
{character_sheet}

STORY CONTEXT:
{story_context}

{bridge_instructions}

ORIGINAL CHAPTERS (for reference):
---
{chapters_text}
---

STORYTELLING RULES:

SIMPLICITY:
1. Write like you're explaining to a friend — SHORT sentences, SIMPLE words
2. When a character first appears: ONE sentence saying who they are and why they matter
3. Before any fight: explain WHY in one sentence
4. Use ONLY the archetype nicknames, NEVER original names

STRUCTURE:
5. Start with context → introduce characters → build tension → conflict → payoff
6. DON'T rush into action — spend the first few scenes setting up WHO the character is, WHAT their situation is, and WHY we should care
7. Weave dialogue into narration, no quotation marks
8. Transitions: 没想到、就在这时、然而、谁知、下一秒
9. CRITICAL: Write ONLY the narration text and [画面] tags. Do NOT output section headers like **设定**, **主角出场**, **矛盾铺垫**, **爽点**, **结局**, **总结** etc. Do NOT output planning notes, markdown headers (###), bullet points, or meta-commentary. Just pure narration.
8. ---SCENE--- break between visual moments, [画面：description] after each

MAKE IT 爽 (SATISFYING):
9. 先抑后扬 — let the villain be arrogant FIRST, then destroy them
10. Face-slap formula: 对方嘲笑 → 主角淡定 → 一招制敌 → 所有人惊呆了
11. Show the CONTRAST — they looked down on him, now they're begging
12. When the MC wins, describe EVERYONE'S reaction, especially the villain's face

FLOW:
13. Third person ONLY, never 咱们
14. End naturally, NO cliffhanger, NO 想知道, NO 点赞关注
15. Target: {target_length} — MUST have at least {target_scenes} scenes, this is IMPORTANT
16. Each scene = 2-4 sentences, every scene moves the story forward
17. Cover ALL major events from the chapters — don't skip important plot points

VISUAL NOTES [画面：]:
- SIMPLE, CONCRETE scenes an artist can draw
- Describe: who, what they're doing, expression, setting, lighting
- NO abstract concepts

Write the complete narration script now. Chinese only."""

BRIDGE_FIRST_VIDEO = """This is the FIRST episode. Set up everything from scratch.
The opening must establish the world, protagonist, and central conflict within the first 3 sentences.
Start naturally — NO "大家好" or "今天给大家讲" openings."""

BRIDGE_CONTINUATION = """PREVIOUS EPISODE CONTEXT:
- The audience has already been told: {audience_knows}
- The audience does NOT know yet: {audience_does_not_know}
- Previous episode ended at: "{cliffhanger}"

CONTINUATION RULES:
1. Continue the story naturally from where the previous episode left off
2. Do NOT re-explain what the audience already knows
3. DO explain anything new
4. NO "上回说到" openings — just continue the story smoothly as if it's one continuous narration
5. NO cliffhangers or hooks at the end
"""

AUDIENCE_EXTRACTION_PROMPT = """Given this narration script, extract what the audience now knows.

SCRIPT:
{script}

STORY BIBLE CONTEXT:
{bible_context}

Return JSON:
{{
  "audience_knows": ["fact 1 the audience was told", "fact 2"],
  "audience_does_not_know": ["secret 1 not yet revealed", "secret 2"],
  "cliffhanger": "the cliffhanger at the end of this video",
  "last_scene_summary": "brief description of the final scene"
}}"""
