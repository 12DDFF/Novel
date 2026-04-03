"""
视频解说 (Video Narration) script generator.

Takes raw novel chapters and produces a dramatic, entertaining narration script
in the style of Chinese Douyin/Kuaishou novel narration videos.

Key features:
- Replaces character names with archetypes (小帅/小美/黄毛)
- Opens with hook/conflict, not exposition
- Compresses multiple chapters into one cohesive arc
- Uses 爽文 narration patterns and transition phrases
- Ends with cliffhanger
- Designed for TTS + image generation pipeline
"""

from __future__ import annotations

import json
import logging
import re

from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

NARRATOR_SYSTEM = """You are a professional Chinese 小说推文 narrator (视频解说).
You retell novel stories in the viral Douyin narration style.
Your scripts are dramatic, entertaining, and make viewers want to watch more.
You always use archetype nicknames instead of real character names.
You write in conversational Chinese (口语化), like a storyteller at a bar.
Always respond in Chinese."""

# ── Step 1: Analyze chapters and plan the arc ────────────────────────────────

ARC_PLANNING_PROMPT = """I'm giving you {num_chapters} chapters from a Chinese web novel. Analyze them and create a narration plan.

NOVEL CHAPTERS:
---
{chapters_text}
---

Do the following:
1. List ALL characters and assign them archetype nicknames:
   - 小帅 = male protagonist
   - 小美 = main female love interest
   - 黄毛 = main bully/antagonist
   - 老大 = powerful boss figure
   - 老头/老爷子 = elder/grandfather/master
   - 渣男 = cheating ex / scumbag rival
   - 渣女 = scheming woman / betrayer
   - 富婆 = rich older woman
   - 小弟 = henchman/follower (generic, no name needed)
   - For other characters, describe by role: "旁边的服务员", "围观群众" etc.

2. Identify the main 爽点 (satisfying moments) in these chapters - these are the MUST-INCLUDE moments

3. Identify parts to SKIP (training montages, internal monologue, worldbuilding, boring setup)

4. Plan the arc: opening hook → conflict → 爽点 chain → cliffhanger ending

Return as JSON:
{{
  "characters": [
    {{"original_name": "沈辰", "nickname": "小帅", "one_line": "重生的豪门大少"}},
    {{"original_name": "蓝溪", "nickname": "小美", "one_line": "贴身女保镖，绝世美女"}}
  ],
  "key_moments": [
    "小帅站在酒店顶楼，得知末日即将降临",
    "小帅和老爷子通话，决定提前布局"
  ],
  "skip": ["世界观详细设定", "配角的内心独白"],
  "hook": "在末日降临的前一天晚上，一个重生的豪门大少站在城市最高处，嘴角勾起一抹冷笑",
  "cliffhanger": "小帅不知道的是，另一个重生者也在暗中注视着他"
}}"""

# ── Step 2: Generate the narration script ────────────────────────────────────

NARRATION_PROMPT = """Based on this plan, write a 视频解说 narration script.

ARC PLAN:
{arc_plan}

ORIGINAL CHAPTERS (for reference):
---
{chapters_text}
---

STORYTELLING RULES (MOST IMPORTANT — FOLLOW THESE FIRST):
1. ASSUME THE AUDIENCE KNOWS NOTHING. Explain everything from scratch.
2. When a character FIRST appears, give a clear 1-sentence introduction:
   BAD: 就在这时黄毛出现了 (who? why? from where?)
   GOOD: 小帅还有个天大的麻烦——小美她爸欠了一个叫黄毛的富二代一大笔钱，黄毛仗着有钱有势，一直想霸占小美
3. Before ANY conflict, explain the RELATIONSHIP and MOTIVATION:
   - WHO is this person and what's their connection to the protagonist?
   - WHY are they angry/scared/attacking?
   - HOW did this situation come about?
4. Follow a CLEAR, LOGICAL order. No jumping around:
   设定介绍(1-2句) → 主角出场 → 主角的处境 → 问题/敌人出现(先介绍再冲突) → 爽点 → 新问题 → 悬念
5. The FIRST scene must set up the whole situation so anyone can understand:
   "话说有这么一个小帅，三年前被人陷害变成了植物人。他有个老婆叫小美，长得天仙似的，这三年不离不弃照顾他。但小美她爸是个赌鬼，欠了一屁股债，现在债主找上门了。"

FORMAT RULES:
6. Use ONLY archetype nicknames (小帅/小美/黄毛). NEVER use original names.
7. Open with a hook — but make sure the audience understands the setup within the first 3 sentences.
8. NO quotation marks — weave dialogue into narration:
   GOOD: 小帅直接问他你以为你是谁
9. Use transitions: 没想到、就在这时、然而、谁知、话音刚落、下一秒
10. Every 3-5 sentences must have a 爽点 or 转折
11. Add ---SCENE--- breaks between visual moments
12. After each ---SCENE--- add [画面：concrete visual description for the artist]
13. Conversational Chinese (口语化), third person
14. End with cliffhanger + 想知道后面发生了什么？
15. Target: {target_length} — about {target_scenes} scenes
16. Each scene = 3-5 sentences
17. 爽点 phrases: 直接吓傻了、所有人都惊呆了、淡淡一笑

VISUAL NOTES [画面：]:
- SIMPLE, CONCRETE scenes an artist can draw
- Describe: who is there, what they're doing, their expression, the setting
- Include lighting: 夜晚霓虹灯下、阳光下、昏暗房间里
- NO abstract concepts, NO complicated action choreography

Write the complete narration script now. Chinese only."""

# ── Step 3: Generate image prompts from scenes ──────────────────────────────

IMAGE_PROMPT_GENERATION = """Convert these scene visual notes into professional AI image generation prompts.

CHARACTER VISUAL DESCRIPTIONS:
{character_visuals}

SCENES:
{scenes}

For each scene, generate an English image prompt following these rules:
- Natural English sentences, NOT tag lists
- Structure: [style] + [camera angle] + [subject doing action] + [setting] + [lighting] + [atmosphere]
- 30-75 words maximum
- Use specific camera angles: wide shot, medium shot, close-up, low angle
- Be specific about lighting direction
- End every prompt with: "natural hand proportions, correct human anatomy"
- For scenes with the main character, start with: "the same character" (for Kontext consistency)
- Art style: anime illustration with detailed background art

Return as JSON array:
[
  {{
    "scene_index": 0,
    "visual_note": "the original Chinese visual note",
    "image_prompt": "the English prompt for Flux",
    "characters_in_scene": ["小帅"],
    "mood": "dramatic"
  }}
]"""


class VideoNarrator:
    """Generates 视频解说 style narration scripts from novel chapters."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_script(
        self,
        chapters_text: list[str],
        target_minutes: float = 3.0,
        target_scenes: int = 15,
    ) -> dict:
        """
        Full pipeline: chapters → arc plan → narration script → scenes with image prompts.

        Args:
            chapters_text: List of chapter texts (raw Chinese).
            target_minutes: Target video duration in minutes.
            target_scenes: Target number of visual scenes.

        Returns:
            Dict with keys: arc_plan, script, scenes, character_map
        """
        combined_text = "\n\n---章节分割---\n\n".join(chapters_text)

        # Truncate if too long for LLM context
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "\n\n[后续章节省略...]"

        # Step 1: Plan the arc
        logger.info("Planning arc...")
        arc_plan = self._plan_arc(combined_text, len(chapters_text))

        # Step 2: Generate narration
        logger.info("Generating narration script...")
        target_length = f"{int(target_minutes)}分钟左右（约{int(target_minutes * 250)}字）"
        script = self._generate_narration(arc_plan, combined_text, target_length, target_scenes)

        # Step 3: Parse scenes from script
        scenes = self._parse_scenes(script)

        # Step 4: Generate image prompts
        logger.info("Generating image prompts...")
        character_visuals = self._build_character_visuals(arc_plan)
        scenes_with_prompts = self._generate_image_prompts(scenes, character_visuals)

        return {
            "arc_plan": arc_plan,
            "script": script,
            "scenes": scenes_with_prompts,
            "character_map": {c["nickname"]: c for c in arc_plan.get("characters", [])},
        }

    def _plan_arc(self, chapters_text: str, num_chapters: int) -> dict:
        prompt = ARC_PLANNING_PROMPT.format(
            num_chapters=num_chapters,
            chapters_text=chapters_text,
        )
        return self.llm.chat_json(prompt=prompt, system=NARRATOR_SYSTEM, temperature=0.4)

    def _generate_narration(self, arc_plan: dict, chapters_text: str, target_length: str, target_scenes: int) -> str:
        prompt = NARRATION_PROMPT.format(
            arc_plan=json.dumps(arc_plan, ensure_ascii=False, indent=2),
            chapters_text=chapters_text,
            target_length=target_length,
            target_scenes=target_scenes,
        )
        return self.llm.chat(prompt=prompt, system=NARRATOR_SYSTEM, temperature=0.6, max_tokens=8192)

    def _generate_image_prompts(self, scenes: list[dict], character_visuals: str) -> list[dict]:
        scenes_text = json.dumps(scenes, ensure_ascii=False, indent=2)
        prompt = IMAGE_PROMPT_GENERATION.format(
            character_visuals=character_visuals,
            scenes=scenes_text,
        )
        result = self.llm.chat_json(prompt=prompt, system=NARRATOR_SYSTEM, temperature=0.3, max_tokens=8192)
        if isinstance(result, list):
            # Merge prompts into scenes
            for i, scene_prompt in enumerate(result):
                if i < len(scenes):
                    scenes[i]["image_prompt"] = scene_prompt.get("image_prompt", "")
                    scenes[i]["characters_in_scene"] = scene_prompt.get("characters_in_scene", [])
                    scenes[i]["mood"] = scene_prompt.get("mood", "dramatic")
        return scenes

    def _build_character_visuals(self, arc_plan: dict) -> str:
        """Build character visual descriptions for image prompt generation."""
        lines = []
        for char in arc_plan.get("characters", []):
            nickname = char.get("nickname", "")
            one_line = char.get("one_line", "")
            visual = char.get("visual_description", one_line)
            lines.append(f"- {nickname}: {visual}")
        return "\n".join(lines) if lines else "No character descriptions available."

    @staticmethod
    def _parse_scenes(script: str) -> list[dict]:
        """Parse ---SCENE--- markers and [画面：] notes from the narration script."""
        scenes = []
        # Split on ---SCENE---, ---, or any line that's just dashes
        raw_scenes = re.split(r'\n-{3,}(?:SCENE)?-{0,}\n', script)

        for i, raw in enumerate(raw_scenes):
            raw = raw.strip()
            if not raw:
                continue

            # Extract visual note
            visual = ""
            visual_match = re.search(r'\[画面[：:](.+?)\]', raw)
            if visual_match:
                visual = visual_match.group(1).strip()

            # Get narration text (remove visual notes)
            narration = re.sub(r'\[画面[：:].+?\]', '', raw).strip()
            # Remove any remaining bracket annotations
            narration = re.sub(r'\[.+?\]', '', narration).strip()

            if narration:
                scenes.append({
                    "index": i,
                    "narration": narration,
                    "visual_note": visual,
                    "image_prompt": "",  # filled in step 4
                    "characters_in_scene": [],
                    "mood": "dramatic",
                })

        return scenes
