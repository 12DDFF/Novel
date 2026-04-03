"""
Novel chapter rewriter: rewrites original text into video-optimized narration.

Takes the original chapter and produces a "screenplay-style" version that:
- Preserves the story, tone, and vibe
- Uses original words (no direct copying)
- Adds explicit speaker tags for dialogue: [角色名]: "对话"
- Adds scene break markers: ---SCENE---
- Embeds visual descriptions for key moments
- Simplifies overly complex/abstract passages
- Keeps it engaging and dramatic
"""

from __future__ import annotations

import logging

from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """You are a professional Chinese screenwriter and narrator.
You rewrite novel chapters into video narration scripts.
Your output will be read aloud by a TTS narrator with multiple character voices.
Always respond in Chinese. Keep the author's tone and vibe."""

REWRITE_PROMPT = """Rewrite this novel chapter into a video narration script.

RULES:
1. RETELL the story in your own words — do NOT copy sentences directly
2. Keep the same tone, mood, and dramatic feeling as the original
3. Keep all dialogue but rewrite it naturally — mark speakers explicitly like this:
   [沈辰]: "对话内容"
   [蓝溪]: "对话内容"
4. Add scene breaks with ---SCENE--- where the visual should change (new location, time skip, new character enters, major action)
5. After each ---SCENE--- add a brief visual note in brackets like:
   [画面：沈辰站在酒店落地窗前，俯瞰城市夜景，手持红酒杯]
6. Keep narration concise — 2-4 sentences of narration between dialogue
7. Add brief character appearance descriptions the FIRST time each character appears, like:
   [外貌：沈辰，二十岁左右，身材瘦高，面如白玉，黑色眼瞳，气质矜贵]
8. Make it dramatic and engaging — this is entertainment content
9. Total length should be similar to the original (don't cut too much)
10. Every piece of dialogue MUST have a speaker tag [名字]:
11. STAY FAITHFUL to the original story — do NOT add sci-fi elements, mechanical body parts, cybernetic enhancements, or technology that isn't in the original text
12. If the original is a fantasy/supernatural story, keep it fantasy. Do NOT turn it into sci-fi.
13. Keep character descriptions grounded — describe clothing, face, hair, expression. Not robots or cyborgs unless the original says so.

CHARACTER PROFILES (use these descriptions consistently):
{character_profiles}

ORIGINAL CHAPTER TEXT:
---
{chapter_text}
---

Output the rewritten narration script. Chinese only. No English. No explanations."""

CHARACTERS_FOR_REWRITE_PROMPT = """Analyze this novel chapter and list ALL characters that appear.
For each character, provide their name and a VISUAL description suitable for AI image generation.

If the text describes their appearance, use that. If not, INVENT an appearance that fits their personality and role.
Make each character visually DISTINCT from the others (different hair, clothing, build, etc).

IMPORTANT: Descriptions must be specific and visual — things an artist can draw.
Bad: "她很漂亮" (too vague)
Good: "长发及腰，淡蓝色眼瞳，肌肤白皙，穿着黑色连衣裙，气质冷艳"

Chapter text:
---
{chapter_text}
---

Return a JSON array:
[
  {{
    "name": "沈辰",
    "gender": "male",
    "role": "protagonist",
    "appearance": "二十岁，身材瘦高，面如白玉，黑曜石般的眼瞳，气质矜贵，穿着深色西装",
    "appearance_en": "20 years old, tall and slim, pale jade-like complexion, obsidian black eyes, noble aura, wearing dark suit"
  }}
]

Return ONLY the JSON array."""


class ChapterRewriter:
    """Rewrites novel chapters into video-optimized narration scripts."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract_characters_visual(self, chapter_text: str) -> list[dict]:
        """
        Extract characters with visual descriptions from a chapter.
        Invents appearances for characters without descriptions.
        """
        prompt = CHARACTERS_FOR_REWRITE_PROMPT.format(chapter_text=chapter_text)
        result = self.llm.chat_json(prompt=prompt, system=REWRITE_SYSTEM, temperature=0.4)

        if not isinstance(result, list):
            raise ValueError(f"Expected list, got {type(result)}")

        # Validate each character has required fields
        validated = []
        for char in result:
            if char.get("name"):
                validated.append({
                    "name": char.get("name", ""),
                    "gender": char.get("gender", "unknown"),
                    "role": char.get("role", "minor"),
                    "appearance": char.get("appearance", "未描述"),
                    "appearance_en": char.get("appearance_en", "not described"),
                })
        return validated

    def rewrite_chapter(
        self,
        chapter_text: str,
        character_profiles: list[dict],
    ) -> str:
        """
        Rewrite a chapter into video narration script.

        Returns the rewritten text with speaker tags and scene breaks.
        """
        # Format character profiles for the prompt
        profiles_str = ""
        for char in character_profiles:
            profiles_str += f"- {char['name']} ({char['role']}): {char['appearance']}\n"

        prompt = REWRITE_PROMPT.format(
            chapter_text=chapter_text,
            character_profiles=profiles_str,
        )

        rewritten = self.llm.chat(
            prompt=prompt,
            system=REWRITE_SYSTEM,
            temperature=0.6,
            max_tokens=8192,
        )

        return rewritten.strip()

    def process_chapter(self, chapter_text: str) -> tuple[list[dict], str]:
        """
        Full pipeline: extract characters then rewrite.
        Returns (characters, rewritten_text).
        """
        logger.info("Extracting characters with visual descriptions...")
        characters = self.extract_characters_visual(chapter_text)
        logger.info(f"Found {len(characters)} characters")

        logger.info("Rewriting chapter...")
        rewritten = self.rewrite_chapter(chapter_text, characters)
        logger.info(f"Rewritten: {len(rewritten)} chars")

        return characters, rewritten


def parse_rewritten_script(text: str) -> list[dict]:
    """
    Parse a rewritten script into structured scenes.

    Returns list of scenes, each with:
    - visual: the [画面：...] description
    - segments: list of {type: "narration"|"dialogue", speaker: str|None, text: str}
    """
    import re

    scenes = []
    raw_scenes = re.split(r'---SCENE---', text)

    for raw in raw_scenes:
        raw = raw.strip()
        if not raw:
            continue

        scene = {"visual": "", "appearance_notes": [], "segments": []}

        # Extract visual description
        visual_match = re.search(r'\[画面[：:](.+?)\]', raw)
        if visual_match:
            scene["visual"] = visual_match.group(1).strip()

        # Extract appearance notes
        for app_match in re.finditer(r'\[外貌[：:](.+?)\]', raw):
            scene["appearance_notes"].append(app_match.group(1).strip())

        # Remove bracket annotations for text processing
        clean = re.sub(r'\[(?:画面|外貌)[：:].+?\]', '', raw).strip()

        # Parse dialogue and narration
        lines = clean.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for speaker tag: [角色名]: "对话" or [角色名]："对话"
            dialogue_match = re.match(
                r'\[(.+?)\][：:]\s*["""「](.+?)["""」]',
                line,
            )
            if dialogue_match:
                scene["segments"].append({
                    "type": "dialogue",
                    "speaker": dialogue_match.group(1),
                    "text": dialogue_match.group(2),
                })
            else:
                # It's narration
                # Clean any remaining brackets
                narration = re.sub(r'\[.+?\]', '', line).strip()
                if narration:
                    scene["segments"].append({
                        "type": "narration",
                        "speaker": None,
                        "text": narration,
                    })

        if scene["segments"] or scene["visual"]:
            scenes.append(scene)

    return scenes
