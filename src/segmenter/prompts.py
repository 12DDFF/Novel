"""LLM prompt templates for scene segmentation with professional Flux prompt engineering."""

CHARACTER_EXTRACTION_SYSTEM = """You are a literary analyst specializing in Chinese fiction.
You extract character information from novel text with precision.
Always respond in valid JSON format."""

CHARACTER_EXTRACTION_PROMPT = """Analyze this Chinese novel chapter and extract ALL characters that appear or are mentioned.

For each character provide:
- name: Their primary name (Chinese)
- aliases: List of other names, nicknames, or titles used in the text
- description: Physical appearance details mentioned in the text (if none described, write "未描述")
- role: One of "protagonist", "antagonist", "supporting", "minor"

Chapter text:
---
{chapter_text}
---

Return a JSON array of characters. Example format:
[
  {{
    "name": "王林",
    "aliases": ["小林子", "林弟"],
    "description": "年轻男子，面容冷峻，穿着灰色长袍",
    "role": "protagonist"
  }}
]

Return ONLY the JSON array, no other text."""

SCENE_SEGMENTATION_SYSTEM = """You are an expert anime film director and AI image prompt engineer.
You break novel text into visual scenes and write professional image generation prompts.
You understand Flux model prompting: natural language, subject first, 30-75 words, specific lighting and camera angles.
Always respond in valid JSON format."""

SCENE_SEGMENTATION_PROMPT = """Break this Chinese novel chapter into visual scenes for video production.

RULES:
- Each scene = one distinct visual moment (2-4 sentences of narration)
- Create a new scene when: location changes, time skips, major action occurs, new character appears, emotional tone shifts, OR every 2-3 sentences
- Target: MINIMUM 10 scenes per chapter, ideally 12-18 scenes. More scenes = better video pacing
- Each scene should be 5-10 seconds of narration (50-100 Chinese characters)
- Every scene MUST have a clear visual that can be illustrated as a single image
- The narration_text should flow naturally when read aloud (adapt from source if needed)
- DO NOT combine multiple actions or dialogue exchanges into one scene — split them

CHARACTERS IN THIS CHAPTER:
{character_profiles}

ART STYLE: {art_style}

CHAPTER TEXT:
---
{chapter_text}
---

For each scene, provide:
- sequence: Scene number (starting from 1)
- narration_text: Chinese text to be narrated (2-6 sentences). IMPORTANT: preserve speaker tags like [沈辰]: "对话" for dialogue lines. The TTS system uses these tags to assign different voices.
- visual_description: What the camera sees (English)
- characters_present: List of character names visible
- mood: One of "tense", "action", "joyful", "melancholy", "romantic", "dramatic", "mysterious", "peaceful", "humorous", "horror"
- setting: Location and environment (English)
- image_prompt: A professional Flux image prompt following these rules:
  * Write in natural English sentences, NOT tag lists
  * Structure: [style] + [camera angle] + [subject doing action] + [setting foreground to background] + [lighting with direction] + [atmosphere]
  * 30-75 words maximum
  * Be specific about lighting direction: "warm golden light from the left", not just "warm lighting"
  * Use specific camera angles: wide shot, medium shot, close-up, low angle, over the shoulder, etc
  * Map mood to camera: tense=close-up/low angle, action=wide/dynamic, peaceful=wide/eye level, romantic=medium/soft focus
  * Include environment layers: foreground detail, middle ground, background/horizon
  * Do NOT use: masterpiece, best quality, 8k, highres, or (weight:1.5) syntax
  * Do NOT use tag-separated lists like "1girl, black_hair, sitting"
  * For Kontext scenes with character reference, start with "the same character" and describe what changes
  * Always end with: "natural hand proportions, correct human anatomy"
  * Do NOT describe mechanical body parts, cyborg elements, or technology unless the story explicitly mentions them
  * Keep characters looking fully human unless the story says otherwise
- transition: "cut", "crossfade", "fade_to_black", "fade_to_white"

MOOD-TO-LIGHTING GUIDE:
- tense: dramatic side lighting with sharp shadows, cool blue tones
- action: bright directional light with rim lighting, high contrast
- joyful: bright warm sunlight, golden hour, vibrant colors
- melancholy: overcast diffused light, muted cool tones, rain or fog
- romantic: soft warm backlight with rim glow, bokeh, gentle atmosphere
- dramatic: strong volumetric light with god rays, storm clouds
- mysterious: dim ambient light, single light source, fog and silhouettes
- peaceful: soft morning light, warm pastels, gentle breeze
- horror: harsh underlighting, deep shadows, desaturated cold blue

Example scene:
{{
  "sequence": 1,
  "narration_text": "沈辰站在七十八楼的落地窗前，俯瞰着江城繁华的夜景。红酒在月光下泛着血色。",
  "visual_description": "Young man in dark suit standing at floor-to-ceiling window overlooking city at night",
  "characters_present": ["沈辰"],
  "mood": "dramatic",
  "setting": "Luxury hotel penthouse, floor-to-ceiling windows, city skyline at night",
  "image_prompt": "anime illustration with detailed background art, medium shot from behind, a tall young Chinese man in a dark suit holding a wine glass standing at a floor-to-ceiling window, sprawling city skyline glittering with neon lights far below, cool blue moonlight from the right mixing with warm interior ambient light, dramatic cinematic atmosphere with reflections in the glass",
  "transition": "fade_to_black"
}}

Return ONLY the JSON array."""
