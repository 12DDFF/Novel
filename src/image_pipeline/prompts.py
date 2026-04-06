"""LLM prompt templates for the image generation pipeline."""

SCENE_ANALYSIS_SYSTEM = """You are a visual scene analyzer for anime-style video production.
You analyze Chinese narration sentences and determine: who is present, where we are,
what mood to convey, and what camera angle to use.
You resolve pronouns (他/她/它) using context.
Always respond in valid JSON format."""

SCENE_ANALYSIS_PROMPT = """Analyze these narration sentences for image generation.

KNOWN CHARACTERS:
{character_list}

KNOWN CREATURES/ENTITIES:
{creature_list}

BACKWARD CONTEXT (previous sentences for reference):
{backward_context}

SENTENCES TO ANALYZE:
{sentences}

FORWARD CONTEXT (upcoming sentences for reference):
{forward_context}

For EACH sentence, determine:
1. characters_present: Which characters are in this scene? Resolve pronouns (他=who? 她=who?) using context. Use ORIGINAL Chinese names.
2. location: Where is this happening? Describe the location in Chinese.
3. location_changed: Did we move to a new location from the previous sentence? (true/false)
4. mood: One of: tense, action, joyful, melancholy, romantic, dramatic, mysterious, peaceful, horror, humorous
5. camera_suggestion: One of: wide_shot, medium_shot, close_up, low_angle, high_angle, over_shoulder
6. creatures_present: Any monsters/creatures in this scene? Use Chinese names.
7. key_action: What's happening, in one short Chinese phrase.
8. background_description: Describe the background/setting in Chinese (reuse from previous if location hasn't changed).

Return JSON array:
[
  {{
    "sentence_index": 0,
    "characters_present": ["顾杀"],
    "location": "教室",
    "location_changed": false,
    "mood": "tense",
    "camera_suggestion": "medium_shot",
    "creatures_present": [],
    "key_action": "顾杀命令同学堵住门窗",
    "background_description": "昏暗的教室，桌椅堆在门口，窗外一片混乱"
  }}
]"""

IMAGE_PROMPT_SYSTEM = """You are a top-tier AI image prompt engineer specializing in anime-style illustration for Flux image generation.

Your job: read a story scene and create a SINGLE image prompt that captures the most visually compelling moment.

You think like a cinematographer — you choose the camera angle, the lighting, the composition, and you describe EXACTLY what the viewer sees in the frame. Not vague feelings. Concrete visuals.

Always respond in valid JSON format."""

IMAGE_PROMPT_TEMPLATE = """I need you to generate image prompts for my image generator Flux.

Here is the FULL SCENE from the story:
---
{scene_text}
---

Here is the SPECIFIC LINE I want to visualize:
"{current_sentence}"

CHARACTERS IN THIS SCENE (use these visual descriptions):
{character_visuals}

SCENE CONTEXT:
- Location: {location}
- Mood: {mood}
- Previous scene background: {previous_background}
- Location changed from previous: {location_changed}

Now generate ONE image prompt following these rules:

FORMAT: "anime illustration, [camera angle], [what the viewer sees in the frame]"

WHAT MAKES A GOOD PROMPT:
- Describe the SCENE like a movie frame — background FIRST, then characters, then action
- Be SPECIFIC: "girl gripping doorframe with white knuckles, tears streaming down face" NOT "girl is scared"
- Include the ENVIRONMENT: walls, floor, lighting sources, objects, weather
- Describe CHARACTER POSES and EXPRESSIONS concretely
- Include lighting direction and color: "cold blue moonlight from the left", "warm overhead fluorescent"

HOW TO REFERENCE CHARACTERS:
- For characters with [REF]: describe them by their KEY VISUAL FEATURES (hair, clothing) so Flux can match them to the reference image
  Example: "the young man with short messy black hair in torn school uniform" (matches ref image 1)
  Example: "the girl with long straight black hair and pink hairpin" (matches ref image 2)
- For characters WITHOUT [REF]: include their full visual description
- NEVER just say "the same character" — always include identifying visual details
- When 2 characters are in the scene, describe BOTH with their distinguishing features

WHAT TO AVOID:
- NO gore, blood, death, killing — use "confronting", "clashing", "overpowering" instead
- NO vague descriptions like "dramatic moment" or "intense scene"
- NO emotions without physical description — show don't tell
- This is NOT a character portrait — there MUST be a background/environment

LENGTH: 40-80 words

End with: "natural hand proportions, correct human anatomy"

Return JSON:
{{
  "image_prompt": "anime illustration, ..."
}}"""
