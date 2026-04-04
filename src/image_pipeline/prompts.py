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

IMAGE_PROMPT_SYSTEM = """You are a professional AI image prompt engineer.
You convert scene descriptions into precise English prompts for Flux image generation.
Your prompts produce consistent, high-quality anime illustrations.
Always respond in valid JSON format."""

IMAGE_PROMPT_TEMPLATE = """Convert these scene analyses into English image generation prompts.

VISUAL CHARACTER DESCRIPTIONS:
{character_visuals}

SCENES TO CONVERT:
{scenes}

LIGHTING GUIDE BY MOOD:
- tense: dramatic side lighting with sharp shadows, cool blue tones
- action: bright directional light with rim lighting, high contrast
- joyful: bright warm sunlight, golden hour, vibrant colors
- melancholy: overcast diffused light, muted cool tones, rain or fog
- romantic: soft warm backlight with rim glow, bokeh, gentle atmosphere
- dramatic: strong volumetric light with god rays, storm clouds
- mysterious: dim ambient light, single light source, fog and silhouettes
- peaceful: soft morning light, warm pastels, gentle breeze
- horror: harsh underlighting, deep shadows, desaturated cold blue
- humorous: bright even lighting, warm tones, exaggerated expressions

RULES:
1. Start with "anime illustration"
2. Include camera angle from scene analysis
3. Describe character(s) using their VISUAL DESCRIPTION (not story info)
4. If character has a reference image, start character description with "the same character"
5. Include setting/background details
6. Match lighting to mood using the guide above
7. 30-75 words maximum per prompt
8. End with "natural hand proportions, correct human anatomy"
9. If location hasn't changed from previous scene, keep background description consistent

Return JSON array:
[
  {{
    "sentence_index": 0,
    "image_prompt": "anime illustration, medium shot, the same character..."
  }}
]"""
