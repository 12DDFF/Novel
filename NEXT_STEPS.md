# Next Steps: Image Generation Pipeline

## What's Built (Narration Pipeline v2)

The narration system is complete and working:

```
Scrape (Fanqie/Hetushu) → Regex Harvester → Story Bible (LLM) → Archetype Assignment → Narration Script → Dedup + Clean
```

### Modules
- `src/narration/harvester.py` — Regex character extraction (no LLM, instant)
- `src/narration/bible.py` — Incremental Story Bible with batch processing
- `src/narration/archetype.py` — 26 archetypes with semantic validation (gender, forbidden roles)
- `src/narration/narrator_v2.py` — Script generator with audience state tracking, dedup, name leak prevention
- `src/narration/orchestrator.py` — End-to-end pipeline orchestrator
- `src/narration/prompts.py` — All LLM prompt templates
- `src/scraper/hetushu.py` — Hetushu scraper (fully free, no paywall)

### Test Results
- 276 unit tests passing
- Tested on 末世降临先杀圣母 (574 chapters) — 108 characters, 154 world facts, 12 episodes
- Tested on 庆余年 (747 chapters) — 60 characters, 145 world facts, 15 episodes
- Scrambled text from hetushu works fine (LLM understands shuffled paragraphs)

### Known Issues to Fix
- Some episodes leak **structure labels** (设定, 主角出场, 爽点) into narration text — post-processing partially handles this but prompt needs more tuning
- Script length varies — some episodes are too short (~800 chars), others too long with repetition
- `[画面：...]` descriptions are rough Chinese, NOT production-ready image prompts

---

## What to Build Next: Image Generation Pipeline

### Overview

Take the narration script and generate consistent, high-quality images for every sentence.

```
Narration Script (per episode, ~30-50 sentences)
    ↓
[1. Sentence Splitter]
    ↓
[2. Scene Context Analyzer]  ← sliding window context
    ↓
[3. Visual Character Sheet]  ← master visual descriptions
    ↓
[4. Image Prompt Generator]  ← English Flux prompts
    ↓
[5. Reference Selector]      ← pick master images per scene
    ↓
[6. Flux Kontext Pro]        ← generate final images
```

### Module 1: Sentence Splitter (`src/image_gen/sentence_splitter.py`)

Split narration into individual image beats. Each beat = one image.

**Input**: Episode narration text (string)
**Output**: List of sentence strings

Logic:
- Split on `---SCENE---` markers first (major scene breaks)
- Within each scene, split on sentence boundaries (。！？)
- Merge very short sentences (< 10 chars) with adjacent ones
- Target: ~30-40 beats per 5-minute episode

### Module 2: Scene Context Analyzer (`src/image_gen/scene_analyzer.py`)

The brain of the image pipeline. Uses **sliding window context** (5 lines backward + 3 lines forward) to analyze each sentence.

**Input**: 
- Current sentence
- Backward context (5 previous sentences, can cross episode boundary)
- Forward context (3 next sentences)
- Story Bible (character info)
- Archetype map

**Output per sentence** (JSON):
```json
{
  "sentence_index": 5,
  "sentence": "他一斧头劈下去，对方的脑袋瞬间被劈成两半",
  "characters_present": ["顾杀", "齐云"],
  "characters_archetype": ["小帅", "心机男"],
  "location": "废弃教学楼天台",
  "location_changed": false,
  "mood": "action",
  "camera_suggestion": "close_up",
  "creatures_present": [],
  "key_action": "顾杀用斧头劈开齐云的头",
  "background_description": "破败的学校天台，灰色水泥地面，远处可见丧尸横行的城市废墟"
}
```

**Key design: Sliding window context**
```
Processing sentence 5:
  backward: [sent 0, 1, 2, 3, 4]  ← who was just mentioned, where are we
  current:  [sent 5]               ← what to analyze
  forward:  [sent 6, 7, 8]        ← what's coming (helps with transitions)

Episode boundary handling:
  Episode 2, sentence 0:
  backward: [ep1_sent_N-5, ..., ep1_sent_N]  ← carry over from previous episode
  current:  [ep2_sent_0]
  forward:  [ep2_sent_1, 2, 3]
```

**Why sliding window instead of full episode**:
- No cold start at episode boundaries
- Scales to any episode length
- Keeps LLM context small and focused
- Can process in parallel (each sentence is independent once context is set)

**Implementation**: One LLM call per ~5-10 sentences (batch them). Use deepseek-chat for speed.

### Module 3: Visual Character Sheet (`src/image_gen/visual_sheet.py`)

Maintains visual descriptions for ALL entities (characters + creatures + locations).

**Input**: Story Bible + archetype map
**Output**: Visual description registry

```python
class VisualEntity:
    name: str                    # "顾杀" or "丧尸"
    archetype: str               # "小帅" or "zombie"
    entity_type: str             # "character", "creature", "vehicle"
    visual_description_en: str   # English description for Flux
    reference_image_path: str    # Path to master reference image
    
class VisualSheet:
    entities: dict[str, VisualEntity]
    
    def get_or_create_reference(self, name) -> str:
        """Get existing reference image or generate new one via Flux Dev"""
```

**Character visual descriptions** (from Bible → English):
```
顾杀 → "young Chinese man, 18 years old, short black hair, cold sharp eyes, 
         athletic build, wearing torn school uniform, carrying a fire axe"
```

**Creature visual descriptions** (from Bible world facts → English):
```
丧尸 → "humanoid zombie with grey rotting skin, glowing red eyes, torn clothing,
         blood stains, hunched posture, exposed teeth"
火猫王 → "massive fire cat beast, 3 meters tall, orange fur with flame patterns,
           burning mane, feral eyes, sharp claws"
```

**Master reference generation**:
- Characters: Flux Dev text-to-image → 1024x1024 portrait
- Creatures: Flux Dev text-to-image → 1024x1024 creature design
- Store in `shared_characters/` and `shared_creatures/` directories
- Generate ONCE, reuse for all scenes

### Module 4: Image Prompt Generator (`src/image_gen/prompt_generator.py`)

Converts each analyzed sentence into a production Flux prompt.

**Input**: Scene analysis (from Module 2) + Visual sheet (from Module 3)
**Output**: English image prompt (30-75 words)

**Prompt structure**:
```
[art style], [camera angle], [subject doing action], [setting with background details], 
[lighting based on mood], [atmosphere], natural hand proportions, correct human anatomy
```

**Rules**:
- If character has reference image: start with "the same character"
- Match lighting to mood (use existing mood-to-lighting mapping from segmenter/prompts.py)
- If `location_changed == false`: reuse previous background description for consistency
- If creature present: describe creature in scene
- Camera angle from scene analysis suggestion
- 30-75 words max
- End with: "natural hand proportions, correct human anatomy"

**Example output**:
```
anime illustration, close-up low angle shot, the same character swinging a fire axe 
downward with fierce expression, blood splatter in motion, abandoned school rooftop 
with cracked concrete, distant burning cityscape, harsh directional lighting from 
above casting sharp shadows, intense action atmosphere, natural hand proportions, 
correct human anatomy
```

### Module 5: Reference Selector (`src/image_gen/ref_selector.py`)

Decides which master reference image(s) to feed to Flux Kontext Pro.

**Logic**:
```python
def select_reference(scene_analysis, visual_sheet):
    characters = scene_analysis["characters_present"]
    creatures = scene_analysis["creatures_present"]
    
    if len(characters) == 1:
        return visual_sheet.get_reference(characters[0])  # single ref
    elif len(characters) > 1:
        refs = [visual_sheet.get_reference(c) for c in characters[:3]]
        return stitch_references(refs)  # side-by-side composite
    elif creatures:
        return visual_sheet.get_reference(creatures[0])  # creature ref
    else:
        return None  # no reference → text-to-image fallback
```

### Module 6: Image Generation (existing `src/image_gen/replicate_backend.py`)

Already built. Just needs to be called with the prompt + reference from modules 4-5.

---

## Implementation Order

1. **Sentence Splitter** — simple, no LLM, foundation for everything
2. **Visual Character Sheet** — extends Bible with visual descriptions, generates master images
3. **Scene Context Analyzer** — the brain, sliding window context, LLM-powered
4. **Image Prompt Generator** — converts analysis to Flux prompts
5. **Reference Selector** — picks which master image to use
6. **Integration** — wire it all together into an orchestrator

## Existing Code to Reuse

- `src/image_gen/replicate_backend.py` — Flux Kontext Pro backend (working)
- `src/image_gen/base.py` — ImageResult dataclass
- `src/segmenter/prompts.py` lines 81-90 — mood-to-lighting mapping
- `src/narration/bible.py` — StoryBible with character descriptions
- `src/narration/archetype.py` — archetype map with gender/role info
- `src/core/llm_client.py` — LLM client with model override

## Key Design Decisions

- **Sliding window context** (5 back + 3 forward) instead of full episode — handles episode boundaries, scales, focused
- **Visual descriptions separate from story descriptions** — Bible tracks plot, Visual Sheet tracks appearance
- **Master references for creatures too** — not just human characters
- **One LLM call per 5-10 sentences** for scene analysis — batched for speed
- **Location continuity tracking** — `location_changed` flag for background consistency
