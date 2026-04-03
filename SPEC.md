# Novel-to-Video Pipeline - Technical Specification

## Project Overview

An automated pipeline that takes Chinese web novels (primarily from Fanqie/Qimao) and produces long-form AI-generated narrative videos with consistent character imagery, narration, subtitles, and background music.

**Proof of Concept Scope**: One chapter from Fanqie Novel -> segmented scenes -> consistent images -> animated video clips -> narrated, subtitled final video.

---

## Architecture Overview

```
[1. Scraper] -> [2. Scene Segmenter] -> [3. Image Generator] -> [4. Video Animator]
                                                                        |
                                                                        v
                  [7. Final Assembly] <- [6. Subtitle Generator] <- [5. TTS Narrator]
                        |
                        v
                  [8. Quality Review]
```

All modules communicate via a shared **project manifest** (JSON) that tracks every asset, its status, and metadata. Each module is an independent Python script/class that reads from and writes to this manifest.

---

## Part 1: Project Manifest & Data Model

### Purpose
Central state file that every module reads/writes. Tracks the entire pipeline state for a single video project.

### File: `manifest.json`

```json
{
  "project_id": "uuid",
  "source": {
    "platform": "fanqie",
    "novel_id": "7345678901234567",
    "novel_title": "...",
    "chapter_id": "7345678901234568",
    "chapter_title": "...",
    "raw_text_path": "data/raw/chapter_001.txt"
  },
  "settings": {
    "target_language": "zh",
    "narration_language": "zh",
    "image_style": "anime",
    "image_model": "flux-kontext",
    "video_model": "kling",
    "tts_model": "cosyvoice",
    "target_resolution": "1080x1920",
    "target_fps": 30,
    "bgm_mood": "dramatic"
  },
  "characters": [
    {
      "id": "char_001",
      "name": "李明",
      "aliases": ["小李", "明哥"],
      "description": "25-year-old male, short black hair, sharp jawline, dark eyes, wears a black leather jacket",
      "reference_image_path": "data/images/characters/char_001_ref.png",
      "image_prompt_prefix": "a 25-year-old Chinese man with short black hair, sharp jawline, dark eyes, wearing a black leather jacket"
    }
  ],
  "scenes": [
    {
      "id": "scene_001",
      "sequence": 1,
      "narration_text": "...",
      "visual_description": "...",
      "characters_present": ["char_001"],
      "mood": "tense",
      "setting": "dark alley at night, rain",
      "image_prompt": "...",
      "assets": {
        "image_path": null,
        "image_status": "pending",
        "video_path": null,
        "video_status": "pending",
        "audio_path": null,
        "audio_status": "pending",
        "subtitle_path": null,
        "subtitle_status": "pending"
      },
      "duration_estimate_seconds": 8.5,
      "quality_review": {
        "image_approved": false,
        "video_approved": false,
        "notes": ""
      }
    }
  ],
  "output": {
    "final_video_path": null,
    "total_duration_seconds": null,
    "status": "in_progress"
  }
}
```

### Implementation

```
src/
  models/
    manifest.py        # Pydantic models for all manifest types
    enums.py           # Status enums (pending, generating, complete, failed, approved)
  core/
    manifest_manager.py  # Read/write/update manifest with file locking
```

### Key Design Decisions
- File-based (JSON), not a database. Simple, inspectable, portable.
- File locking via `filelock` library to prevent corruption if modules run concurrently.
- Every module loads manifest -> does work -> updates manifest -> saves.
- All file paths in manifest are relative to project root.

### Dependencies
```
pydantic>=2.0
filelock>=3.12
```

---

## Part 2: Content Scraper

### Purpose
Download novel text from Fanqie Novel (primary) and Qimao (secondary). Handle font decryption, rate limiting, and output clean plaintext.

### Supported Platforms (PoC: Fanqie only)

#### Fanqie Novel (fanqienovel.com)
- **Method**: HTTP requests to web version, parse HTML
- **Anti-scraping**: Custom font encryption (static TTF mapping)
- **Decryption**: Download the custom font file from page CSS, parse the TTF glyph mappings using `fonttools`, build a character substitution table, apply to raw text
- **Rate limiting**: 1 request per 2 seconds, rotating User-Agent strings
- **Auth**: None required (all content is free)

### Module Interface

```python
class FanqieScraper:
    def search(self, query: str) -> list[NovelInfo]:
        """Search for novels by title/keyword."""

    def get_novel_info(self, novel_id: str) -> NovelInfo:
        """Get novel metadata: title, author, description, chapter list."""

    def get_chapter_list(self, novel_id: str) -> list[ChapterInfo]:
        """Get ordered list of all chapters with IDs and titles."""

    def download_chapter(self, novel_id: str, chapter_id: str) -> str:
        """Download and decrypt a single chapter. Returns clean plaintext."""

    def download_novel(self, novel_id: str, chapter_range: tuple[int, int] = None) -> list[str]:
        """Download multiple chapters. Optional range (start, end)."""
```

### Font Decryption Pipeline
1. Fetch chapter HTML from `fanqienovel.com/reader/{chapter_id}`
2. Extract custom font URL from embedded `<style>` or CSS `@font-face`
3. Download the `.woff`/`.woff2` font file
4. Parse font using `fonttools.TTFont` to extract `cmap` table
5. Build mapping: `{encrypted_codepoint: real_character}`
6. Apply mapping to raw scraped text
7. Cache the font mapping (it's static per font file, not per request)

### Output
- Clean UTF-8 plaintext saved to `data/raw/chapter_{n}.txt`
- Novel metadata saved to `data/raw/novel_info.json`
- Font mapping cache at `data/cache/font_map_{hash}.json`

### Directory Structure
```
src/
  scraper/
    __init__.py
    base.py              # Abstract BaseScraper class
    fanqie.py            # Fanqie implementation
    qimao.py             # Qimao implementation (future)
    font_decryptor.py    # Font file parsing and text decryption
    rate_limiter.py      # Request throttling
```

### Dependencies
```
httpx>=0.27
beautifulsoup4>=4.12
fonttools>=4.50
brotli>=1.1            # for woff2 decompression
lxml>=5.0
```

### Error Handling
- Retry failed requests 3x with exponential backoff
- If font decryption fails, save raw text + font file for manual inspection
- Log all requests with timestamps for debugging rate limit issues

---

## Part 3: Scene Segmenter (LLM-Powered)

### Purpose
Take raw chapter text and split it into discrete visual scenes. For each scene, extract characters, setting, mood, visual description, narration text, and generate an image prompt. This is the "brain" of the pipeline.

### Two-Pass Approach

#### Pass 1: Character Extraction
Before segmenting scenes, scan the full chapter to identify all characters and build character profiles.

```
Input:  Full chapter text
Output: List of Character objects with names, aliases, physical descriptions
```

**LLM Prompt Template (Character Extraction)**:
```
You are analyzing a Chinese novel chapter to extract all characters that appear.
For each character, provide:
1. Primary name
2. All aliases/nicknames used in the text
3. Physical description (infer from text or mark as "not described")
4. Role (protagonist, antagonist, side character)

Chapter text:
{chapter_text}

Return as JSON array.
```

#### Pass 2: Scene Segmentation
Split the chapter into scenes, each representing a distinct visual moment.

**Scene boundary triggers** (detected by LLM):
- Location/setting change
- Time skip
- New character enters/exits
- Major action or event
- Emotional tone shift
- Every 150-300 characters of continuous narration (ensures frequent image changes)

**LLM Prompt Template (Scene Segmentation)**:
```
You are a film director breaking a novel chapter into visual scenes for a video.

Rules:
- Each scene should be 2-6 sentences of narration
- Create a new scene whenever: location changes, time skips, a major action occurs,
  a new character appears, or the emotional tone shifts significantly
- Aim for one scene every 150-300 characters of source text
- Every scene MUST have a clear visual that can be illustrated

For each scene provide:
1. narration_text: The text to be narrated (can be adapted from source for flow)
2. visual_description: What the "camera" sees in this moment
3. characters_present: Which characters are visible
4. setting: Location and environment details
5. mood: Emotional tone (tense, joyful, melancholy, action, romantic, etc.)
6. image_prompt: A detailed prompt for an AI image generator. Must include:
   - Art style (consistent across all scenes)
   - Character descriptions (from character profiles)
   - Setting details
   - Lighting and atmosphere
   - Camera angle suggestion

Chapter text:
{chapter_text}

Character profiles:
{character_profiles}

Art style to use: {art_style}

Return as JSON array of scenes.
```

### Scene Density Control
- Config parameter: `min_scenes_per_1000_chars` (default: 4)
- Config parameter: `max_scene_narration_length` (default: 300 chars)
- If LLM produces too few scenes, re-prompt with stricter splitting instructions
- If too many, merge adjacent scenes with same setting/characters

### Image Prompt Construction
Each image prompt follows a template:

```
{art_style}, {camera_angle}, {setting_description}, {character_descriptions_with_actions},
{lighting}, {mood_atmosphere}, {color_palette}
```

Example:
```
cinematic anime style, medium shot, dark rainy alley with neon signs reflecting in puddles,
a 25-year-old Chinese man with short black hair and sharp jawline wearing a black leather
jacket standing with fists clenched facing a shadowy figure, dramatic side lighting with
blue and red neon glow, tense and confrontational atmosphere, dark blue and crimson color palette
```

### Module Interface

```python
class SceneSegmenter:
    def __init__(self, llm_client, art_style: str = "cinematic anime style"):
        ...

    def extract_characters(self, chapter_text: str) -> list[Character]:
        """Pass 1: Extract all characters from the chapter."""

    def segment_scenes(self, chapter_text: str, characters: list[Character]) -> list[Scene]:
        """Pass 2: Split chapter into visual scenes."""

    def refine_prompts(self, scenes: list[Scene], character_refs: dict[str, str]) -> list[Scene]:
        """Add character reference image context to prompts if available."""

    def estimate_duration(self, scene: Scene, tts_wpm: int = 250) -> float:
        """Estimate scene duration in seconds based on narration length."""
```

### Directory Structure
```
src/
  segmenter/
    __init__.py
    segmenter.py         # Main SceneSegmenter class
    prompts.py           # LLM prompt templates
    character_extractor.py  # Character extraction logic
    prompt_builder.py    # Image prompt construction
```

### Dependencies
```
anthropic>=0.40        # or openai>=1.50, for LLM API calls
tiktoken>=0.7          # token counting for context management
```

### Long Chapter Handling
- If chapter exceeds LLM context window, split into overlapping chunks (500 char overlap)
- Process chunks sequentially, passing previous chunk's last scene as context
- Character extraction always runs on full chapter first (summarize if needed)

---

## Part 4: Image Generator

### Purpose
Generate consistent, high-quality images for each scene. Maintain character appearance and art style across all images in a project.

### Supported Backends (PoC: Flux Kontext via API)

#### Option A: Flux Kontext (Recommended for PoC)
- **Why**: Best automation potential, open-weights available, strong character consistency
- **API**: Available via BFL API, Replicate, or self-hosted
- **Consistency method**: Pass character reference image as input context
- **Resolution**: 1024x1024 or 1024x1792 (portrait for vertical video)

#### Option B: Midjourney (Best aesthetics, harder to automate)
- **Consistency**: `--cref {url}` for character, `--sref {code}` for style
- **Automation**: No official API; requires Discord bot or third-party proxy
- **Best for**: Final production quality, manual/semi-automated workflows

#### Option C: ComfyUI + Flux + IP-Adapter (Fully local)
- **Consistency**: IP-Adapter face embedding + LoRA for style
- **Automation**: REST API via ComfyUI server
- **Best for**: High volume, no API costs, full control

### Consistency Strategy

#### Character Reference Workflow
1. **First scene with a character**: Generate a "character sheet" image
   - Front-facing, neutral pose, clear features, plain background
   - This becomes the reference image stored in manifest
2. **Subsequent scenes**: Always pass character reference image alongside scene prompt
3. **Style anchor**: Use a fixed style reference (SREF code or style image) for all generations

#### Character Sheet Generation
```
Prompt: "character design sheet, front view, {character_description},
         {art_style}, plain white background, full body, clear features,
         high detail, reference sheet"
```

### Module Interface

```python
class ImageGenerator:
    def __init__(self, backend: str = "flux-kontext", api_key: str = None):
        ...

    def generate_character_sheet(self, character: Character) -> str:
        """Generate reference image for a character. Returns image path."""

    def generate_scene_image(
        self,
        scene: Scene,
        character_refs: dict[str, str],  # char_id -> reference image path
        style_ref: str = None,           # style reference image path
        resolution: tuple[int, int] = (1024, 1792)
    ) -> str:
        """Generate scene image with character consistency. Returns image path."""

    def regenerate(self, scene: Scene, feedback: str = None) -> str:
        """Regenerate a scene image, optionally with adjustment feedback."""
```

### Generation Pipeline Per Scene
1. Load character reference images for all characters in scene
2. Construct final prompt: `scene.image_prompt` + consistency modifiers
3. Call image generation API with prompt + reference images
4. Save output to `data/images/scenes/scene_{id}.png`
5. Update manifest with image path and status

### Quality Checks (Automated)
- **Resolution check**: Verify output meets target resolution
- **Face detection**: Use a lightweight face detector to verify character faces are present and clear
- **NSFW filter**: Optional safety check before proceeding
- **Similarity check**: Compare generated image against character reference using CLIP embeddings. Flag if similarity score < threshold.

### Directory Structure
```
src/
  image_gen/
    __init__.py
    base.py              # Abstract BaseImageGenerator
    flux_kontext.py      # Flux Kontext implementation
    midjourney.py        # Midjourney implementation (future)
    comfyui.py           # ComfyUI implementation (future)
    consistency.py       # Character reference management, CLIP similarity
    quality_check.py     # Automated quality validation
```

### Dependencies
```
httpx>=0.27            # API calls
Pillow>=10.0           # Image processing
clip-interrogator>=0.6 # Optional: CLIP-based consistency checking
```

### Rate Limiting & Costs
- Flux Kontext API: ~$0.03-0.06 per image
- Budget tracking: Log cost per image in manifest
- Batch generation: Queue all scene images, process with concurrency limit of 3
- Retry failed generations 2x before marking as failed

---

## Part 5: Video Animator (Image-to-Video)

### Purpose
Animate each scene's static image into a short video clip (4-10 seconds) with motion that matches the scene's action and mood.

### Supported Backends (PoC: Kling API)

#### Option A: Kling 2.5 (Recommended for PoC)
- **Why**: Good quality, 40% cheaper than Runway, API available, strong Chinese market
- **Output**: ~5-10 second clips at 720p/1080p
- **Input**: Image + motion prompt
- **API**: Available via Kuaishou's developer platform

#### Option B: Runway Gen-4
- **Best quality**, especially for cinematic motion
- **API**: Available via Runway's developer API
- **Cost**: ~$0.05-0.10 per second of video

#### Option C: Hailuo AI (Free Tier)
- **Best for**: Prototyping, zero-cost testing
- **Output**: 6 second clips at 720p
- **Limitation**: Rate limited, lower resolution

### Motion Prompt Strategy
The LLM scene segmenter generates visual descriptions. The animator needs a **motion-specific prompt** derived from this:

```python
def generate_motion_prompt(scene: Scene) -> str:
    """
    Convert scene description to motion-specific prompt.
    Focus on: camera movement, character motion, environmental motion.
    """
    # Examples:
    # "slow zoom into character's face, rain falling, neon signs flickering"
    # "camera pans left to right, character walks forward, wind blowing hair"
    # "static shot, character turns head slowly, smoke rising in background"
```

**Motion categories by mood:**
| Mood | Camera Motion | Character Motion | Environment |
|------|--------------|-----------------|-------------|
| Tense | Slow push-in | Minimal, deliberate | Subtle (flickering lights) |
| Action | Fast pan/track | Dynamic movement | Particles, impacts |
| Romantic | Slow orbit | Gentle gestures | Soft particles (petals, snow) |
| Melancholy | Static or slow pull-out | Still or turning away | Rain, fog |
| Dramatic | Low angle push-in | Bold gestures | Wind, dramatic lighting |

### Module Interface

```python
class VideoAnimator:
    def __init__(self, backend: str = "kling", api_key: str = None):
        ...

    def animate_scene(
        self,
        image_path: str,
        motion_prompt: str,
        duration: float = 5.0,       # target duration in seconds
        resolution: tuple[int, int] = (1080, 1920)
    ) -> str:
        """Animate a single scene image. Returns video clip path."""

    def check_status(self, job_id: str) -> dict:
        """Check generation status for async APIs."""

    def download_result(self, job_id: str, output_path: str) -> str:
        """Download completed video. Returns path."""
```

### Async Generation Pipeline
Most video APIs are asynchronous (submit job -> poll -> download):

1. Submit all scene images for animation concurrently (respect rate limits)
2. Poll for completion every 10 seconds
3. Download completed clips to `data/videos/clips/scene_{id}.mp4`
4. Update manifest with video path and status
5. Retry failed jobs once, then mark as failed

### Duration Matching
- Scene narration length determines target video duration
- If TTS audio is 8 seconds but video clip is 5 seconds:
  - Option A: Request longer video generation
  - Option B: Slow down the clip (up to 1.5x) with frame interpolation
  - Option C: Hold last frame as a subtle zoom/pan
- If video is longer than audio: trim or speed up slightly

### Directory Structure
```
src/
  video_gen/
    __init__.py
    base.py              # Abstract BaseVideoAnimator
    kling.py             # Kling API implementation
    runway.py            # Runway implementation (future)
    hailuo.py            # Hailuo implementation (future)
    motion_prompts.py    # Motion prompt generation from scene data
    duration_sync.py     # Duration matching between audio and video
```

### Dependencies
```
httpx>=0.27            # API calls
moviepy>=2.0           # Video clip manipulation, speed adjustment
```

---

## Part 6: TTS Narrator

### Purpose
Convert each scene's narration text into natural-sounding Chinese speech audio. Support voice cloning for consistent narrator voice across all scenes.

### Supported Backends (PoC: Edge TTS for prototyping, CosyVoice for production)

#### Option A: Edge TTS (Prototyping - Free)
- **Why**: Zero cost, zero setup, decent Chinese quality
- **Voices**: `zh-CN-YunxiNeural` (male), `zh-CN-XiaoxiaoNeural` (female)
- **Limitation**: No voice cloning, limited emotion control

#### Option B: CosyVoice 3 (Production - Free, Self-hosted)
- **Why**: Best open-source Chinese TTS, 18 dialect support, voice cloning
- **Setup**: Self-hosted via Docker or direct Python install
- **Voice cloning**: Provide 5-10 seconds of reference audio

#### Option C: Fish Audio (Production - Paid API)
- **Why**: #1 ranked TTS, excellent Chinese, easy API
- **Cost**: Pay per character
- **Voice cloning**: Upload reference audio via dashboard

### Module Interface

```python
class TTSNarrator:
    def __init__(
        self,
        backend: str = "edge-tts",
        voice: str = "zh-CN-YunxiNeural",
        reference_audio: str = None     # for voice cloning backends
    ):
        ...

    def synthesize_scene(self, scene: Scene) -> TTSResult:
        """
        Generate narration audio for a single scene.
        Returns TTSResult with audio_path and word-level timestamps.
        """

    def synthesize_all(self, scenes: list[Scene], concurrency: int = 3) -> list[TTSResult]:
        """Batch synthesize all scenes."""

    def get_word_timestamps(self, audio_path: str, text: str) -> list[WordTimestamp]:
        """Extract word-level timestamps for subtitle alignment."""
```

### TTSResult Model
```python
class WordTimestamp(BaseModel):
    word: str
    start_time: float  # seconds
    end_time: float    # seconds

class TTSResult(BaseModel):
    audio_path: str
    duration_seconds: float
    word_timestamps: list[WordTimestamp]
    sample_rate: int
```

### Word Timestamps
Critical for subtitle synchronization. Acquisition methods by backend:
- **Edge TTS**: Provides word boundaries via SSML events
- **CosyVoice**: Extract from model output alignment
- **Fallback**: Use Whisper (faster-whisper) to force-align text to generated audio

### Audio Post-Processing
- Normalize volume across all scene audio clips (target: -16 LUFS)
- Add 0.3s silence padding at start/end of each clip
- Optional: Light compression to even out dynamics

### Directory Structure
```
src/
  tts/
    __init__.py
    base.py              # Abstract BaseTTS
    edge_tts.py          # Edge TTS implementation
    cosyvoice.py         # CosyVoice implementation
    fish_audio.py        # Fish Audio implementation (future)
    audio_processing.py  # Normalization, padding, compression
    alignment.py         # Whisper-based forced alignment fallback
```

### Dependencies
```
edge-tts>=6.1          # Free Microsoft TTS
pydub>=0.25            # Audio processing
pyloudnorm>=0.1        # LUFS normalization
faster-whisper>=1.0    # Fallback alignment (optional)
```

---

## Part 7: Subtitle Generator

### Purpose
Generate time-synced subtitles from TTS word timestamps. Output SRT and ASS formats for embedding in final video.

### Subtitle Styles
- **Default**: White text with black outline, bottom-center
- **Karaoke-style**: Highlight current word/phrase (for engagement)
- **Bilingual** (future): Chinese + English translation stacked

### Module Interface

```python
class SubtitleGenerator:
    def __init__(self, style: SubtitleStyle = SubtitleStyle.DEFAULT):
        ...

    def generate_srt(self, scenes: list[Scene], tts_results: list[TTSResult]) -> str:
        """Generate SRT subtitle file. Returns file path."""

    def generate_ass(self, scenes: list[Scene], tts_results: list[TTSResult]) -> str:
        """Generate ASS subtitle file with styling. Returns file path."""

    def adjust_timing(self, subtitle_path: str, offset: float) -> str:
        """Shift all subtitle timings by offset seconds."""
```

### Chunking Strategy for Chinese Text
Chinese has no word spaces, so subtitle line breaks need special handling:
- Use punctuation as natural break points (。，！？、)
- Max 15 characters per subtitle line
- Max 2 lines visible simultaneously
- Minimum display duration: 1.0 second per subtitle

### Directory Structure
```
src/
  subtitles/
    __init__.py
    generator.py         # SRT/ASS generation
    chunker.py           # Chinese text line-breaking logic
    styles.py            # ASS style definitions
```

### Dependencies
```
pysubs2>=1.7           # Subtitle file handling (SRT, ASS)
```

---

## Part 8: Final Assembly

### Purpose
Combine all generated assets (video clips, audio narration, subtitles, background music) into the final output video.

### Assembly Pipeline

```
1. Load all scene video clips in sequence order
2. Load all scene audio narration clips
3. Duration sync: adjust video/audio to match per scene
4. Concatenate scenes with transitions
5. Overlay narration audio
6. Add background music (ducked under narration)
7. Burn in subtitles
8. Add intro/outro (optional)
9. Export final video
```

### Module Interface

```python
class VideoAssembler:
    def __init__(self, manifest: Manifest):
        ...

    def sync_scene(self, scene_id: str) -> SyncedScene:
        """
        Sync video clip duration to match audio narration for one scene.
        Returns SyncedScene with matched video + audio.
        """

    def add_transitions(
        self,
        scenes: list[SyncedScene],
        transition_type: str = "crossfade",
        transition_duration: float = 0.5
    ) -> VideoClip:
        """Concatenate scenes with transitions between them."""

    def add_bgm(
        self,
        video: VideoClip,
        bgm_path: str,
        bgm_volume: float = 0.15,     # relative to narration
        duck_during_speech: bool = True
    ) -> VideoClip:
        """Add background music with auto-ducking during narration."""

    def burn_subtitles(self, video: VideoClip, subtitle_path: str) -> VideoClip:
        """Burn ASS subtitles into video."""

    def export(
        self,
        video: VideoClip,
        output_path: str,
        codec: str = "libx264",
        audio_codec: str = "aac",
        bitrate: str = "8000k"
    ) -> str:
        """Export final video. Returns output path."""

    def assemble_full(self) -> str:
        """Run the full assembly pipeline end-to-end. Returns final video path."""
```

### Transition Types
| Type | When to Use | Duration |
|------|------------|----------|
| Crossfade | Default, smooth scene changes | 0.5s |
| Cut | Action sequences, fast pacing | 0s |
| Fade to black | Time skips, chapter breaks | 1.0s |
| Fade to white | Flashbacks, dream sequences | 1.0s |

The LLM segmenter can suggest transition types per scene in the manifest.

### Background Music
- Source: Royalty-free library or AI-generated (Suno/Udio)
- Auto-duck: Reduce BGM volume by 60-80% when narration is playing
- Multiple BGM tracks mapped to mood segments (defined in manifest)
- Crossfade between BGM tracks at mood transition points

### Export Profiles
```python
PROFILES = {
    "youtube_vertical": {  # YouTube Shorts / TikTok / Douyin
        "resolution": (1080, 1920),
        "fps": 30,
        "codec": "libx264",
        "bitrate": "8000k",
        "audio_codec": "aac",
        "audio_bitrate": "192k"
    },
    "youtube_horizontal": {  # Standard YouTube
        "resolution": (1920, 1080),
        "fps": 30,
        "codec": "libx264",
        "bitrate": "10000k",
        "audio_codec": "aac",
        "audio_bitrate": "192k"
    },
    "draft": {  # Fast preview
        "resolution": (540, 960),
        "fps": 24,
        "codec": "libx264",
        "bitrate": "2000k",
        "audio_codec": "aac",
        "audio_bitrate": "128k"
    }
}
```

### Directory Structure
```
src/
  assembler/
    __init__.py
    assembler.py         # Main VideoAssembler class
    transitions.py       # Transition effects
    audio_mixer.py       # BGM mixing and ducking
    export.py            # Export profiles and encoding
```

### Dependencies
```
moviepy>=2.0           # Video composition
pydub>=0.25            # Audio ducking/mixing
ffmpeg-python>=0.2     # FFmpeg wrapper for complex operations
```

---

## Part 9: Quality Review & Iteration

### Purpose
Automated and semi-automated review of generated assets before final assembly. Catch bad generations early to save time and API costs.

### Automated Checks

#### Image Quality
| Check | Method | Threshold |
|-------|--------|-----------|
| Character presence | Face detection (MediaPipe) | >= expected face count |
| Character consistency | CLIP embedding similarity to reference | > 0.75 cosine similarity |
| Style consistency | CLIP embedding similarity to style reference | > 0.80 cosine similarity |
| Resolution | Pillow image dimensions | Matches target |
| NSFW | Safety classifier | Score < 0.3 |

#### Video Quality
| Check | Method | Threshold |
|-------|--------|-----------|
| Duration | FFprobe | Within 20% of target |
| Static/frozen | Frame difference analysis | < 95% frame similarity |
| Artifacts | Visual quality metric (BRISQUE) | Score > 30 |

#### Audio Quality
| Check | Method | Threshold |
|-------|--------|-----------|
| Duration | Audio file length | Matches narration text estimate |
| Silence ratio | Energy analysis | < 30% silence |
| Volume level | LUFS measurement | -20 to -14 LUFS |

### LLM Review (Optional Enhancement)
Send the generated image + scene description to a vision-capable LLM:
```
"Does this image accurately depict: {scene.visual_description}?
 Are these characters present and recognizable: {scene.characters_present}?
 Rate accuracy 1-10 and list any issues."
```

### Review Workflow
1. Auto-checks run immediately after each asset is generated
2. Failed checks -> auto-regenerate (up to 2 retries)
3. Still failing -> flag for manual review in manifest
4. Optional: Open a simple web UI showing all scenes with approve/reject buttons

### Module Interface

```python
class QualityReviewer:
    def __init__(self, manifest: Manifest):
        ...

    def review_image(self, scene: Scene) -> ReviewResult:
        """Run all automated image quality checks."""

    def review_video(self, scene: Scene) -> ReviewResult:
        """Run all automated video quality checks."""

    def review_audio(self, scene: Scene) -> ReviewResult:
        """Run all automated audio quality checks."""

    def review_all(self) -> list[ReviewResult]:
        """Review all assets in manifest. Returns list of issues."""

    def llm_review_image(self, scene: Scene) -> ReviewResult:
        """Use vision LLM to review image accuracy."""
```

### Directory Structure
```
src/
  quality/
    __init__.py
    reviewer.py          # Main QualityReviewer class
    image_checks.py      # Image quality checks
    video_checks.py      # Video quality checks
    audio_checks.py      # Audio quality checks
    llm_review.py        # Optional LLM-based review
```

---

## Part 10: CLI & Orchestrator (Full Control Center)

### Purpose
The CLI is the single interface for everything — GPU management, pipeline execution, progress monitoring, configuration, and project management. The user should never need to leave the terminal.

### CLI Command Groups

```bash
n2v <command-group> <command> [options]
```

---

### 10.1 GPU & Infrastructure Management

```bash
# ── Setup & Config ──────────────────────────────────────────
n2v setup                                  # First-time setup wizard (API keys, defaults)
n2v config show                            # Show current configuration
n2v config set image.model flux-dev        # Set any config value
n2v config set gpu.provider runpod         # Set GPU provider (runpod / vastai)
n2v config set gpu.type RTX4090            # Preferred GPU type
n2v config set deepseek.api_key sk-...     # Set API keys
n2v config set tts.voice zh-CN-YunxiNeural # Set TTS voice

# ── GPU Instance Management ─────────────────────────────────
n2v gpu status                             # Show current GPU instance status
n2v gpu start                              # Spin up a GPU instance (uses config defaults)
n2v gpu start --type RTX3090 --provider vastai  # Override defaults
n2v gpu stop                               # Shut down instance (saves money!)
n2v gpu ssh                                # SSH into running instance
n2v gpu cost                               # Show current session cost + monthly projection
n2v gpu deploy-comfyui                     # Install/update ComfyUI + Flux on instance
n2v gpu test                               # Test connectivity & generate a test image
n2v gpu benchmark                          # Run speed test (images/minute on current GPU)
```

**GPU Auto-Management**: The pipeline can auto-start a GPU before image generation and auto-stop it after, so you never pay for idle time.

```python
class GPUManager:
    def __init__(self, provider: str = "runpod", api_key: str = None):
        ...

    def start_instance(self, gpu_type: str = "RTX4090", template: str = "comfyui-flux") -> Instance:
        """Spin up a cloud GPU with ComfyUI pre-installed."""

    def stop_instance(self, instance_id: str) -> None:
        """Shut down instance to stop billing."""

    def get_status(self) -> InstanceStatus:
        """Get current instance status, uptime, cost."""

    def ensure_ready(self) -> Instance:
        """Start instance if not running, wait until ComfyUI is responsive."""

    def get_comfyui_endpoint(self) -> str:
        """Return the ComfyUI REST API URL for the running instance."""

    def deploy_comfyui(self, instance: Instance) -> None:
        """Install ComfyUI + Flux Dev + IP-Adapter on the instance."""

    def get_session_cost(self) -> float:
        """Get current session cost in dollars."""
```

#### RunPod Integration
```python
# Uses RunPod's API: https://docs.runpod.io/reference
# Template: RunPod has pre-built ComfyUI templates
# Flow: Create pod -> Wait for ready -> Get IP -> Use ComfyUI API at http://{ip}:8188
```

#### Vast.ai Integration
```python
# Uses Vast.ai's API: https://vast.ai/docs/api
# Flow: Search offers -> Rent instance -> SSH setup -> Deploy ComfyUI -> Use API
```

---

### 10.2 Project Management

```bash
# ── Projects ────────────────────────────────────────────────
n2v project create "仙逆" --url "https://fanqienovel.com/page/7345678"
                                           # Create a new video project from novel URL
n2v project create --file ./my_story.txt   # Create from local text file
n2v project list                           # List all projects with status
n2v project show 仙逆                       # Show detailed project status
n2v project delete 仙逆                     # Delete a project and its assets

# ── Novel/Chapter Management ────────────────────────────────
n2v novel info --url "https://fanqienovel.com/page/7345678"
                                           # Show novel info without downloading
n2v novel chapters --url "https://fanqienovel.com/page/7345678"
                                           # List all chapters
n2v novel download --url "https://fanqienovel.com/page/7345678" --chapters 1-10
                                           # Download specific chapters
```

---

### 10.3 Pipeline Execution

```bash
# ── Full Pipeline (one command to rule them all) ────────────
n2v run --url "https://fanqienovel.com/reader/73456" --style anime
                                           # Full pipeline: scrape -> segment -> images -> TTS -> assemble
n2v run --file ./chapter1.txt --style realistic
                                           # From local file
n2v run --project 仙逆 --chapters 1-5      # Run for specific chapters in existing project

# ── Individual Stages ───────────────────────────────────────
n2v scrape --url "https://fanqienovel.com/reader/73456"
n2v segment --project 仙逆 --chapter 1     # Segment a chapter into scenes
n2v generate-images --project 仙逆 --chapter 1  # Generate all scene images
n2v narrate --project 仙逆 --chapter 1     # Generate TTS narration
n2v assemble --project 仙逆 --chapter 1    # Assemble final video

# ── Retry & Fix ─────────────────────────────────────────────
n2v retry --project 仙逆 --failed          # Retry all failed assets
n2v retry --project 仙逆 --scene 15        # Retry specific scene
n2v regenerate --project 仙逆 --scene 15 --feedback "character should have red hair"
                                           # Regenerate with adjustment
```

---

### 10.4 Progress Monitoring (Real-Time Dashboard)

```bash
# ── Live Progress ───────────────────────────────────────────
n2v status                                 # Quick status of current/recent job
n2v status --project 仙逆                   # Detailed status for a project
n2v status --live                          # Live-updating dashboard (refreshes every 2s)
n2v logs                                   # Tail pipeline logs
n2v logs --stage images                    # Logs for specific stage only
```

#### Live Dashboard (Rich Terminal UI)

```
┌─────────────────────────────────────────────────────────────┐
│  n2v - Novel to Video Pipeline                    00:12:34  │
├─────────────────────────────────────────────────────────────┤
│  Project: 仙逆 Chapter 3                                     │
│  GPU: RunPod RTX 4090 ($0.29/hr) ● Running  Cost: $0.06    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1/6] Scrape     ████████████████████████████████ 100% ✓   │
│  [2/6] Segment    ████████████████████████████████ 100% ✓   │
│        → 180 scenes, 4 characters extracted                 │
│  [3/6] Images     ████████████████░░░░░░░░░░░░░░░░  52%     │
│        → 94/180 generated  ~12 min remaining                │
│        → 2 failed (will retry), 0 flagged                   │
│  [4/6] Narrate    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%     │
│  [5/6] Subtitles  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%     │
│  [6/6] Assemble   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%     │
│                                                             │
│  Latest: scene_094.png generated (1024x1792, 0.8s)          │
│  ETA: 23 minutes total                                      │
│                                                             │
│  Cost this session: GPU $0.06 + DeepSeek $0.01 = $0.07      │
└─────────────────────────────────────────────────────────────┘
```

Implementation uses `rich.live.Live` with a custom layout that polls manifest status every 2 seconds.

```python
class LiveDashboard:
    def __init__(self, manifest: Manifest):
        ...

    def start(self) -> None:
        """Start live dashboard in terminal."""

    def update(self) -> None:
        """Refresh display from manifest state."""

    def render_progress_bar(self, stage: str, completed: int, total: int) -> str:
        """Render a single stage progress bar."""

    def render_gpu_status(self) -> str:
        """Show GPU instance status and running cost."""

    def render_cost_tracker(self) -> str:
        """Show real-time cost breakdown."""
```

---

### 10.5 Configuration & Style Management

```bash
# ── Art Style Presets ───────────────────────────────────────
n2v style list                             # Show available art styles
n2v style show anime                       # Show style details and example prompt
n2v style create wuxia --base anime \
    --color-palette "ink wash, gold accents" \
    --lighting "dramatic, volumetric" \
    --description "Chinese martial arts fantasy"
                                           # Create custom style preset

# ── Character Management ────────────────────────────────────
n2v characters list --project 仙逆          # List extracted characters
n2v characters show 王林 --project 仙逆     # Show character details + reference image
n2v characters edit 王林 --project 仙逆 \
    --description "30-year-old cultivator, white robes, cold expression, sword on back"
                                           # Edit character description (regenerates ref image)
n2v characters regenerate 王林 --project 仙逆
                                           # Regenerate character reference sheet

# ── Scene Management ────────────────────────────────────────
n2v scenes list --project 仙逆 --chapter 1  # List all scenes in a chapter
n2v scenes show 42 --project 仙逆           # Show scene details (prompt, image, status)
n2v scenes edit 42 --project 仙逆 \
    --visual "王林 stands on cliff edge, storm clouds gathering, lightning in distance"
                                           # Edit scene visual description
n2v scenes preview --project 仙逆 --chapter 1
                                           # Open image grid of all scenes (in terminal or browser)
```

---

### 10.6 Batch & Scheduling

```bash
# ── Batch Processing ────────────────────────────────────────
n2v batch --project 仙逆 --chapters 1-30   # Queue 30 chapters for sequential processing
n2v batch status                           # Show batch queue status
n2v batch pause                            # Pause batch after current chapter finishes
n2v batch resume                           # Resume batch processing

# ── Daily Automation ────────────────────────────────────────
n2v schedule set --time 02:00 \
    --project 仙逆 \
    --auto-next-chapter \
    --auto-gpu                             # Run next chapter every night at 2 AM
                                           # Auto-starts GPU, generates, shuts down GPU
n2v schedule show                          # Show scheduled jobs
n2v schedule disable                       # Disable scheduled runs
```

#### Auto-Pipeline Mode
For daily video production, the pipeline can run fully unattended:

```python
class AutoPipeline:
    """Fully automated daily video production."""

    def run_next_chapter(self, project_id: str) -> str:
        """
        1. Find the next un-processed chapter in the project
        2. Start GPU instance
        3. Run full pipeline (scrape -> segment -> images -> TTS -> assemble)
        4. Stop GPU instance
        5. Save final video to output/
        6. Send notification (optional: webhook, email, desktop notification)
        7. Return path to finished video
        """

    def run_batch(self, project_id: str, chapter_range: tuple[int, int]) -> None:
        """Process multiple chapters sequentially with auto GPU management."""
```

---

### 10.7 Review & Quality Control

```bash
# ── Review ──────────────────────────────────────────────────
n2v review --project 仙逆 --chapter 1      # Run automated quality checks
n2v review --project 仙逆 --chapter 1 --open
                                           # Open review in browser (simple HTML gallery)
n2v review approve --project 仙逆 --chapter 1
                                           # Mark all assets as approved
n2v review reject --project 仙逆 --scene 42 --reason "wrong character"
                                           # Reject and queue for regeneration

# ── Preview ─────────────────────────────────────────────────
n2v preview --project 仙逆 --chapter 1     # Quick draft export (low res, fast)
n2v preview --project 仙逆 --chapter 1 --open
                                           # Export and open in default video player
```

---

### 10.8 Cost & Analytics

```bash
# ── Cost Tracking ───────────────────────────────────────────
n2v cost today                             # Today's spending breakdown
n2v cost month                             # This month's total
n2v cost project 仙逆                       # Total cost for a project
n2v cost estimate --chapters 30            # Estimate cost for N chapters

# Output example:
# ┌─────────────────────────────────────────┐
# │  Cost Report - March 2026               │
# ├─────────────────────────────────────────┤
# │  GPU (RunPod)      $4.35  (14.5 hrs)   │
# │  DeepSeek API      $0.28  (930K tokens) │
# │  Total             $4.63               │
# │  Videos produced   28                   │
# │  Cost per video    $0.17               │
# └─────────────────────────────────────────┘
```

---

### 10.9 First-Time Setup Wizard

```bash
n2v setup
```

Interactive setup that configures everything:

```
┌─────────────────────────────────────────────────────────────┐
│  Welcome to Novel-to-Video Pipeline Setup                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1/5: GPU Provider                                     │
│  ┌─ Which cloud GPU provider? ────────────────────────────┐ │
│  │  > RunPod (recommended, easiest)                       │ │
│  │    Vast.ai (cheapest)                                  │ │
│  │    Local GPU (I have my own)                           │ │
│  │    Skip (configure later)                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 2/5: RunPod API Key                                   │
│  Enter your RunPod API key: ████████████████████            │
│  ✓ Connected! 3 GPU types available in your region.         │
│                                                             │
│  Step 3/5: DeepSeek API Key                                 │
│  Enter your DeepSeek API key: ████████████████████          │
│  ✓ Connected! Balance: $5.00                                │
│                                                             │
│  Step 4/5: Default Art Style                                │
│  ┌─ Choose default art style ─────────────────────────────┐ │
│  │  > Anime (vibrant, clear characters)                   │ │
│  │    Realistic (photorealistic, cinematic)                │ │
│  │    Ink Wash (Chinese traditional painting style)        │ │
│  │    Comic (bold lines, high contrast)                    │ │
│  │    Custom (define your own)                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 5/5: Test Run                                         │
│  Starting GPU... ✓ (RTX 4090, $0.29/hr)                    │
│  Generating test image... ✓ (3.2 seconds)                   │
│  TTS test... ✓                                              │
│  Shutting down GPU... ✓                                     │
│                                                             │
│  ✓ Setup complete! Run `n2v run --url <URL>` to start.     │
└─────────────────────────────────────────────────────────────┘
```

---

### CLI Directory Structure
```
src/
  cli/
    __init__.py
    main.py              # Typer app entry point, command group registration
    setup.py             # First-time setup wizard
    gpu_commands.py      # GPU management commands
    project_commands.py  # Project & novel commands
    pipeline_commands.py # Run, stage, retry commands
    status_commands.py   # Status, logs, dashboard
    style_commands.py    # Art style & character management
    batch_commands.py    # Batch & scheduling commands
    review_commands.py   # Review & preview commands
    cost_commands.py     # Cost tracking & estimates
  core/
    pipeline.py          # Pipeline orchestrator
    auto_pipeline.py     # Fully automated daily pipeline
    config.py            # PipelineConfig with defaults
  infra/
    __init__.py
    gpu_manager.py       # Abstract GPU manager
    runpod.py            # RunPod API integration
    vastai.py            # Vast.ai API integration
    local_gpu.py         # Local GPU / ComfyUI detection
  dashboard/
    __init__.py
    live.py              # Rich live dashboard
    cost_tracker.py      # Cost tracking and reporting
    notifications.py     # Desktop/webhook notifications
```

### Dependencies
```
typer>=0.12            # CLI framework
rich>=13.0             # Terminal UI, progress bars, tables, live display
httpx>=0.27            # API calls to RunPod/Vast.ai
apscheduler>=3.10      # Job scheduling for daily automation
plyer>=2.1             # Desktop notifications (cross-platform)
```

---

## Project Structure (Complete)

```
novel-to-video/
├── SPEC.md
├── README.md
├── pyproject.toml
├── .env.example           # API keys template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── manifest.py    # Pydantic models
│   │   └── enums.py       # Status enums
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifest_manager.py
│   │   ├── pipeline.py
│   │   └── config.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fanqie.py
│   │   ├── font_decryptor.py
│   │   └── rate_limiter.py
│   ├── segmenter/
│   │   ├── __init__.py
│   │   ├── segmenter.py
│   │   ├── prompts.py
│   │   ├── character_extractor.py
│   │   └── prompt_builder.py
│   ├── image_gen/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── flux_kontext.py
│   │   ├── consistency.py
│   │   └── quality_check.py
│   ├── video_gen/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── kling.py
│   │   ├── motion_prompts.py
│   │   └── duration_sync.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── edge_tts.py
│   │   ├── cosyvoice.py
│   │   ├── audio_processing.py
│   │   └── alignment.py
│   ├── subtitles/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── chunker.py
│   │   └── styles.py
│   ├── assembler/
│   │   ├── __init__.py
│   │   ├── assembler.py
│   │   ├── transitions.py
│   │   ├── audio_mixer.py
│   │   └── export.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── reviewer.py
│   │   ├── image_checks.py
│   │   ├── video_checks.py
│   │   ├── audio_checks.py
│   │   └── llm_review.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py              # Typer app entry point
│   │   ├── setup.py             # First-time setup wizard
│   │   ├── gpu_commands.py      # GPU management (start/stop/status)
│   │   ├── project_commands.py  # Project & novel commands
│   │   ├── pipeline_commands.py # Run, stage, retry commands
│   │   ├── status_commands.py   # Status, logs, live dashboard
│   │   ├── style_commands.py    # Art style & character management
│   │   ├── batch_commands.py    # Batch & scheduling
│   │   ├── review_commands.py   # Review & preview
│   │   └── cost_commands.py     # Cost tracking
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── gpu_manager.py       # Abstract GPU manager
│   │   ├── runpod.py            # RunPod API integration
│   │   ├── vastai.py            # Vast.ai API integration
│   │   └── local_gpu.py         # Local GPU detection
│   └── dashboard/
│       ├── __init__.py
│       ├── live.py              # Rich live terminal dashboard
│       ├── cost_tracker.py      # Cost tracking and reporting
│       └── notifications.py     # Desktop/webhook notifications
│
├── data/                  # Generated per-project (gitignored)
│   ├── raw/               # Scraped text
│   ├── images/
│   │   ├── characters/    # Character reference sheets
│   │   └── scenes/        # Scene images
│   ├── videos/
│   │   └── clips/         # Animated scene clips
│   ├── audio/
│   │   ├── narration/     # TTS audio clips
│   │   └── bgm/           # Background music
│   ├── subtitles/         # SRT/ASS files
│   └── output/            # Final assembled videos
│
├── config/
│   ├── default.yaml       # Default pipeline config
│   └── styles/            # Art style presets
│       ├── anime.yaml
│       ├── realistic.yaml
│       └── comic.yaml
│
└── tests/
    ├── test_scraper.py
    ├── test_segmenter.py
    ├── test_image_gen.py
    ├── test_tts.py
    └── test_assembler.py
```

---

## Implementation Order (PoC)

Build and test each part independently before connecting them:

| Phase | Part | Effort | Depends On |
|-------|------|--------|-----------|
| 1 | Part 1: Manifest & Data Models | 1 session | Nothing |
| 2 | Part 2: Fanqie Scraper | 1-2 sessions | Part 1 |
| 3 | Part 3: Scene Segmenter | 1-2 sessions | Part 1 |
| 4 | Part 6: TTS Narrator (Edge TTS) | 1 session | Part 1, 3 |
| 5 | Part 7: Subtitle Generator | 1 session | Part 6 |
| 6 | Part 4: Image Generator (Flux) | 1-2 sessions | Part 1, 3 |
| 7 | Part 5: Video Animator (Kling) | 1-2 sessions | Part 4 |
| 8 | Part 8: Final Assembly | 1-2 sessions | Parts 4-7 |
| 9 | Part 9: Quality Review | 1 session | Parts 4-6 |
| 10 | Part 10: CLI & Orchestrator | 1 session | All above |

**Total PoC: ~10-14 working sessions**

---

## Finalized Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| **LLM** (segmentation) | DeepSeek API | ~$0.01/chapter |
| **Image Generation** | Flux Dev via ComfyUI on cloud GPU | ~$0.15/chapter (GPU time) |
| **Video Animation** | SKIPPED (Ken Burns pan/zoom on static images) | $0 |
| **TTS** | Edge TTS | $0 |
| **Subtitles** | pysubs2 (local) | $0 |
| **Assembly** | MoviePy + FFmpeg (local) | $0 |
| **GPU** | RunPod (primary) / Vast.ai (backup) | ~$0.15-0.30/session |

## API Keys Required

```env
# LLM (scene segmentation)
DEEPSEEK_API_KEY=sk-...

# GPU Provider (pick one)
RUNPOD_API_KEY=...
VASTAI_API_KEY=...

# TTS (Edge TTS needs no key)
```

---

## Cost Estimate Per Video (30-min video, 180 images)

| Component | Unit Cost | Quantity | Total |
|-----------|----------|----------|-------|
| DeepSeek (segmentation) | ~$0.01 | 2 calls | $0.02 |
| GPU rental (image gen) | $0.29/hr | ~0.5 hr | $0.15 |
| TTS (Edge TTS) | Free | 180 clips | $0.00 |
| Assembly (local) | Free | 1 video | $0.00 |
| **Total per video** | | | **~$0.17** |

**Monthly (30 videos/day):** ~$5.10
**Yearly:** ~$62
