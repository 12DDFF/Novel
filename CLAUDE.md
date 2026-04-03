# Novel-to-Video Pipeline

Automated pipeline that converts Chinese web novels into AI-generated narration videos (视频解说 style).

## Quick Start

```bash
# Set PATH for ffmpeg
export PATH="$PATH:/c/Users/zihao/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin"

# Generate a 视频解说 video (chapters 1-5, placeholder images)
PYTHONIOENCODING=utf-8 python -m src.cli.main narrate --novel-id 7067003728020114473 --start 1 --end 5 --minutes 3 --scenes 15 --placeholder

# Generate with real Kontext Pro images (costs ~$0.04/image)
PYTHONIOENCODING=utf-8 python -m src.cli.main narrate --novel-id 7067003728020114473 --start 1 --end 5 --minutes 3 --scenes 15

# Old chapter-by-chapter mode
PYTHONIOENCODING=utf-8 python -m src.cli.main run --novel-id 7330513034248457278 --chapter 1

# Plan character sheets for entire novel (scan first N chapters)
PYTHONIOENCODING=utf-8 python -m src.cli.main plan-characters --novel-id 7330513034248457278 --scan 5

# Re-run only subtitles + assembly (fast, no API cost)
PYTHONIOENCODING=utf-8 python -m src.cli.main run --novel-id 7330513034248457278 --chapter 1 --from-step subtitles
```

## Architecture

```
Scrape (Fanqie) → LLM Narration (视频解说 style) → Image Gen (Flux Kontext Pro via Replicate) → TTS (Edge TTS multi-voice) → Subtitles → Ken Burns Assembly → Final MP4
```

## Key Modules
- `src/scraper/` — Fanqie novel scraper with font decryption
- `src/segmenter/narrator.py` — 视频解说 script generator (小帅/小美/黄毛 archetypes)
- `src/segmenter/rewriter.py` — Chapter rewriter (old approach)
- `src/image_gen/replicate_backend.py` — Flux Kontext Pro via Replicate API
- `src/tts/multi_voice.py` — Multi-voice TTS with character voice assignment
- `src/assembler/` — Ken Burns effects + video assembly
- `src/cli/main.py` — CLI entry point

## API Keys (in .env)
- OPENROUTER_API_KEY — for LLM (DeepSeek via OpenRouter)
- REPLICATE_API_TOKEN — for image generation (Flux Kontext Pro)
- Edge TTS — free, no key needed

## Current Novels
- `novel_7330513034248457278` — 末世财阀 (chapter-by-chapter, 5 chapters done)
- `novel_7067003728020114473` — 都市玄门医婿 (视频解说 narration, ch1-5)

## Settings
- Subtitles: fontsize=7, marginv=55, max 25 chars/line
- TTS speed: +25-30%
- Narrator voice: YunxiNeural, Protagonist: YunyangNeural, Female: XiaoxiaoNeural, Elder: YunjianNeural
- Images: Flux Kontext Pro ($0.04/img), character refs stored in shared_characters/

## Tests
137 unit tests: `python -m pytest tests/ -k "not Integration" -q`
