from enum import Enum


class AssetStatus(str, Enum):
    """Status of a generated asset (image, audio, subtitle)."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"
    APPROVED = "approved"


class PipelineStage(str, Enum):
    """Stages of the video pipeline."""
    SCRAPE = "scrape"
    SEGMENT = "segment"
    IMAGES = "images"
    NARRATE = "narrate"
    SUBTITLES = "subtitles"
    ASSEMBLE = "assemble"


class ProjectStatus(str, Enum):
    """Overall project status."""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    PAUSED = "paused"


class Mood(str, Enum):
    """Scene mood/emotional tone."""
    TENSE = "tense"
    ACTION = "action"
    JOYFUL = "joyful"
    MELANCHOLY = "melancholy"
    ROMANTIC = "romantic"
    DRAMATIC = "dramatic"
    MYSTERIOUS = "mysterious"
    PEACEFUL = "peaceful"
    HUMOROUS = "humorous"
    HORROR = "horror"


class TransitionType(str, Enum):
    """Transition between scenes."""
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_TO_BLACK = "fade_to_black"
    FADE_TO_WHITE = "fade_to_white"


class GPUProvider(str, Enum):
    """Supported GPU cloud providers."""
    RUNPOD = "runpod"
    VASTAI = "vastai"
    LOCAL = "local"


class ImageModel(str, Enum):
    """Supported image generation models."""
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    SDXL = "sdxl"


class TTSBackend(str, Enum):
    """Supported TTS backends."""
    EDGE_TTS = "edge-tts"
    COSYVOICE = "cosyvoice"
    FISH_AUDIO = "fish-audio"
