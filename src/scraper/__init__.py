from .base import BaseScraper, ChapterContent, ChapterInfo, NovelInfo
from .fanqie import FanqieScraper
from .font_decryptor import decrypt_text, is_encrypted

__all__ = [
    "BaseScraper",
    "ChapterContent",
    "ChapterInfo",
    "FanqieScraper",
    "NovelInfo",
    "decrypt_text",
    "is_encrypted",
]
