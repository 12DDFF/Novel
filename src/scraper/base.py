from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class NovelInfo:
    """Metadata about a novel."""
    novel_id: str = ""
    title: str = ""
    author: str = ""
    description: str = ""
    cover_url: str = ""
    word_count: str = ""
    chapter_count: int = 0
    platform: str = ""


@dataclass
class ChapterInfo:
    """Metadata about a single chapter."""
    chapter_id: str = ""
    title: str = ""
    sequence: int = 0
    url: str = ""


@dataclass
class ChapterContent:
    """Downloaded chapter content."""
    chapter_id: str = ""
    title: str = ""
    text: str = ""
    images: list[str] = field(default_factory=list)  # image URLs found in chapter


class BaseScraper(ABC):
    """Abstract base class for novel scrapers."""

    @abstractmethod
    def get_novel_info(self, novel_id: str) -> NovelInfo:
        """Get novel metadata."""

    @abstractmethod
    def get_chapter_list(self, novel_id: str) -> list[ChapterInfo]:
        """Get ordered list of all chapters."""

    @abstractmethod
    def download_chapter(self, chapter_id: str) -> ChapterContent:
        """Download a single chapter's content."""

    def download_chapters(
        self,
        novel_id: str,
        start: int = 0,
        end: int | None = None,
    ) -> list[ChapterContent]:
        """Download a range of chapters."""
        chapters = self.get_chapter_list(novel_id)
        if end is None:
            end = len(chapters)
        results = []
        for ch in chapters[start:end]:
            content = self.download_chapter(ch.chapter_id)
            results.append(content)
        return results

    @staticmethod
    def extract_novel_id(url: str) -> str:
        """Extract novel/book ID from a URL. Override per platform."""
        import re
        # Try /page/{id} pattern
        match = re.search(r"/page/(\d+)", url)
        if match:
            return match.group(1)
        # Try book_id={id} pattern
        match = re.search(r"book_id=(\d+)", url)
        if match:
            return match.group(1)
        # Try /reader/{id} pattern
        match = re.search(r"/reader/(\d+)", url)
        if match:
            return match.group(1)
        # Maybe it's just a number
        if url.strip().isdigit():
            return url.strip()
        raise ValueError(f"Could not extract novel ID from: {url}")
