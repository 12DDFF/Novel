"""
Hetushu.com (和图书) scraper.

Fully free novel platform with server-side rendered content.
No paywall, no font encryption, no JavaScript rendering needed.
"""

from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup

from .base import BaseScraper, ChapterContent, ChapterInfo, NovelInfo
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.hetushu.com"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class HetushuScraper(BaseScraper):
    """Scraper for hetushu.com — fully free, no paywall."""

    def __init__(self, rate_limit: float = 1.0, max_retries: int = 3, timeout: float = 15.0):
        self.rate_limiter = RateLimiter(min_interval=rate_limit)
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": _USER_AGENTS[0],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

    def _request(self, url: str) -> httpx.Response:
        import time
        last_error = None
        for attempt in range(self.max_retries):
            self.rate_limiter.wait()
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                wait_time = (attempt + 1) * 2
                logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                time.sleep(wait_time)
        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_error}")

    def get_novel_info(self, novel_id: str) -> NovelInfo:
        url = f"{_BASE_URL}/book/{novel_id}/index.html"
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")

        title_el = soup.select_one("h2") or soup.select_one("h1")
        title = title_el.get_text(strip=True) if title_el else ""

        author_el = soup.select_one(".book_info a[href*='/author/']")
        author = author_el.get_text(strip=True) if author_el else ""

        desc_el = soup.select_one(".intro") or soup.select_one(".book_info")
        description = desc_el.get_text(strip=True) if desc_el else ""

        chapters = self._get_chapter_links(soup, novel_id)

        return NovelInfo(
            novel_id=novel_id,
            title=title,
            author=author,
            description=description[:200],
            cover_url="",
            word_count="",
            chapter_count=len(chapters),
            platform="hetushu",
        )

    def get_chapter_list(self, novel_id: str) -> list[ChapterInfo]:
        url = f"{_BASE_URL}/book/{novel_id}/index.html"
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")
        return self._get_chapter_links(soup, novel_id)

    def download_chapter(self, chapter_id: str) -> ChapterContent:
        url = f"{_BASE_URL}{chapter_id}" if chapter_id.startswith("/") else chapter_id
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")

        # Get title
        title_el = soup.select_one("h2") or soup.select_one("h1") or soup.select_one("title")
        title = title_el.get_text(strip=True) if title_el else ""

        # Get content
        content_el = soup.select_one("#content")
        if content_el:
            # Remove only script and ad tags, NOT divs (content is in divs)
            for script in content_el.find_all(["script", "ins"]):
                script.decompose()
            text = content_el.get_text(separator="\n", strip=True)
        else:
            text = ""

        return ChapterContent(
            chapter_id=chapter_id,
            title=title,
            text=text,
            images=[],
        )

    def download_chapters(
        self, novel_id: str, start: int = 0, end: int | None = None
    ) -> list[ChapterContent]:
        chapters = self.get_chapter_list(novel_id)
        if end is not None:
            chapters = chapters[start:end]
        else:
            chapters = chapters[start:]

        results = []
        for ch in chapters:
            logger.info("Downloading: %s", ch.title)
            content = self.download_chapter(ch.url)
            results.append(content)
        return results

    def search(self, query: str) -> list[NovelInfo]:
        # Hetushu doesn't have a search API, return empty
        return []

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _get_chapter_links(soup: BeautifulSoup, novel_id: str) -> list[ChapterInfo]:
        pattern = re.compile(rf"/book/{novel_id}/\d+\.html")
        chapters = []
        seen = set()
        for i, a in enumerate(soup.select("a[href]")):
            href = a.get("href", "")
            if pattern.search(href) and href not in seen:
                seen.add(href)
                chapters.append(
                    ChapterInfo(
                        chapter_id=href,
                        title=a.get_text(strip=True),
                        sequence=i,
                        url=href,
                    )
                )
        return chapters
