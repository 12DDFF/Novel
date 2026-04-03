"""
Fanqie Novel (fanqienovel.com) scraper.

Uses two approaches:
1. Primary: novel.snssdk.com API (returns clean text, no decryption needed)
2. Fallback: Direct HTML scraping (requires font decryption)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time

import httpx
from bs4 import BeautifulSoup

from .base import BaseScraper, ChapterContent, ChapterInfo, NovelInfo
from .font_decryptor import decrypt_text, is_encrypted
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Chapter content API (returns clean text)
_API_URL = (
    "https://novel.snssdk.com/api/novel/book/reader/full/v1/"
    "?device_platform=android&parent_enterfrom=novel_channel_search.tab."
    "&aid=2329&platform_id=1&group_id={chapter_id}&item_id={chapter_id}"
)

# Search API
_SEARCH_URL = (
    "https://fanqienovel.com/api/author/search/search_book/v1"
    "?filter=127,127,127,127&page_count=10&page_index=0"
    "&query_type=0&query_word={query}"
)

_BASE_URL = "https://fanqienovel.com"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class FanqieScraper(BaseScraper):
    """Scraper for fanqienovel.com."""

    def __init__(
        self,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: float = 15.0,
    ):
        self.rate_limiter = RateLimiter(min_interval=rate_limit)
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=self._make_headers(),
        )

    def _make_headers(self) -> dict[str, str]:
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }

    def _request(self, url: str, **kwargs) -> httpx.Response:
        """Make a rate-limited HTTP request with retries."""
        last_error = None
        for attempt in range(self.max_retries):
            self.rate_limiter.wait()
            try:
                response = self._client.get(url, **kwargs)
                response.raise_for_status()

                # Check for WAF/captcha
                if "验证" in response.text[:500] or "WAF" in response.text[:500]:
                    raise RuntimeError("Captcha/WAF detected. Try again later or use a different IP.")

                return response
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                wait_time = (attempt + 1) * 2
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_error}")

    def search(self, query: str) -> list[NovelInfo]:
        """Search for novels by keyword."""
        url = _SEARCH_URL.format(query=query)
        response = self._request(url)
        data = response.json()

        results = []
        for item in data.get("data", {}).get("search_book_data_list", []):
            results.append(NovelInfo(
                novel_id=str(item.get("book_id", "")),
                title=item.get("book_name", ""),
                author=item.get("author", ""),
                description=item.get("abstract", ""),
                word_count=item.get("word_count", ""),
                platform="fanqie",
            ))
        return results

    def get_novel_info(self, novel_id: str) -> NovelInfo:
        """Get novel metadata by scraping the info page."""
        url = f"{_BASE_URL}/page/{novel_id}"
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")

        title = ""
        title_el = soup.find("h1")
        if title_el:
            title = decrypt_text(title_el.get_text(strip=True))

        author = ""
        author_el = soup.find("span", class_="author-name-text")
        if author_el:
            author = decrypt_text(author_el.get_text(strip=True))

        description = ""
        desc_el = soup.find("div", class_="page-abstract-content")
        if desc_el:
            description = decrypt_text(desc_el.get_text(strip=True))

        # Cover image from JSON-LD
        cover_url = ""
        script_tag = soup.find("script", type="application/ld+json")
        if script_tag and script_tag.string:
            try:
                ld_data = json.loads(script_tag.string)
                images = ld_data.get("image", [])
                if images:
                    cover_url = images[0] if isinstance(images, list) else images
            except json.JSONDecodeError:
                pass

        # Count chapters
        chapters = soup.find_all("div", class_="chapter-item")

        return NovelInfo(
            novel_id=novel_id,
            title=title,
            author=author,
            description=description,
            cover_url=cover_url,
            chapter_count=len(chapters),
            platform="fanqie",
        )

    def get_chapter_list(self, novel_id: str) -> list[ChapterInfo]:
        """Get ordered list of chapters by scraping the novel page."""
        url = f"{_BASE_URL}/page/{novel_id}"
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")

        chapters = []
        chapter_items = soup.find_all("div", class_="chapter-item")

        for i, item in enumerate(chapter_items):
            link = item.find("a")
            if not link:
                continue

            title = decrypt_text(link.get_text(strip=True))
            href = link.get("href", "")
            chapter_id_match = re.search(r"/reader/(\d+)", href)
            if not chapter_id_match:
                continue

            chapter_id = chapter_id_match.group(1)
            chapters.append(ChapterInfo(
                chapter_id=chapter_id,
                title=title,
                sequence=i + 1,
                url=f"{_BASE_URL}{href}" if not href.startswith("http") else href,
            ))

        return chapters

    def download_chapter(self, chapter_id: str) -> ChapterContent:
        """
        Download a chapter's content.
        Primary: HTML scraping with font decryption (more reliable).
        Fallback: API endpoint (may be geo-blocked or deprecated).
        """
        try:
            return self._download_via_html(chapter_id)
        except Exception as e:
            logger.warning(f"HTML download failed for chapter {chapter_id}: {e}. Trying API fallback...")
            return self._download_via_api(chapter_id)

    def _download_via_api(self, chapter_id: str) -> ChapterContent:
        """Download chapter via the snssdk API (returns clean text)."""
        url = _API_URL.format(chapter_id=chapter_id)
        response = self._request(url)
        data = response.json()

        content_data = data.get("data", {})
        title = content_data.get("title", "")
        html_content = content_data.get("content", "")

        if not html_content:
            raise ValueError(f"No content returned for chapter {chapter_id}")

        # Extract text from HTML
        text = self._html_to_text(html_content)

        # Extract images
        images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_content)

        return ChapterContent(
            chapter_id=chapter_id,
            title=title,
            text=text,
            images=images,
        )

    def _download_via_html(self, chapter_id: str) -> ChapterContent:
        """Download chapter by scraping the reader page (requires decryption)."""
        url = f"{_BASE_URL}/reader/{chapter_id}"
        response = self._request(url)
        soup = BeautifulSoup(response.text, "lxml")

        # Find content container
        content_div = (
            soup.select_one(".muye-reader-content")
            or soup.select_one(".muye-reader-content-16")
        )

        if not content_div:
            raise ValueError(f"Could not find content div for chapter {chapter_id}")

        # Extract text from paragraphs
        paragraphs = []
        images = []
        for element in content_div.find_all(["p", "img"]):
            if element.name == "p":
                raw_text = element.get_text()
                if raw_text.strip():
                    paragraphs.append(decrypt_text(raw_text.strip()))
            elif element.name == "img":
                src = element.get("src", "")
                if src:
                    images.append(src)

        # Get title
        title = ""
        title_el = soup.find("h1") or soup.find("div", class_="chapter-title")
        if title_el:
            title = decrypt_text(title_el.get_text(strip=True))

        return ChapterContent(
            chapter_id=chapter_id,
            title=title,
            text="\n".join(paragraphs),
            images=images,
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML content to clean plaintext."""
        # Extract from <article> if present
        article_match = re.search(r"<article>([\s\S]*?)</article>", html)
        if article_match:
            html = article_match.group(1)

        # Replace <p> tags with newlines
        text = re.sub(r"<p\b[^>]*>", "\n", html)
        # Strip all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Decrypt if needed
        if is_encrypted(text):
            text = decrypt_text(text)

        return text

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
