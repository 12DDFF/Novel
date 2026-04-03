"""Tests for the scraper module."""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.scraper.base import BaseScraper, ChapterContent, ChapterInfo, NovelInfo
from src.scraper.font_decryptor import decrypt_char, decrypt_text, is_encrypted
from src.scraper.rate_limiter import RateLimiter


# ── Font Decryptor Tests ─────────────────────────────────────────────────────


class TestFontDecryptor:
    def test_decrypt_normal_char_unchanged(self):
        assert decrypt_char("A") == "A"
        assert decrypt_char("你") == "你"
        assert decrypt_char("1") == "1"

    def test_decrypt_encoded_char(self):
        # U+E3D8 (58344) -> index 0 -> 'D'
        encoded = chr(58344)
        assert decrypt_char(encoded) == "D"

    def test_decrypt_encoded_char_chinese(self):
        # U+E3D9 (58345) -> index 1 -> '在'
        encoded = chr(58345)
        assert decrypt_char(encoded) == "在"

    def test_decrypt_last_char_in_range(self):
        # The charset has 372 entries, last valid index = 371
        encoded = chr(58344 + 371)
        result = decrypt_char(encoded)
        assert isinstance(result, str)
        assert len(result) == 1

    def test_decrypt_out_of_range_unchanged(self):
        # Just above the range
        above = chr(58716)
        assert decrypt_char(above) == above
        # Just below the range
        below = chr(58343)
        assert decrypt_char(below) == below

    def test_decrypt_text_mixed(self):
        # Mix of normal and encoded characters
        normal = "你好"
        encoded_D = chr(58344)  # -> 'D'
        encoded_zai = chr(58345)  # -> '在'
        text = f"{normal}{encoded_D}{encoded_zai}"
        result = decrypt_text(text)
        assert result == "你好D在"

    def test_decrypt_text_no_encoded(self):
        text = "这是一段普通的中文文本"
        assert decrypt_text(text) == text

    def test_decrypt_text_empty(self):
        assert decrypt_text("") == ""

    def test_is_encrypted_true(self):
        text = f"abc{chr(58344)}def"
        assert is_encrypted(text) is True

    def test_is_encrypted_false(self):
        assert is_encrypted("normal text 你好") is False
        assert is_encrypted("") is False


# ── Rate Limiter Tests ───────────────────────────────────────────────────────


class TestRateLimiter:
    def test_first_call_immediate(self):
        limiter = RateLimiter(min_interval=1.0)
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # First call should be instant

    def test_second_call_waits(self):
        limiter = RateLimiter(min_interval=0.2)
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # Should wait ~0.2s

    def test_no_wait_after_interval(self):
        limiter = RateLimiter(min_interval=0.1)
        limiter.wait()
        time.sleep(0.15)  # Wait longer than interval
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should not need to wait


# ── Base Scraper Tests ───────────────────────────────────────────────────────


class TestBaseScraper:
    def test_extract_novel_id_from_page_url(self):
        url = "https://fanqienovel.com/page/7345678901234567"
        assert BaseScraper.extract_novel_id(url) == "7345678901234567"

    def test_extract_novel_id_from_reader_url(self):
        url = "https://fanqienovel.com/reader/7345678901234568"
        assert BaseScraper.extract_novel_id(url) == "7345678901234568"

    def test_extract_novel_id_from_book_id_param(self):
        url = "https://changdunovel.com/page?book_id=12345"
        assert BaseScraper.extract_novel_id(url) == "12345"

    def test_extract_novel_id_plain_number(self):
        assert BaseScraper.extract_novel_id("7345678901234567") == "7345678901234567"

    def test_extract_novel_id_invalid(self):
        with pytest.raises(ValueError, match="Could not extract"):
            BaseScraper.extract_novel_id("https://example.com/not-a-novel")


# ── Fanqie Scraper Tests (with mocked HTTP) ──────────────────────────────────


# Sample HTML responses for mocking
SAMPLE_NOVEL_PAGE = """
<html>
<head>
<script type="application/ld+json">{"image": ["https://example.com/cover.jpg"]}</script>
</head>
<body>
<h1>仙逆</h1>
<span class="author-name-text">耳根</span>
<div class="page-abstract-content">一个资质平庸的修仙少年的故事</div>
<div class="chapter-item"><a href="/reader/100001">第一章 王林</a></div>
<div class="chapter-item"><a href="/reader/100002">第二章 铁皮</a></div>
<div class="chapter-item"><a href="/reader/100003">第三章 修仙</a></div>
</body>
</html>
"""

SAMPLE_API_RESPONSE = {
    "code": 0,
    "data": {
        "title": "第一章 王林",
        "content": '<article><p>王林走进了山洞。</p><p>他看到了一个老人。</p><p>老人微微一笑。</p></article>',
    },
}

SAMPLE_HTML_READER = """
<html>
<body>
<h1>第一章 王林</h1>
<div class="muye-reader-content">
    <p>王林走进了山洞。</p>
    <p>他看到了一个老人。</p>
    <img src="https://example.com/img1.jpg" />
    <p>老人微微一笑。</p>
</div>
</body>
</html>
"""

SAMPLE_SEARCH_RESPONSE = {
    "code": 0,
    "data": {
        "search_book_data_list": [
            {
                "book_id": "7345678901234567",
                "book_name": "仙逆",
                "author": "耳根",
                "abstract": "修仙小说",
                "word_count": "6000000",
            },
            {
                "book_id": "7345678901234568",
                "book_name": "一念永恒",
                "author": "耳根",
                "abstract": "修仙小说2",
                "word_count": "5000000",
            },
        ]
    },
}


class TestFanqieScraper:
    """Tests using mocked HTTP responses."""

    def _mock_response(self, text: str = "", json_data: dict | None = None, status_code: int = 200):
        """Create a mock httpx.Response."""
        mock = MagicMock()
        mock.status_code = status_code
        mock.text = text or (json.dumps(json_data, ensure_ascii=False) if json_data else "")
        mock.json.return_value = json_data
        mock.raise_for_status.return_value = None
        return mock

    @patch("src.scraper.fanqie.httpx.Client")
    def test_get_novel_info(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response(text=SAMPLE_NOVEL_PAGE)
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0)
        scraper._client = mock_client

        info = scraper.get_novel_info("12345")
        assert info.title == "仙逆"
        assert info.author == "耳根"
        assert info.chapter_count == 3
        assert info.cover_url == "https://example.com/cover.jpg"
        assert "修仙少年" in info.description

    @patch("src.scraper.fanqie.httpx.Client")
    def test_get_chapter_list(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response(text=SAMPLE_NOVEL_PAGE)
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0)
        scraper._client = mock_client

        chapters = scraper.get_chapter_list("12345")
        assert len(chapters) == 3
        assert chapters[0].chapter_id == "100001"
        assert chapters[0].title == "第一章 王林"
        assert chapters[0].sequence == 1
        assert chapters[2].chapter_id == "100003"
        assert chapters[2].sequence == 3

    @patch("src.scraper.fanqie.httpx.Client")
    def test_download_chapter_via_api(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response(json_data=SAMPLE_API_RESPONSE)
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0)
        scraper._client = mock_client

        content = scraper._download_via_api("100001")
        assert content.chapter_id == "100001"
        assert content.title == "第一章 王林"
        assert "王林走进了山洞" in content.text
        assert "老人微微一笑" in content.text

    @patch("src.scraper.fanqie.httpx.Client")
    def test_download_chapter_via_html(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response(text=SAMPLE_HTML_READER)
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0)
        scraper._client = mock_client

        content = scraper._download_via_html("100001")
        assert content.title == "第一章 王林"
        assert "王林走进了山洞" in content.text
        assert "老人微微一笑" in content.text
        assert "https://example.com/img1.jpg" in content.images

    @patch("src.scraper.fanqie.httpx.Client")
    def test_search(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response(json_data=SAMPLE_SEARCH_RESPONSE)
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0)
        scraper._client = mock_client

        results = scraper.search("仙逆")
        assert len(results) == 2
        assert results[0].title == "仙逆"
        assert results[0].author == "耳根"
        assert results[0].novel_id == "7345678901234567"

    @patch("src.scraper.fanqie.httpx.Client")
    def test_download_chapter_html_fallback_to_api(self, mock_client_class):
        """When HTML fails, should fall back to API."""
        mock_client = MagicMock()

        # First call (HTML) returns bad page, second call (API) succeeds
        bad_html_response = self._mock_response(text="<html><body>no content here</body></html>")
        api_response = self._mock_response(json_data=SAMPLE_API_RESPONSE)

        mock_client.get.side_effect = [bad_html_response, api_response]
        mock_client_class.return_value = mock_client

        from src.scraper.fanqie import FanqieScraper
        scraper = FanqieScraper(rate_limit=0.0, max_retries=1)
        scraper._client = mock_client

        content = scraper.download_chapter("100001")
        assert "王林走进了山洞" in content.text

    def test_html_to_text(self):
        from src.scraper.fanqie import FanqieScraper
        html = '<article><p>第一段</p><p>第二段</p><p>第三段</p></article>'
        text = FanqieScraper._html_to_text(html)
        assert "第一段" in text
        assert "第二段" in text
        assert "<p>" not in text
        assert "<article>" not in text

    def test_html_to_text_no_article(self):
        from src.scraper.fanqie import FanqieScraper
        html = '<p>段落一</p><p>段落二</p>'
        text = FanqieScraper._html_to_text(html)
        assert "段落一" in text
        assert "<p>" not in text

    def test_context_manager(self):
        from src.scraper.fanqie import FanqieScraper
        with FanqieScraper(rate_limit=0.0) as scraper:
            assert scraper is not None
