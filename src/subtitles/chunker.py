"""
Chinese text chunking for subtitles.

Chinese has no word spaces, so we break on punctuation and character count limits.
"""

from __future__ import annotations

import re

# Max characters per subtitle line
MAX_CHARS_PER_LINE = 20
# Punctuation that makes good break points
_BREAK_PUNCT = set("，。！？；：、…—")
# Quotes and brackets to strip or keep with adjacent text
_QUOTES = set('"""\'\'「」『』【】')


def _clean_for_subtitle(text: str) -> str:
    """Clean text for subtitle display — remove orphan quotes and brackets."""
    text = text.strip()
    # Remove standalone quotes/brackets
    for q in _QUOTES:
        text = text.replace(q, "")
    # Clean up multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """
    Split Chinese text into subtitle-sized chunks.

    Breaks at punctuation when possible, otherwise at max_chars boundary.
    """
    text = _clean_for_subtitle(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Look for a punctuation break point within the limit
        best_break = -1
        for i in range(min(max_chars, len(remaining))):
            if remaining[i] in _BREAK_PUNCT:
                best_break = i + 1  # include the punctuation

        if best_break > 0:
            chunks.append(remaining[:best_break])
            remaining = remaining[best_break:].lstrip()
        else:
            # No punctuation found, hard break at max_chars
            chunks.append(remaining[:max_chars])
            remaining = remaining[max_chars:]

    return [c for c in chunks if c.strip()]


def split_into_subtitle_groups(
    text: str,
    max_chars_per_line: int = MAX_CHARS_PER_LINE,
    max_lines: int = 4,
) -> list[str]:
    """
    Split text into subtitle display groups.
    Each group has at most max_lines lines, each at most max_chars_per_line.
    """
    chunks = chunk_text(text, max_chars_per_line)

    groups = []
    for i in range(0, len(chunks), max_lines):
        group = "\n".join(chunks[i:i + max_lines])
        groups.append(group)

    return groups
