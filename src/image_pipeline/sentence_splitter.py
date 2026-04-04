"""
Sentence splitter for narration scripts.

Splits episode narration into individual image beats.
Each beat = one image in the final video.
Pure Python, no LLM needed.
"""

from __future__ import annotations

import re


def split_narration(
    episode_text: str,
    min_chars: int = 10,
    max_chars: int = 150,
) -> list[str]:
    """
    Split episode narration into individual sentences for image generation.

    Each returned sentence represents one image beat in the video.

    Args:
        episode_text: Raw episode narration text (may contain [画面：] tags and ---SCENE--- markers).
        min_chars: Minimum characters per beat. Shorter sentences merge with previous.
        max_chars: Maximum characters per beat. Longer sentences get split further.

    Returns:
        List of clean sentence strings, each representing one image.
    """
    # Step 1: Strip visual notes and scene markers
    text = _strip_markers(episode_text)

    # Step 2: Split on sentence boundaries
    raw_sentences = _split_sentences(text)

    # Step 3: Merge short sentences, split long ones
    sentences = _normalize_lengths(raw_sentences, min_chars, max_chars)

    return sentences


def _strip_markers(text: str) -> str:
    """Remove [画面：...] tags, ---SCENE--- markers, and other non-narration content."""
    # Remove visual notes: [画面：...] or [画面:...]
    text = re.sub(r"\[画面[：:][^\]]*\]", "", text)

    # Remove any remaining bracket annotations
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Remove scene markers: ---SCENE---, ---, ===...===
    text = re.sub(r"-{3,}(?:SCENE)?-{0,}", "", text)
    text = re.sub(r"={3,}", "", text)

    # Remove episode headers like "第1集 (章节 1-50)"
    text = re.sub(r"第\d+集\s*\(章节\s*\d+-\d+\)", "", text)

    # Remove markdown formatting that leaked through
    text = re.sub(r"\*\*[^*]+\*\*", "", text)
    text = re.sub(r"###?\s+.+", "", text)

    # Remove parenthetical stage directions: （开场镜头）, （收尾）
    text = re.sub(r"[（(][^）)]*[）)]", "", text)

    # Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", "", text)

    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text on Chinese sentence boundaries."""
    # Split on: 。！？ and newlines
    # Keep the punctuation attached to the sentence
    parts = re.split(r"(?<=[。！？])|(?=\n)", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences


def _normalize_lengths(
    sentences: list[str],
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Merge too-short sentences and split too-long ones."""
    if not sentences:
        return []

    # First pass: merge short sentences with previous
    merged = []
    for sent in sentences:
        if merged and len(sent) < min_chars:
            merged[-1] = merged[-1] + sent
        else:
            merged.append(sent)

    # Second pass: split overly long sentences at comma boundaries
    result = []
    for sent in merged:
        if len(sent) <= max_chars:
            result.append(sent)
        else:
            # Split on commas (，、) trying to keep halves roughly even
            sub_parts = re.split(r"(?<=[，、；])", sent)
            current = ""
            for part in sub_parts:
                if len(current) + len(part) <= max_chars:
                    current += part
                else:
                    if current:
                        result.append(current)
                    current = part
            if current:
                result.append(current)

    # Final cleanup
    return [s.strip() for s in result if s.strip()]
