# src/deepseek_ocr_pdf/grounding.py
"""Parse DeepSeek-OCR grounding responses.

The model emits a label, a bounding box, then the text::

    sub_title[[83, 63, 429, 91]]
    ## ACME CORPORATION

Ollama's template strips DeepSeek's ``<|ref|>``/``<|det|>`` special tokens but
keeps the label and coordinates. Coordinates are normalized 0-999 on each axis
independently, relative to the submitted image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(
    r"^(?P<label>[A-Za-z_][A-Za-z0-9_]*)\[\[\s*"
    r"(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*"
    r"(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\s*\]\]\s*$",
    re.MULTILINE,
)

# A line that opens a block but whose coordinates do not parse. Used only to
# count malformed blocks so the caller can log them.
_BLOCK_LIKE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\[\[[^\]]*\]\]\s*$", re.MULTILINE)

_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_EMPHASIS = re.compile(r"\*\*|__|\*|(?<![A-Za-z0-9])_(?![A-Za-z0-9])")
_MATH_DELIM = re.compile(r"\\[()\[\]]")
_ESCAPED = re.compile(r"\\([$%&_{}#])")


@dataclass(frozen=True)
class Region:
    """One labelled box of text, coordinates still in raw 0-999 space."""

    label: str
    bbox: tuple[int, int, int, int]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ParseResult:
    regions: tuple[Region, ...]
    malformed: int


def strip_markup(text: str) -> str:
    """Remove markdown and LaTeX artifacts the model adds to recognized text.

    The output is an invisible text layer, so markup would be searchable
    noise. v2 emits markdown headings and HTML tables; v1 additionally wrapped
    figures in LaTeX math delimiters.
    """
    text = _HTML_TAG.sub("", text)
    text = _HEADING.sub("", text)
    text = _MATH_DELIM.sub("", text)
    text = _ESCAPED.sub(r"\1", text)
    text = _EMPHASIS.sub("", text)
    return text.strip()


def parse(response: str) -> ParseResult:
    """Turn a raw model response into regions, skipping malformed blocks."""
    matches = list(_BLOCK.finditer(response))
    malformed = len(_BLOCK_LIKE.findall(response)) - len(matches)

    regions: list[Region] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        body = response[start:end]

        lines = tuple(
            stripped
            for raw in body.splitlines()
            if (stripped := strip_markup(raw))
        )
        if not lines:
            continue

        regions.append(
            Region(
                label=match.group("label"),
                bbox=(
                    int(match.group("x1")),
                    int(match.group("y1")),
                    int(match.group("x2")),
                    int(match.group("y2")),
                ),
                lines=lines,
            )
        )

    return ParseResult(regions=tuple(regions), malformed=max(0, malformed))
