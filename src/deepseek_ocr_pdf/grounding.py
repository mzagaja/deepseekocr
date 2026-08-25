# src/deepseek_ocr_pdf/grounding.py
"""Parse DeepSeek-OCR grounding responses.

The model emits a label, a bounding box, then the text::

    sub_title[[83, 63, 429, 91]]
    ## ACME CORPORATION

A label may carry several boxes, one per line, when the model groups
consecutive lines under one reference::

    text[[60, 381, 611, 415], [75, 413, 340, 449]]
    Extra Services & Fees
    Return Receipt (hardcopy)

Ollama's template strips DeepSeek's ``<|ref|>``/``<|det|>`` special tokens but
keeps the label and coordinates. Coordinates are normalized 0-999 on each axis
independently, relative to the submitted image.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

#: One ``[x1, y1, x2, y2]`` group. Whitespace and newlines inside are tolerated
#: because the model wraps long box lists.
_COORDS = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"

_BLOCK = re.compile(
    r"^(?P<label>[A-Za-z_][A-Za-z0-9_]*)\[\s*"
    rf"(?P<boxes>{_COORDS}(?:\s*,\s*{_COORDS})*)"
    r"\s*\]\s*$",
    re.MULTILINE,
)

# A line that opens a block but whose coordinates do not parse. Used only to
# count malformed blocks so the caller can log them. The body is allowed to
# contain brackets so that a malformed multi-box list is still counted.
_BLOCK_LIKE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\[\[.*?\]\]\s*$", re.MULTILINE | re.DOTALL
)

_ONE_BOX = re.compile(r"-?\d+")

_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
_SPACES = re.compile(r"[ \t]{2,}")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_EMPHASIS = re.compile(r"\*\*|__|\*|(?<![A-Za-z0-9])_(?![A-Za-z0-9])")
_MATH_DELIM = re.compile(r"\\[()\[\]]")
_ESCAPED = re.compile(r"\\([$%&_{}#])")


@dataclass(frozen=True)
class Region:
    """One labelled reference, coordinates still in raw 0-999 space.

    ``boxes`` holds every box the model attached to the label, in order.
    ``bbox`` is their union, kept for callers that want a single rectangle.
    """

    label: str
    bbox: tuple[int, int, int, int]
    lines: tuple[str, ...]
    boxes: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class ParseResult:
    regions: tuple[Region, ...]
    malformed: int


def strip_markup(text: str) -> str:
    """Remove markdown and LaTeX artifacts the model adds to recognized text.

    The output is an invisible text layer, so markup would be searchable
    noise. v2 emits markdown headings and HTML tables; v1 additionally wrapped
    figures in LaTeX math delimiters.

    Tags become a space rather than nothing. A table's cells are adjacent in
    the markup but separate words on the page, so deleting ``</td><td>``
    outright welds them into one token and hides both from any search that
    respects word boundaries. Entities are decoded last, after tag removal,
    so that a decoded ``&lt;`` is never mistaken for the start of a tag.
    """
    text = _HTML_TAG.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _MATH_DELIM.sub("", text)
    text = _ESCAPED.sub(r"\1", text)
    text = _EMPHASIS.sub("", text)
    text = html.unescape(text)
    return _SPACES.sub(" ", text).strip()


def _parse_boxes(raw: str) -> tuple[tuple[int, int, int, int], ...]:
    """Split a matched box list into individual coordinate tuples."""
    return tuple(
        tuple(int(n) for n in _ONE_BOX.findall(group))  # type: ignore[misc]
        for group in re.findall(_COORDS, raw)
    )


def _union(boxes: tuple[tuple[int, int, int, int], ...]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


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

        boxes = _parse_boxes(match.group("boxes"))
        if not boxes:
            continue

        regions.append(
            Region(
                label=match.group("label"),
                bbox=_union(boxes),
                lines=lines,
                boxes=boxes,
            )
        )

    return ParseResult(regions=tuple(regions), malformed=max(0, malformed))
