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

The ``OCR this image.`` prompt answers in a different shape: no labels, no
markdown, and the box trailing the text it belongs to, one physical line per
box::

    TheCommitteehasreviewed the submitted materials and finds[[87, 165, 897, 183]]
    theconditionsenumeratedinSection4ofthegoverning ordinance.[[86, 188, 834, 206]]

``parse`` reads the first shape, ``parse_lines`` the second. They are never
mixed: each belongs to one prompt.

Ollama's template strips DeepSeek's ``<|ref|>``/``<|det|>`` special tokens but
keeps the label and coordinates. Coordinates are normalized 0-999 on each axis
independently, relative to the submitted image.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
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

#: A line of the ``OCR this image.`` response: text, then its box list. The
#: text is non-greedy so the box list swallows the whole bracketed tail.
_LINE_REF = re.compile(
    rf"^(?P<text>.*?)\[\s*(?P<boxes>{_COORDS}(?:\s*,\s*{_COORDS})*)\s*\]\s*$"
)

#: DeepSeek's grounding tokens. Ollama's current template strips them; another
#: template may not, and they would otherwise end up in the text layer.
_SPECIAL = re.compile(r"<\|/?(?:ref|det)\|>")

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
class LineRef:
    """One physical line the model located, coordinates still in 0-999 space."""

    text: str
    bbox: tuple[int, int, int, int]


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


def share_text(text: str, weights: Sequence[float]) -> list[str]:
    """Split text into one run per weight, in proportion to the weights.

    Used wherever the text of several physical lines arrives as one run and
    their boxes are known separately: each box's width says how much of the
    text belongs to it. Splitting on word boundaries keeps every word whole,
    and each cut lands where the running character count crosses that line's
    cumulative share, so one misjudged word does not shift the rest.
    """
    words = text.split()
    count = len(weights)
    if count <= 1 or len(words) < count:
        return [text]

    total_weight = sum(weights)
    if total_weight <= 0:
        return [text]

    total_chars = sum(len(word) for word in words)
    # Cumulative character offset at which each line but the last should end.
    cuts = []
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        cuts.append(running / total_weight * total_chars)

    runs: list[list[str]] = [[] for _ in weights]
    index = consumed = 0
    for word in words:
        # A word goes to whichever line holds most of its characters.
        while index < count - 1 and consumed + len(word) / 2 > cuts[index]:
            index += 1
        runs[index].append(word)
        consumed += len(word)

    if all(runs):
        return [" ".join(run) for run in runs]

    # One long word can swallow a whole line's share and strand the next line.
    # Splitting by word count instead keeps every box holding something.
    step = len(words) / count
    return [
        " ".join(words[round(i * step) : round((i + 1) * step)])
        for i in range(count)
    ]


def parse_lines(response: str) -> tuple[LineRef, ...]:
    """Read an ``OCR this image.`` response as one reference per physical line.

    Lines without coordinates are dropped rather than counted as malformed: the
    model prefaces its answer with stray text often enough that logging every
    one would be noise, and this response is only ever used for geometry.
    """
    refs: list[LineRef] = []

    for raw in _SPECIAL.sub("", response).splitlines():
        match = _LINE_REF.match(raw.strip())
        if match is None:
            continue

        text = strip_markup(match.group("text"))
        if not text:
            continue

        boxes = _parse_boxes(match.group("boxes"))
        widths = [box[2] - box[0] for box in boxes]
        for run, box in zip(share_text(text, widths), boxes):
            if run:
                refs.append(LineRef(text=run, bbox=box))

    return tuple(refs)
