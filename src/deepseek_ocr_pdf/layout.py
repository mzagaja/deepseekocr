# src/deepseek_ocr_pdf/layout.py
"""Turn parsed grounding regions into an ocrmypdf OcrElement tree."""

from __future__ import annotations

from collections.abc import Sequence

from ocrmypdf.hocrtransform import BoundingBox, OcrClass, OcrElement

from deepseek_ocr_pdf.geometry import Box
from deepseek_ocr_pdf.grounding import Region

#: DeepSeek-OCR normalizes coordinates to 0-999 on each axis independently.
GROUNDING_SCALE = 999


def rescale(
    bbox: tuple[int, int, int, int], width_px: int, height_px: int
) -> Box | None:
    """Map a raw 0-999 grounding box onto image pixels.

    Returns None for boxes that are inverted or have no area; ocrmypdf's
    BoundingBox raises on those, so they must be dropped rather than passed on.
    """
    x1, y1, x2, y2 = bbox
    left = min(max(x1 / GROUNDING_SCALE * width_px, 0.0), float(width_px))
    right = min(max(x2 / GROUNDING_SCALE * width_px, 0.0), float(width_px))
    top = min(max(y1 / GROUNDING_SCALE * height_px, 0.0), float(height_px))
    bottom = min(max(y2 / GROUNDING_SCALE * height_px, 0.0), float(height_px))

    if right <= left or bottom <= top:
        return None
    return Box(left, top, right, bottom)


def split_lines(box: Box, lines: tuple[str, ...]) -> list[tuple[str, Box]]:
    """Divide a region box into one band per text line.

    Bands are equal-height slices top to bottom. Each band's right edge is
    then pulled in proportionally to its character count against the longest
    line in the region, so a short line inside a wide paragraph box does not
    claim the full width. A single-line region is returned unchanged.
    """
    if not lines:
        return []

    longest = max(len(line) for line in lines)
    band_height = box.height / len(lines)
    bands: list[tuple[str, Box]] = []

    for index, line in enumerate(lines):
        top = box.top + index * band_height
        width_ratio = (len(line) / longest) if longest else 1.0
        bands.append(
            (
                line,
                Box(
                    left=box.left,
                    top=top,
                    right=box.left + box.width * width_ratio,
                    bottom=top + band_height,
                ),
            )
        )
    return bands


def split_words(text: str, band: Box) -> list[tuple[str, Box]]:
    """Divide a line band into one box per word.

    Width is allocated by character count, with one character-width gap
    between words. This is a heuristic: accurate for proportional text,
    slightly off for monospace and unusual kerning. It only affects how
    precisely a viewer highlights a selection, never the text itself.
    """
    words = text.split()
    if not words:
        return []

    units = sum(len(word) for word in words) + (len(words) - 1)
    if units <= 0:
        return []

    per_unit = band.width / units
    result: list[tuple[str, Box]] = []
    cursor = band.left

    for word in words:
        right = cursor + per_unit * len(word)
        result.append((word, Box(cursor, band.top, right, band.bottom)))
        cursor = right + per_unit

    return result


#: Grounding labels that mark non-text page furniture. These carry no text
#: layer; whatever text the model attaches to them is a caption it already
#: emitted separately, or a description it invented.
NON_TEXT_LABELS = frozenset({"image", "figure", "photo"})

_LABEL_TO_CLASS = {
    "title": OcrClass.HEADER,
    "sub_title": OcrClass.HEADER,
    "subtitle": OcrClass.HEADER,
    "header": OcrClass.HEADER,
    "footer": OcrClass.FOOTER,
    "caption": OcrClass.CAPTION,
}


def _to_bbox(box: Box) -> BoundingBox:
    return BoundingBox(
        left=box.left, top=box.top, right=box.right, bottom=box.bottom
    )


def build_page(
    regions: Sequence[Region],
    width_px: int,
    height_px: int,
    dpi: float,
    page_number: int,
) -> OcrElement:
    """Assemble parsed regions into a page tree ocrmypdf can render.

    The tree is page > paragraph > line > word. A single paragraph wrapper is
    used because the grounding output carries no paragraph grouping of its own,
    and the renderer only needs the line and word geometry.
    """
    page = OcrElement(
        ocr_class=OcrClass.PAGE,
        bbox=BoundingBox(left=0, top=0, right=width_px, bottom=height_px),
        dpi=dpi,
        page_number=page_number,
    )
    paragraph = OcrElement(
        ocr_class=OcrClass.PARAGRAPH,
        bbox=BoundingBox(left=0, top=0, right=width_px, bottom=height_px),
    )

    for region in regions:
        if region.label in NON_TEXT_LABELS:
            continue
        box = rescale(region.bbox, width_px, height_px)
        if box is None:
            continue

        line_class = _LABEL_TO_CLASS.get(region.label, OcrClass.LINE)
        for text, band in split_lines(box, region.lines):
            words = split_words(text, band)
            if not words:
                continue
            paragraph.children.append(
                OcrElement(
                    ocr_class=line_class,
                    bbox=_to_bbox(band),
                    text=text,
                    children=[
                        OcrElement(
                            ocr_class=OcrClass.WORD,
                            bbox=_to_bbox(word_box),
                            text=word,
                        )
                        for word, word_box in words
                    ],
                )
            )

    if paragraph.children:
        page.children.append(paragraph)
    return page


def page_text(page: OcrElement) -> str:
    """Plain text of a page, one line per line element, for the sidecar."""
    return "\n".join(
        line.text for line in page.iter_by_class(*OcrClass.LINE_TYPES)
    )
