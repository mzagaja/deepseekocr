# src/deepseek_ocr_pdf/plugin.py
"""ocrmypdf plugin exposing DeepSeek-OCR-2 as an OCR engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.hocrtransform import OcrClass, OcrElement
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image

from deepseek_ocr_pdf import coverage, grounding, layout
from deepseek_ocr_pdf.geometry import Box, covered_fraction
from deepseek_ocr_pdf.layout import GROUNDING_SCALE
from deepseek_ocr_pdf.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OllamaClient,
)

log = logging.getLogger(__name__)

#: A recovered region overlapping existing text by more than this is a
#: duplicate of what the full-page pass already found.
DUPLICATE_THRESHOLD = 0.50

Detector = Callable[[Path, list[str]], list[Box]]


def _regions_from(response: str, page_number: int) -> list[grounding.Region]:
    result = grounding.parse(response)
    if result.malformed:
        log.warning(
            "page %d: skipped %d malformed grounding block(s)",
            page_number + 1,
            result.malformed,
        )
    return list(result.regions)


def shift_into_page(
    region: grounding.Region,
    crop: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> grounding.Region:
    """Re-express a crop-relative grounding box in whole-page 0-999 space.

    Coordinates come back normalized to the crop that was sent, so they must be
    projected onto the crop's position in the page before they mean anything.

    Page dimensions are parameters rather than shared state on purpose:
    ocrmypdf OCRs pages in worker threads, and stashing them anywhere shared
    would let concurrent pages corrupt each other's coordinates.
    """
    crop_left, crop_top, crop_right, crop_bottom = crop
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    x1, y1, x2, y2 = region.bbox

    def to_page(value: int, origin: int, extent: int, page_extent: int) -> int:
        absolute = origin + value / GROUNDING_SCALE * extent
        return int(round(absolute / page_extent * GROUNDING_SCALE))

    return grounding.Region(
        label=region.label,
        bbox=(
            to_page(x1, crop_left, crop_width, page_width),
            to_page(y1, crop_top, crop_height, page_height),
            to_page(x2, crop_left, crop_width, page_width),
            to_page(y2, crop_top, crop_height, page_height),
        ),
        lines=region.lines,
    )


def is_duplicate(box: Box, existing: list[Box]) -> bool:
    """True when a recovered box mostly repeats text already captured.

    Crops are padded, so they can reach back over text the full-page pass
    already read. Without this the overlap would be written twice.
    """
    return covered_fraction(box, existing) > DUPLICATE_THRESHOLD


def _to_grounding(
    box: Box, width_px: int, height_px: int
) -> tuple[int, int, int, int]:
    """Express a pixel box back in the model's 0-999 grounding space."""
    return (
        round(box.left / width_px * GROUNDING_SCALE),
        round(box.top / height_px * GROUNDING_SCALE),
        round(box.right / width_px * GROUNDING_SCALE),
        round(box.bottom / height_px * GROUNDING_SCALE),
    )


def without_covered_lines(
    region: grounding.Region,
    existing: list[Box],
    width_px: int,
    height_px: int,
) -> grounding.Region | None:
    """Drop the lines of a recovered region that repeat text already captured.

    Crops are padded, so a re-read reaches back over neighbouring lines the
    full-page pass already found. Judging the region as a whole keeps those
    repeats whenever the new text outweighs them, and the layer then holds the
    same words twice at overlapping positions -- which extracts as "AXTAX
    YEARYEAR". Deciding line by line keeps the recovery and drops the echo.

    Returns None when nothing new remains.
    """
    kept = [
        (text, box)
        for text, box in layout.region_bands(region, width_px, height_px)
        if not is_duplicate(box, existing)
    ]
    if not kept:
        return None

    boxes = tuple(_to_grounding(box, width_px, height_px) for _, box in kept)
    return grounding.Region(
        label=region.label,
        bbox=(
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ),
        lines=tuple(text for text, _ in kept),
        boxes=boxes,
    )


def ocr_page_image(
    image_path: Path,
    client: OllamaClient,
    detect: Detector,
    languages: list[str],
    dpi: float,
    page_number: int,
) -> tuple[OcrElement, str]:
    """OCR one page image, repairing regions the model dropped.

    Runs exactly one repair pass. Regions still missing afterwards are logged,
    never retried, so a page the model refuses to read cannot loop.
    """
    with Image.open(image_path) as img:
        width, height = img.size

    regions = _regions_from(client.ocr_image(image_path), page_number)

    detected = detect(image_path, languages)
    if detected:
        grounded_boxes = [
            box
            for region in regions
            if (box := layout.rescale(region.bbox, width, height)) is not None
        ]
        missed = coverage.uncovered(detected, grounded_boxes)
        for crop_box in coverage.group_into_crops(missed, width, height):
            crop = (
                int(crop_box.left),
                int(crop_box.top),
                int(crop_box.right),
                int(crop_box.bottom),
            )
            recovered = _regions_from(
                client.ocr_image(image_path, crop=crop), page_number
            )
            for region in recovered:
                shifted = shift_into_page(region, crop, width, height)
                trimmed = without_covered_lines(
                    shifted, grounded_boxes, width, height
                )
                if trimmed is None:
                    continue
                regions.append(trimmed)
                # Record the surviving lines individually, so the next crop is
                # compared against real line geometry rather than one big
                # rectangle spanning the gaps between them.
                grounded_boxes.extend(
                    box
                    for _, box in layout.region_bands(trimmed, width, height)
                )

        still_missing = coverage.uncovered(detected, grounded_boxes)
        for box in still_missing:
            log.warning(
                "page %d: text at (%d, %d)-(%d, %d) not recovered",
                page_number + 1,
                box.left,
                box.top,
                box.right,
                box.bottom,
            )

    page = layout.build_page(regions, width, height, dpi, page_number)
    return page, layout.page_text(page)


def _dpi_of(image_path: Path) -> float:
    with Image.open(image_path) as img:
        info = img.info.get("dpi", (72, 72))
    value = info[0] if isinstance(info, tuple) else info
    return float(value) or 72.0


class DeepSeekOcrEngine(OcrEngine):
    """DeepSeek-OCR-2 running under Ollama, with a Tesseract coverage guard."""

    model = DEFAULT_MODEL
    host = DEFAULT_HOST
    timeout = DEFAULT_TIMEOUT
    max_dim: int | None = None
    coverage_guard = True

    @staticmethod
    def version() -> str:
        return DeepSeekOcrEngine.model

    @staticmethod
    def creator_tag(options) -> str:
        return f"DeepSeek-OCR via Ollama ({DeepSeekOcrEngine.model})"

    def __str__(self) -> str:
        return f"DeepSeek-OCR ({DeepSeekOcrEngine.model})"

    @staticmethod
    def languages(options) -> set[str]:
        # The model is multilingual and publishes no language list, so accept
        # whatever was asked for rather than blocking the run.
        return set(getattr(options, "languages", None) or ["eng"])

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        # The model does not report orientation. Zero confidence tells
        # ocrmypdf this engine has no opinion.
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(
        input_file: Path, options, page_number: int = 0
    ) -> tuple[OcrElement, str]:
        client = OllamaClient(
            host=DeepSeekOcrEngine.host,
            model=DeepSeekOcrEngine.model,
            timeout=DeepSeekOcrEngine.timeout,
            max_dim=DeepSeekOcrEngine.max_dim,
        )
        detect: Detector = (
            coverage.detect_lines
            if DeepSeekOcrEngine.coverage_guard
            else (lambda path, langs: [])
        )
        return ocr_page_image(
            input_file,
            client,
            detect=detect,
            languages=list(getattr(options, "languages", None) or []),
            dpi=_dpi_of(input_file),
            page_number=page_number,
        )

    @staticmethod
    def generate_hocr(
        input_file: Path, output_hocr: Path, output_text: Path, options
    ) -> None:
        """Serialize the same tree as hOCR, for --pdf-renderer hocr."""
        page, text = DeepSeekOcrEngine.generate_ocr(input_file, options)
        output_hocr.write_text(_to_hocr(page), encoding="utf-8")
        output_text.write_text(text, encoding="utf-8")

    @staticmethod
    def generate_pdf(
        input_file: Path, output_pdf: Path, output_text: Path, options
    ) -> None:
        raise NotImplementedError(
            "DeepSeekOcrEngine does not render PDFs directly. "
            "Use --pdf-renderer fpdf2 (the default) instead of sandwich."
        )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _to_hocr(page: OcrElement) -> str:
    """Minimal hOCR serialization of a page tree."""

    def bbox_attr(element: OcrElement) -> str:
        box = element.bbox
        return (
            f"bbox {int(box.left)} {int(box.top)} "
            f"{int(box.right)} {int(box.bottom)}"
        )

    body = []
    for line in page.iter_by_class(*OcrClass.LINE_TYPES):
        words = "".join(
            f"<span class='ocrx_word' title='{bbox_attr(word)}'>"
            f"{_escape(word.text)}</span> "
            for word in line.children
        )
        body.append(
            f"<span class='{line.ocr_class}' title='{bbox_attr(line)}'>"
            f"{words}</span>"
        )

    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<html xmlns='http://www.w3.org/1999/xhtml'>\n"
        "<head><meta http-equiv='Content-Type' "
        "content='text/html;charset=utf-8'/>\n"
        "<meta name='ocr-system' content='deepseek-ocr-pdf'/></head>\n"
        f"<body><div class='ocr_page' title='{bbox_attr(page)}'>"
        f"<p class='ocr_par'>{''.join(body)}</p></div></body>\n</html>\n"
    )


@hookimpl
def get_ocr_engine(options):
    return DeepSeekOcrEngine()
