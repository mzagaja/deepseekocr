# src/deepseek_ocr_pdf/coverage.py
"""Detect text the vision model dropped.

DeepSeek-OCR-2 terminates early when it reaches an isolated element after a
large blank region, silently omitting it. Measured on 2026-08-24: a footer at
86% page height was dropped on every full-page run at both Q4_K_M and Q8_0,
with done_reason "stop" and no truncation, yet read correctly from a tight crop.

Tesseract is run purely as a detector here. Its recognized text is discarded;
only the geometry is used, so its own OCR quality is irrelevant.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from statistics import median

from deepseek_ocr_pdf.geometry import Box, covered_fraction

log = logging.getLogger(__name__)

#: A detected line counts as missed when less than this fraction of its area
#: overlaps a grounding box.
UNCOVERED_THRESHOLD = 0.30

#: Crops are padded by this fraction of the page's larger dimension. The model
#: reads a tight crop reliably but needs a little surrounding context.
CROP_PADDING_RATIO = 0.02

_WORD_LEVEL = "5"


def parse_tsv(tsv: str) -> list[Box]:
    """Group Tesseract TSV word rows into line boxes.

    Rows are split on tabs with a column limit so that a tab inside the
    recognized text cannot shift the numeric columns.
    """
    lines: dict[tuple[str, str, str], Box] = {}

    for row in tsv.splitlines():
        parts = row.split("\t", 11)
        if len(parts) < 12 or parts[0] != _WORD_LEVEL:
            continue

        _, _, block, par, line_num, _, left, top, width, height, conf, text = parts
        if not text.strip():
            continue
        try:
            if float(conf) < 0:
                continue
            box = Box(
                float(left),
                float(top),
                float(left) + float(width),
                float(top) + float(height),
            )
        except ValueError:
            continue

        key = (block, par, line_num)
        lines[key] = lines[key].union(box) if key in lines else box

    return list(lines.values())


def detect_lines(image_path: Path, languages: list[str]) -> list[Box]:
    """Run Tesseract in TSV mode and return line boxes.

    Returns an empty list if Tesseract is missing or fails, which disables the
    coverage guard rather than failing the run.
    """
    language_arg = "+".join(languages) if languages else "eng"
    try:
        completed = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", language_arg, "tsv"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        log.warning("tesseract not found; coverage guard disabled")
        return []
    except subprocess.CalledProcessError as exc:
        log.warning("tesseract detection failed: %s", exc.stderr.strip()[:200])
        return []

    return parse_tsv(completed.stdout)


def uncovered(detected: list[Box], grounded: list[Box]) -> list[Box]:
    """Detected line boxes that the grounding pass largely missed."""
    return [
        box
        for box in detected
        if covered_fraction(box, grounded) < UNCOVERED_THRESHOLD
    ]


def group_into_crops(boxes: list[Box], width_px: int, height_px: int) -> list[Box]:
    """Cluster missed boxes vertically and pad each cluster into a crop.

    Boxes separated by less than twice the median line height belong to the
    same block of text and are re-read in one pass rather than one call each.
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: (b.top, b.left))
    gap_limit = 2 * median([b.height for b in ordered if b.height > 0] or [1.0])

    clusters: list[Box] = [ordered[0]]
    for box in ordered[1:]:
        if box.top - clusters[-1].bottom < gap_limit:
            clusters[-1] = clusters[-1].union(box)
        else:
            clusters.append(box)

    pad = CROP_PADDING_RATIO * max(width_px, height_px)
    return [
        Box(
            left=max(0.0, cluster.left - pad),
            top=max(0.0, cluster.top - pad),
            right=min(float(width_px), cluster.right + pad),
            bottom=min(float(height_px), cluster.bottom + pad),
        )
        for cluster in clusters
    ]
