# tests/test_integration.py
"""End-to-end tests. Require a running Ollama with the model pulled."""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from deepseek_ocr_pdf.ollama_client import DEFAULT_HOST, DEFAULT_MODEL
from tests.make_page import (
    FOOTER_PHONE,
    build_page_pdf,
    build_prose_pdf,
    prose_lines,
)

# Resolve CLI to .venv/bin/deepseek-ocr-pdf when not on PATH (pytest without venv activation)
_CLI = Path(__file__).parent.parent / ".venv" / "bin" / "deepseek-ocr-pdf"
CLI = str(_CLI) if _CLI.exists() else "deepseek-ocr-pdf"

pytestmark = pytest.mark.slow


def _ollama_has_model() -> bool:
    try:
        with urllib.request.urlopen(f"{DEFAULT_HOST}/api/tags", timeout=2) as response:
            return DEFAULT_MODEL.split(":")[0] in response.read().decode()
    except (urllib.error.URLError, OSError):
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_has_model(),
    reason=f"needs Ollama at {DEFAULT_HOST} with {DEFAULT_MODEL} pulled",
)


def _pdf_text(path) -> str:
    return subprocess.run(
        ["pdftotext", str(path), "-"], capture_output=True, text=True, check=True
    ).stdout


@needs_ollama
def test_footer_survives_the_coverage_guard(tmp_path):
    """The measured failure: a footer after a blank gap is dropped.

    Without the coverage guard this text is silently absent from the output.
    """
    source = build_page_pdf(tmp_path / "statement.pdf")
    output = tmp_path / "out.pdf"

    result = subprocess.run(
        [
            CLI,
            str(source),
            str(output),
            "--output-type",
            "pdf",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    text = _pdf_text(output)
    assert FOOTER_PHONE in text, "coverage guard failed to recover the footer"


@needs_ollama
def test_body_text_and_figures_are_present(tmp_path):
    source = build_page_pdf(tmp_path / "statement.pdf")
    output = tmp_path / "out.pdf"

    subprocess.run(
        [CLI, str(source), str(output), "--output-type", "pdf"],
        capture_output=True,
        text=True,
        check=True,
    )

    text = _pdf_text(output)
    assert "ACME CORPORATION" in text
    assert "13,930.21" in text
    assert "Matthew Zagaja" in text


@needs_ollama
def test_word_positions_track_the_source_layout(tmp_path):
    """Words drawn near the top must land near the top of the text layer."""
    source = build_page_pdf(tmp_path / "statement.pdf")
    output = tmp_path / "out.pdf"
    subprocess.run(
        [CLI, str(source), str(output), "--output-type", "pdf"],
        capture_output=True,
        text=True,
        check=True,
    )

    layout_text = subprocess.run(
        ["pdftotext", "-layout", str(output), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    non_empty = [line for line in layout_text if line.strip()]
    assert "ACME" in non_empty[0]
    assert FOOTER_PHONE in non_empty[-1]


@needs_ollama
def test_a_wrapped_paragraph_selects_one_line_at_a_time(tmp_path):
    """The reason the line pass exists.

    Without it the markdown prompt returns each paragraph as one run of text
    under one box, and the layer offers the whole block as a single selection.
    """
    source = build_prose_pdf(tmp_path / "decision.pdf")
    output = tmp_path / "out.pdf"
    sidecar = tmp_path / "out.txt"
    subprocess.run(
        [CLI, str(source), str(output), "--sidecar", str(sidecar)],
        capture_output=True,
        text=True,
        check=True,
    )

    printed = prose_lines()
    # The two headings and anything the guard recovers are shorter than any
    # line of the body, so length alone separates the prose out.
    body = [line for line in sidecar.read_text().splitlines() if len(line) > 40]

    # One line of tolerance: the model splits this page the same way run after
    # run, but the count is its judgement, not ours, and one merged line would
    # not mean the layer had collapsed.
    assert len(printed) - 1 <= len(body) <= len(printed) + 1
    assert body[0].startswith("The Committee")
    assert body[-1].endswith("Town Clerk.")
    # The real check: no line may hold a whole paragraph. A collapsed layer
    # puts all 400-odd characters of one on a single run.
    assert max(len(line) for line in body) < len(max(printed, key=len)) + 25


@needs_ollama
def test_the_detector_carries_line_geometry_without_the_line_pass(tmp_path):
    """`--no-line-split-pass` costs precision, not line-granular selection.

    Tesseract's boxes stand in for the model's, at half the time and with the
    occasional break landing a word off. Only with the coverage guard off as
    well does the paragraph collapse back into one run per paragraph.
    """
    source = build_prose_pdf(tmp_path / "decision.pdf")

    def body(name, *extra):
        sidecar = tmp_path / f"{name}.txt"
        subprocess.run(
            [CLI, str(source), str(tmp_path / f"{name}.pdf"),
             "--sidecar", str(sidecar), *extra],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in sidecar.read_text().splitlines() if len(line) > 40]

    printed = prose_lines()
    detector_only = body("detector", "--no-line-split-pass")
    neither = body("neither", "--no-line-split-pass", "--no-coverage-guard")

    assert len(printed) - 1 <= len(detector_only) <= len(printed) + 1
    assert len(neither) * 2 < len(printed)
