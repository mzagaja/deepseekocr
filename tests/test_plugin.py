# tests/test_plugin.py
from pathlib import Path

import pytest
from ocrmypdf.hocrtransform import OcrClass
from PIL import Image

from deepseek_ocr_pdf.geometry import Box
from deepseek_ocr_pdf.grounding import Region
from deepseek_ocr_pdf.ollama_client import GROUNDING_PROMPT, LINE_GROUNDING_PROMPT
from deepseek_ocr_pdf.plugin import (
    DeepSeekOcrEngine,
    is_duplicate,
    ocr_page_image,
    shift_into_page,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def page_image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (1314, 1700), "white").save(path)
    return path


class FakeClient:
    """Records calls and replays scripted responses.

    The line-grounding pass is scripted separately from the markdown pass
    because it uses its own prompt, so a test that only cares about one of
    them does not have to script around the other.
    """

    def __init__(self, responses, lines=""):
        self.responses = list(responses)
        self.lines = lines
        self.crops = []
        self.prompts = []

    def ocr_image(self, image_path, crop=None, prompt=GROUNDING_PROMPT):
        self.prompts.append(prompt)
        if prompt == LINE_GROUNDING_PROMPT:
            return self.lines
        self.crops.append(crop)
        return self.responses.pop(0) if self.responses else ""


def test_engine_reports_generate_ocr_support():
    assert DeepSeekOcrEngine.supports_generate_ocr() is True


def test_orientation_is_reported_with_zero_confidence(page_image):
    result = DeepSeekOcrEngine.get_orientation(page_image, options=None)
    assert result.angle == 0
    assert result.confidence == 0.0


def test_sandwich_renderer_is_refused(tmp_path):
    with pytest.raises(NotImplementedError) as excinfo:
        DeepSeekOcrEngine.generate_pdf(
            tmp_path / "in.png", tmp_path / "out.pdf", tmp_path / "out.txt", None
        )
    assert "sandwich" in str(excinfo.value)


def test_page_builds_from_a_single_grounding_pass(page_image):
    client = FakeClient([(FIXTURES / "v2_statement.txt").read_text()])
    page, text = ocr_page_image(
        page_image, client, detect=lambda path, langs: [], languages=[], dpi=200.0,
        page_number=0,
    )
    assert len(page.iter_by_class(*OcrClass.LINE_TYPES)) == 13
    assert "ACME CORPORATION" in text


def test_missed_region_triggers_one_repair_crop(page_image):
    # Grounding finds nothing; the detector finds a line near the page bottom.
    client = FakeClient(["", "text[[100, 400, 900, 600]]\nrecovered footer"])
    detected = [Box(150, 1450, 900, 1480)]

    page, text = ocr_page_image(
        page_image, client, detect=lambda path, langs: detected, languages=[],
        dpi=200.0, page_number=0,
    )

    assert client.crops[0] is None          # full page first
    assert client.crops[1] is not None      # then a crop
    assert "recovered" in text


def test_repair_coordinates_land_inside_the_crop(page_image):
    client = FakeClient(["", "text[[0, 0, 999, 999]]\nfooter"])
    detected = [Box(150, 1450, 900, 1480)]

    page, _ = ocr_page_image(
        page_image, client, detect=lambda path, langs: detected, languages=[],
        dpi=200.0, page_number=0,
    )

    crop = client.crops[1]
    line = page.iter_by_class(*OcrClass.LINE_TYPES)[0]
    # Quantization through 0-999 integer space introduces ~1px rounding
    assert crop[0] - 1 <= line.bbox.left <= crop[2] + 1
    assert crop[1] - 1 <= line.bbox.top <= crop[3] + 1


def test_repair_runs_only_once(page_image):
    # Every pass returns nothing, so a recursive design would loop forever.
    client = FakeClient(["", "", "", ""])
    detected = [Box(150, 1450, 900, 1480)]

    ocr_page_image(
        page_image, client, detect=lambda path, langs: detected, languages=[],
        dpi=200.0, page_number=0,
    )
    assert len(client.crops) == 2


def test_shift_maps_crop_coordinates_into_page_space():
    # A box filling a crop that occupies the bottom-left quarter of a
    # 1000x1000 page must come back as the bottom-left quarter of the page.
    region = Region("text", (0, 0, 999, 999), ("footer",))
    shifted = shift_into_page(region, (0, 500, 500, 1000), 1000, 1000)
    # 500/1000*999 = 499.5 rounds to 500
    assert shifted.bbox == (0, 500, 500, 999)


def test_shift_preserves_label_and_text():
    region = Region("sub_title", (10, 10, 900, 900), ("Heading",))
    shifted = shift_into_page(region, (0, 0, 100, 100), 1000, 1000)
    assert shifted.label == "sub_title"
    assert shifted.lines == ("Heading",)


def test_overlapping_recovery_is_flagged_duplicate():
    existing = [Box(0, 0, 100, 100)]
    assert is_duplicate(Box(10, 10, 90, 90), existing) is True


def test_disjoint_recovery_is_not_duplicate():
    existing = [Box(0, 0, 100, 100)]
    assert is_duplicate(Box(500, 500, 600, 600), existing) is False


def test_recovered_region_keeps_only_its_new_lines():
    """A padded crop re-reads neighbours; only the new lines should survive.

    Judging a recovered region as a whole keeps its repeated lines whenever the
    new text outweighs them, which is how "TAX YEAR" lands in the text layer
    twice and extracts as "AXTAX YEARYEAR".
    """
    from deepseek_ocr_pdf.grounding import Region
    from deepseek_ocr_pdf.plugin import without_covered_lines

    # Two lines: the first sits on top of text already captured, the second is new.
    region = Region(
        "text",
        (0, 0, 999, 400),
        ("already read", "genuinely new"),
        boxes=((0, 0, 999, 200), (0, 200, 999, 400)),
    )
    existing = [Box(0, 0, 1000, 200)]

    trimmed = without_covered_lines(region, existing, 1000, 1000)

    assert trimmed is not None
    assert trimmed.lines == ("genuinely new",)


def test_fully_covered_region_is_dropped_entirely():
    from deepseek_ocr_pdf.grounding import Region
    from deepseek_ocr_pdf.plugin import without_covered_lines

    region = Region(
        "text",
        (0, 0, 999, 200),
        ("already read",),
        boxes=((0, 0, 999, 200),),
    )
    existing = [Box(0, 0, 1000, 1000)]

    assert without_covered_lines(region, existing, 1000, 1000) is None


def test_wholly_new_region_survives_intact():
    from deepseek_ocr_pdf.grounding import Region
    from deepseek_ocr_pdf.plugin import without_covered_lines

    region = Region(
        "text",
        (0, 600, 999, 800),
        ("footer nobody read",),
        boxes=((0, 600, 999, 800),),
    )
    trimmed = without_covered_lines(region, [Box(0, 0, 1000, 100)], 1000, 1000)

    assert trimmed is not None
    assert trimmed.lines == ("footer nobody read",)


def test_line_pass_splits_a_reflowed_paragraph(page_image):
    """The whole point: one selectable run per printed line, not per paragraph."""
    markdown = "text[[100, 100, 900, 400]]\naaa bbb ccc ddd eee fff"
    lines = (
        "aaa bbb[[100, 100, 900, 200]]\n"
        "ccc ddd[[100, 200, 900, 300]]\n"
        "eee fff[[100, 300, 900, 400]]"
    )
    client = FakeClient([markdown], lines=lines)
    page, _ = ocr_page_image(
        page_image, client, detect=lambda path, langs: [], languages=[], dpi=200.0,
        page_number=0,
    )

    bands = page.iter_by_class(*OcrClass.LINE_TYPES)
    assert [band.text for band in bands] == ["aaa bbb", "ccc ddd", "eee fff"]
    assert bands[0].bbox.bottom == pytest.approx(bands[1].bbox.top, abs=1.0)


def test_line_pass_runs_once_per_page(page_image):
    client = FakeClient([""], lines="")
    ocr_page_image(
        page_image, client, detect=lambda path, langs: [], languages=[], dpi=200.0,
        page_number=0,
    )
    assert client.prompts.count(LINE_GROUNDING_PROMPT) == 1


def test_line_pass_can_be_switched_off(page_image):
    client = FakeClient([""])
    ocr_page_image(
        page_image, client, detect=lambda path, langs: [], languages=[], dpi=200.0,
        page_number=0, line_split_pass=False,
    )
    assert LINE_GROUNDING_PROMPT not in client.prompts


def test_paragraph_survives_an_empty_line_pass(page_image):
    """A refusal on the geometry pass must not cost us the text."""
    markdown = "text[[100, 100, 900, 400]]\naaa bbb ccc"
    client = FakeClient([markdown], lines="")
    page, text = ocr_page_image(
        page_image, client, detect=lambda path, langs: [], languages=[], dpi=200.0,
        page_number=0,
    )
    assert text == "aaa bbb ccc"


def test_shift_preserves_per_line_boxes():
    region = Region(
        "text", (0, 0, 999, 999), ("top", "bottom"),
        boxes=((0, 0, 999, 400), (0, 500, 999, 999)),
    )
    shifted = shift_into_page(region, (0, 0, 1000, 1000), 1000, 1000)
    assert shifted.boxes == region.boxes


def test_detector_fills_a_gap_the_line_pass_left():
    """The line pass drops lines too; the detector's boxes cover for it."""
    from deepseek_ocr_pdf.layout import LineSpan
    from deepseek_ocr_pdf.plugin import merge_line_spans

    spans = [LineSpan(Box(0, 0, 100, 20), 8)]
    detected = [Box(0, 100, 100, 120)]
    merged = merge_line_spans(spans, detected)

    assert [span.box for span in merged] == [Box(0, 0, 100, 20), Box(0, 100, 100, 120)]
    # Tesseract's text is discarded, so the filled line is weighted by width.
    assert merged[1].chars == 0


def test_detector_does_not_double_a_line_the_model_measured():
    from deepseek_ocr_pdf.layout import LineSpan
    from deepseek_ocr_pdf.plugin import merge_line_spans

    spans = [LineSpan(Box(0, 0, 100, 20), 8)]
    merged = merge_line_spans(spans, [Box(2, 1, 98, 19)])

    assert len(merged) == 1


def test_nothing_detected_leaves_the_spans_alone():
    from deepseek_ocr_pdf.layout import LineSpan
    from deepseek_ocr_pdf.plugin import merge_line_spans

    spans = [LineSpan(Box(0, 0, 100, 20), 8)]
    assert merge_line_spans(spans, []) == spans
