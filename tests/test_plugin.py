# tests/test_plugin.py
from pathlib import Path

import pytest
from ocrmypdf.hocrtransform import OcrClass
from PIL import Image

from deepseek_ocr_pdf.geometry import Box
from deepseek_ocr_pdf.grounding import Region
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
    """Records calls and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.crops = []

    def ocr_image(self, image_path, crop=None):
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
