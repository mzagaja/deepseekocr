# tests/test_layout.py
from pathlib import Path

import pytest
from ocrmypdf.hocrtransform import OcrClass

from deepseek_ocr_pdf.geometry import Box
from deepseek_ocr_pdf.grounding import Region
from deepseek_ocr_pdf.layout import GROUNDING_SCALE, build_page, rescale, split_lines, split_words

FIXTURES = Path(__file__).parent / "fixtures"


def test_grounding_scale_is_999():
    assert GROUNDING_SCALE == 999


def test_rescale_matches_measured_case():
    # Measured on this machine: an 800x400 image returned [74, 152, 543, 238]
    # for text drawn at (60, 60). Each axis is normalized independently.
    box = rescale((74, 152, 543, 238), 800, 400)
    assert box.left == pytest.approx(59.3, abs=0.5)
    assert box.top == pytest.approx(60.9, abs=0.5)
    assert box.right == pytest.approx(434.8, abs=0.5)
    assert box.bottom == pytest.approx(95.3, abs=0.5)


def test_rescale_full_extent_maps_to_full_image():
    box = rescale((0, 0, 999, 999), 1000, 500)
    assert box.left == 0
    assert box.top == 0
    assert box.right == pytest.approx(1000)
    assert box.bottom == pytest.approx(500)


def test_rescale_clamps_out_of_range_coordinates():
    box = rescale((-20, -20, 1200, 1200), 100, 100)
    assert box.left == 0
    assert box.top == 0
    assert box.right == 100
    assert box.bottom == 100


def test_rescale_returns_none_for_inverted_box():
    # BoundingBox raises on inverted input, so these must be filtered earlier.
    assert rescale((500, 10, 100, 20), 100, 100) is None


def test_rescale_returns_none_for_zero_area_box():
    assert rescale((10, 10, 10, 400), 100, 100) is None


def test_single_line_region_is_unchanged():
    bands = split_lines(Box(0, 0, 100, 20), ("hello",))
    assert len(bands) == 1
    assert bands[0][0] == "hello"
    assert bands[0][1] == Box(0, 0, 100, 20)


def test_three_lines_become_three_equal_bands():
    bands = split_lines(Box(0, 0, 100, 30), ("aaa", "bbb", "ccc"))
    assert [b[1].top for b in bands] == [0, 10, 20]
    assert [b[1].bottom for b in bands] == [10, 20, 30]


def test_short_line_does_not_claim_full_width():
    # "aaaaaaaaaa" is 10 chars, "bb" is 2, so the short band is 20% as wide.
    bands = split_lines(Box(0, 0, 100, 20), ("aaaaaaaaaa", "bb"))
    assert bands[0][1].right == 100
    assert bands[1][1].right == 20


def test_band_left_edge_is_preserved():
    bands = split_lines(Box(30, 0, 100, 20), ("aaaa", "bb"))
    assert all(band[1].left == 30 for band in bands)


def test_empty_lines_produce_no_bands():
    assert split_lines(Box(0, 0, 100, 20), ()) == []


def test_single_word_fills_the_band():
    words = split_words("hello", Box(0, 0, 100, 10))
    assert len(words) == 1
    assert words[0][0] == "hello"
    assert words[0][1].left == 0
    assert words[0][1].right == pytest.approx(100)


def test_words_are_ordered_left_to_right_without_overlap():
    words = split_words("alpha beta gamma", Box(0, 0, 160, 10))
    lefts = [w[1].left for w in words]
    assert lefts == sorted(lefts)
    for earlier, later in zip(words, words[1:]):
        assert earlier[1].right <= later[1].left + 1e-9


def test_word_widths_are_proportional_to_length():
    # "aa" and "bbbb": 2 and 4 chars, plus one space unit = 7 units total.
    words = split_words("aa bbbb", Box(0, 0, 70, 10))
    assert words[0][1].width == pytest.approx(20)
    assert words[1][1].width == pytest.approx(40)


def test_last_word_ends_at_the_band_edge():
    words = split_words("one two three", Box(10, 0, 110, 10))
    assert words[-1][1].right == pytest.approx(110)


def test_words_inherit_band_vertical_extent():
    words = split_words("a b", Box(0, 5, 100, 25))
    assert all(w[1].top == 5 and w[1].bottom == 25 for w in words)


def test_runs_of_whitespace_collapse():
    assert len(split_words("a    b", Box(0, 0, 100, 10))) == 2


def test_blank_text_yields_no_words():
    assert split_words("   ", Box(0, 0, 100, 10)) == []


def test_page_root_carries_size_and_dpi():
    page = build_page([], width_px=800, height_px=400, dpi=200.0, page_number=3)
    assert page.ocr_class == OcrClass.PAGE
    assert page.bbox.right == 800
    assert page.bbox.bottom == 400
    assert page.dpi == 200.0
    assert page.page_number == 3


def test_text_region_becomes_a_line_with_words():
    regions = [Region("text", (0, 0, 999, 99), ("hello world",))]
    page = build_page(regions, 1000, 1000, 72.0, 0)
    lines = page.iter_by_class(OcrClass.LINE)
    assert len(lines) == 1
    words = page.iter_by_class(OcrClass.WORD)
    assert [w.text for w in words] == ["hello", "world"]


def test_sub_title_maps_to_header_class():
    regions = [Region("sub_title", (0, 0, 999, 99), ("Title",))]
    page = build_page(regions, 1000, 1000, 72.0, 0)
    assert len(page.iter_by_class(OcrClass.HEADER)) == 1


def test_caption_maps_to_caption_class():
    regions = [Region("caption", (0, 0, 999, 99), ("Figure 1",))]
    page = build_page(regions, 1000, 1000, 72.0, 0)
    assert len(page.iter_by_class(OcrClass.CAPTION)) == 1


def test_image_regions_contribute_no_text():
    regions = [Region("image", (0, 0, 999, 99), ("logo",))]
    page = build_page(regions, 1000, 1000, 72.0, 0)
    assert page.iter_by_class(OcrClass.LINE) == []
    assert page.iter_by_class(OcrClass.WORD) == []


def test_degenerate_box_is_skipped_without_raising():
    regions = [
        Region("text", (500, 10, 100, 20), ("inverted",)),
        Region("text", (0, 0, 999, 99), ("kept",)),
    ]
    page = build_page(regions, 1000, 1000, 72.0, 0)
    assert [w.text for w in page.iter_by_class(OcrClass.WORD)] == ["kept"]


def test_real_response_builds_a_full_tree():
    from deepseek_ocr_pdf.grounding import parse

    result = parse((FIXTURES / "v2_statement.txt").read_text())
    page = build_page(list(result.regions), 1314, 1700, 200.0, 0)
    # 11 regions, one of which holds 3 lines, so 13 line elements.
    assert len(page.iter_by_class(*OcrClass.LINE_TYPES)) == 13
    text = " ".join(w.text for w in page.iter_by_class(OcrClass.WORD))
    assert "ACME" in text
    assert "13,930.21" in text
