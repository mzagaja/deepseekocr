# tests/test_coverage.py
from deepseek_ocr_pdf.coverage import UNCOVERED_THRESHOLD, group_into_crops, parse_tsv, uncovered
from deepseek_ocr_pdf.geometry import Box

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def _tsv(*rows: str) -> str:
    return "\n".join([HEADER, *rows])


def test_words_on_one_line_merge_into_one_box():
    tsv = _tsv(
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t96\tHello",
        "5\t1\t1\t1\t1\t2\t50\t20\t40\t10\t95\tworld",
    )
    boxes = parse_tsv(tsv)
    assert boxes == [Box(10, 20, 90, 30)]


def test_separate_lines_stay_separate():
    tsv = _tsv(
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t96\tone",
        "5\t1\t1\t1\t2\t1\t10\t40\t30\t10\t96\ttwo",
    )
    assert len(parse_tsv(tsv)) == 2


def test_non_word_levels_are_ignored():
    tsv = _tsv(
        "4\t1\t1\t1\t1\t0\t0\t0\t999\t999\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t96\tHello",
    )
    assert parse_tsv(tsv) == [Box(10, 20, 40, 30)]


def test_empty_text_is_ignored():
    tsv = _tsv("5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t96\t   ")
    assert parse_tsv(tsv) == []


def test_negative_confidence_is_ignored():
    tsv = _tsv("5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t-1\tjunk")
    assert parse_tsv(tsv) == []


def test_tab_inside_text_does_not_break_parsing():
    # Tesseract writes raw text; a stray tab must not shift the columns.
    tsv = _tsv("5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t96\tHello\tthere")
    assert parse_tsv(tsv) == [Box(10, 20, 40, 30)]


def test_header_only_output_yields_nothing():
    assert parse_tsv(HEADER) == []


def test_threshold_is_thirty_percent():
    assert UNCOVERED_THRESHOLD == 0.30


def test_fully_covered_line_is_not_reported():
    detected = [Box(0, 0, 100, 10)]
    assert uncovered(detected, [Box(0, 0, 100, 10)]) == []


def test_untouched_line_is_reported():
    detected = [Box(0, 500, 100, 510)]
    assert uncovered(detected, [Box(0, 0, 100, 10)]) == detected


def test_line_just_under_threshold_is_reported():
    # 25% covered, below the 30% threshold.
    detected = [Box(0, 0, 100, 10)]
    assert uncovered(detected, [Box(0, 0, 25, 10)]) == detected


def test_line_just_over_threshold_is_not_reported():
    detected = [Box(0, 0, 100, 10)]
    assert uncovered(detected, [Box(0, 0, 35, 10)]) == []


def test_no_grounding_boxes_reports_everything():
    detected = [Box(0, 0, 100, 10), Box(0, 20, 100, 30)]
    assert uncovered(detected, []) == detected


def test_nearby_boxes_group_into_one_crop():
    boxes = [Box(10, 100, 90, 110), Box(10, 115, 90, 125)]
    crops = group_into_crops(boxes, 1000, 1000)
    assert len(crops) == 1


def test_distant_boxes_stay_separate():
    boxes = [Box(10, 100, 90, 110), Box(10, 800, 90, 810)]
    assert len(group_into_crops(boxes, 1000, 1000)) == 2


def test_crop_is_padded_around_the_content():
    crops = group_into_crops([Box(100, 100, 200, 120)], 1000, 1000)
    assert crops[0].left < 100
    assert crops[0].top < 100
    assert crops[0].right > 200
    assert crops[0].bottom > 120


def test_crop_is_clamped_to_page_bounds():
    crops = group_into_crops([Box(0, 0, 20, 10)], 1000, 1000)
    assert crops[0].left == 0
    assert crops[0].top == 0


def test_no_boxes_yields_no_crops():
    assert group_into_crops([], 1000, 1000) == []
