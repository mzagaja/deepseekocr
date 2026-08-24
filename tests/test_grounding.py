# tests/test_grounding.py
from pathlib import Path

from deepseek_ocr_pdf.grounding import parse

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_every_block_from_real_response():
    result = parse((FIXTURES / "v2_statement.txt").read_text())
    assert len(result.regions) == 11
    assert result.malformed == 0


def test_first_region_label_and_box():
    result = parse((FIXTURES / "v2_statement.txt").read_text())
    first = result.regions[0]
    assert first.label == "sub_title"
    assert first.bbox == (83, 63, 429, 91)


def test_heading_markup_is_stripped_from_text():
    result = parse((FIXTURES / "v2_statement.txt").read_text())
    assert result.regions[0].lines == ("ACME CORPORATION",)


def test_multi_line_region_keeps_each_line():
    result = parse((FIXTURES / "v2_statement.txt").read_text())
    paragraph = result.regions[5]
    assert len(paragraph.lines) == 3
    assert paragraph.lines[0] == "This statement summarizes all transactions posted to your"
    assert paragraph.lines[2] == "entry carefully and report any discrepancy within 30 days."


def test_single_block_without_trailing_newline():
    result = parse("text[[1, 2, 3, 4]]\nhello")
    assert result.regions[0].bbox == (1, 2, 3, 4)
    assert result.regions[0].lines == ("hello",)


def test_whitespace_inside_coordinates_is_tolerated():
    result = parse("text[[1,2,  3 , 4]]\nhi")
    assert result.regions[0].bbox == (1, 2, 3, 4)


def test_region_with_no_text_is_dropped():
    result = parse("image[[0, 0, 10, 10]]\n\ntext[[1, 2, 3, 4]]\nreal")
    assert len(result.regions) == 1
    assert result.regions[0].label == "text"


def test_malformed_block_is_counted_not_raised():
    result = parse("text[[1, 2, 3]]\nbroken\n\ntext[[1, 2, 3, 4]]\ngood")
    assert len(result.regions) == 1
    assert result.malformed == 1


def test_prose_with_no_boxes_yields_nothing():
    result = parse("The document appears to be a bank statement.")
    assert result.regions == ()
    assert result.malformed == 0
