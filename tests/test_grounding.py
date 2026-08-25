# tests/test_grounding.py
from pathlib import Path

from deepseek_ocr_pdf.grounding import parse, strip_markup

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


def test_v1_latex_delimiters_are_removed():
    result = parse((FIXTURES / "v1_balances.txt").read_text())
    text = result.regions[0].lines[0]
    assert "\\(" not in text
    assert "\\)" not in text
    assert "12,480.55" in text
    assert "Closing Balance" in text


def test_escaped_dollar_becomes_literal_dollar():
    assert strip_markup(r"total \$5") == "total $5"


def test_html_table_tags_are_removed():
    assert strip_markup("<td>Widgets</td>") == "Widgets"


def test_underscore_inside_word_is_kept():
    # Identifiers must survive; only markdown emphasis is stripped.
    assert strip_markup("field_name_here") == "field_name_here"


def test_multi_box_reference_is_a_block_not_text():
    """The model groups consecutive lines under one label with one box each.

    Observed on USPS receipts: ``text[[60, 381, 611, 415], [75, 413, 340,
    449]]`` followed by two lines. A single-box pattern does not match that
    header, so the coordinates fall through into the body and end up in the
    text layer as searchable garbage.
    """
    result = parse(
        "text[[60, 381, 611, 415], [75, 413, 340, 449]]\nfirst line\nsecond line"
    )
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.lines == ("first line", "second line")
    assert region.boxes == ((60, 381, 611, 415), (75, 413, 340, 449))


def test_multi_box_bbox_is_the_union():
    result = parse("text[[10, 20, 30, 40], [5, 50, 60, 70]]\na\nb")
    assert result.regions[0].bbox == (5, 20, 60, 70)


def test_single_box_region_still_exposes_boxes():
    result = parse("text[[1, 2, 3, 4]]\nhello")
    assert result.regions[0].boxes == ((1, 2, 3, 4),)


def test_multi_box_spanning_newlines():
    result = parse("text[[1, 2, 3, 4],\n[5, 6, 7, 8]]\none\ntwo")
    assert len(result.regions) == 1
    assert result.regions[0].boxes == ((1, 2, 3, 4), (5, 6, 7, 8))


def test_table_cells_do_not_glue_together():
    """Removing an HTML tag must leave a word boundary behind.

    v2 returns tables as HTML. Deleting <td> outright ran the cells together --
    "INVOICE" and "GAUDREAU, ERIN" became "INVOICEGAUDREAU, ERIN" in the text
    layer, which defeats any search that respects word boundaries.
    """
    result = parse(
        "text[[1, 2, 3, 4]]\n<td>INVOICE</td><td>GAUDREAU, ERIN</td>"
    )
    assert result.regions[0].lines == ("INVOICE GAUDREAU, ERIN",)


def test_html_entities_are_decoded():
    result = parse("text[[1, 2, 3, 4]]\nBRAKE SHOES &amp;/OR PADS")
    assert result.regions[0].lines == ("BRAKE SHOES &/OR PADS",)


def test_quote_and_numeric_entities_are_decoded():
    result = parse('text[[1, 2, 3, 4]]\n&quot;Paid&quot; &#8212; in full')
    assert result.regions[0].lines == ('"Paid" — in full',)


def test_decoded_entity_is_not_reparsed_as_a_tag():
    result = parse("text[[1, 2, 3, 4]]\nif a &lt;b&gt; c")
    assert result.regions[0].lines == ("if a <b> c",)


def test_tag_removal_does_not_leave_double_spaces():
    result = parse("text[[1, 2, 3, 4]]\n<p>Total</p> <b>due</b>")
    assert result.regions[0].lines == ("Total due",)
