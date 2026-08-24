from deepseek_ocr_pdf.geometry import Box, covered_fraction, intersection_area


def test_area_and_dimensions():
    b = Box(10, 20, 40, 60)
    assert b.width == 30
    assert b.height == 40
    assert b.area == 1200


def test_intersection_area_overlapping():
    assert intersection_area(Box(0, 0, 10, 10), Box(5, 5, 15, 15)) == 25


def test_intersection_area_disjoint_is_zero():
    assert intersection_area(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0


def test_intersection_area_touching_edges_is_zero():
    assert intersection_area(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0


def test_covered_fraction_fully_covered():
    assert covered_fraction(Box(0, 0, 10, 10), [Box(-5, -5, 15, 15)]) == 1.0


def test_covered_fraction_uncovered():
    assert covered_fraction(Box(0, 0, 10, 10), [Box(50, 50, 60, 60)]) == 0.0


def test_covered_fraction_half():
    assert covered_fraction(Box(0, 0, 10, 10), [Box(0, 0, 5, 10)]) == 0.5


def test_covered_fraction_caps_at_one_when_covers_overlap():
    # Two overlapping covers must not sum past 1.0
    covers = [Box(0, 0, 8, 10), Box(2, 0, 10, 10)]
    assert covered_fraction(Box(0, 0, 10, 10), covers) == 1.0


def test_covered_fraction_of_empty_box_is_one():
    # A zero-area box is trivially covered; it must never be reported missing.
    assert covered_fraction(Box(5, 5, 5, 5), []) == 1.0
