import pytest

from aitracking.geometry import Box, point_in_polygon, segments_intersect, side_of_line, to_pixels


def test_side_of_line_vertical_top_to_bottom():
    p1, p2 = (0.5, 0.0), (0.5, 1.0)
    assert side_of_line(p1, p2, (0.2, 0.5)) == 1  # screen-left
    assert side_of_line(p1, p2, (0.8, 0.5)) == -1  # screen-right
    assert side_of_line(p1, p2, (0.5, 0.3)) == 0


def test_segments_intersect():
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (0.4, 0.4), (0, 1), (1, 0))
    # parallel
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
    # touching at an end point counts
    assert segments_intersect((0, 0), (0.5, 0.5), (0.5, 0.5), (1, 0))


def test_point_in_polygon():
    square = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
    assert point_in_polygon((0.3, 0.3), square)
    assert not point_in_polygon((0.6, 0.3), square)
    assert not point_in_polygon((0.3, 0.3), square[:2])  # degenerate


def test_box_anchor_and_pixels():
    b = Box(10, 20, 30, 60)
    assert b.center == (20, 40)
    assert b.bottom_center == (20, 60)
    assert b.anchor("center") == (20, 40)
    with pytest.raises(ValueError):
        b.anchor("top")
    assert to_pixels([(0.5, 0.25)], 640, 480) == [(320, 120)]
