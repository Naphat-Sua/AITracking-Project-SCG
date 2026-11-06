"""Small 2D geometry helpers used by the line counters and zones.

All shapes are defined in *normalized* coordinates (0..1 of the frame width /
height) so the same config works for any camera resolution. ``to_pixels``
converts them for a concrete frame size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """Z component of the cross product (a - o) x (b - o).

    > 0 when ``b`` is to the left of the directed line o -> a (in image
    coordinates, where y grows downward, that is visually clockwise),
    < 0 on the other side and 0 when the three points are collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side_of_line(p1: Point, p2: Point, point: Point) -> int:
    """Return +1 / -1 for the side of the directed line p1 -> p2, 0 if on it."""
    c = cross(p1, p2, point)
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """True when segment a1-a2 strictly crosses or touches segment b1-b2."""
    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

    if d1 == 0 and on_segment(b1, b2, a1):
        return True
    if d2 == 0 and on_segment(b1, b2, a2):
        return True
    if d3 == 0 and on_segment(a1, a2, b1):
        return True
    if d4 == 0 and on_segment(a1, a2, b2):
        return True
    return False


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test (points on the edge may go either way)."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def to_pixels(points: Iterable[Point], width: int, height: int) -> List[Tuple[int, int]]:
    """Convert normalized points to integer pixel coordinates."""
    return [(int(round(x * width)), int(round(y * height))) for x, y in points]


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Where a person/vehicle touches the ground - the best point for counting."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def anchor(self, mode: str = "bottom") -> Point:
        if mode == "center":
            return self.center
        if mode == "bottom":
            return self.bottom_center
        raise ValueError(f"unknown anchor mode {mode!r} (use 'bottom' or 'center')")
