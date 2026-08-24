"""Axis-aligned box arithmetic shared by layout and coverage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """An axis-aligned box with a top-left origin."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def union(self, other: Box) -> Box:
        return Box(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )


def intersection_area(a: Box, b: Box) -> float:
    """Area shared by two boxes. Zero when they only touch at an edge."""
    left = max(a.left, b.left)
    right = min(a.right, b.right)
    top = max(a.top, b.top)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def covered_fraction(target: Box, covers: Iterable[Box]) -> float:
    """How much of ``target`` is covered, as a fraction of its own area.

    Overlapping covers are summed and then capped at 1.0 rather than unioned.
    Grounding boxes rarely overlap, and the cap keeps an overlap from
    reporting more than full coverage.
    """
    if target.area <= 0:
        return 1.0
    total = sum(intersection_area(target, c) for c in covers)
    return min(1.0, total / target.area)
