"""The data passed between the detector/tracker and the counting logic."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Box


@dataclass(frozen=True)
class Track:
    """One tracked object in one frame."""

    track_id: int
    cls: str
    conf: float
    box: Box
