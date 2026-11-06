"""Test doubles for the tracking app: a scripted tracker and synthetic videos."""

from typing import Callable, List

import numpy as np

from aitracking.geometry import Box
from aitracking.tracks import Track

W, H = 640, 480


def box_at(cx: float, cy: float, w: float = 40, h: float = 100) -> Box:
    """Box whose *bottom centre* is at (cx, cy)."""
    return Box(cx - w / 2, cy - h, cx + w / 2, cy)


def walker(track_id: int, start, end, frames: int, cls: str = "person") -> Callable[[int], List[Track]]:
    """A single object moving linearly from ``start`` to ``end`` (pixel bottom-centre) over ``frames``."""

    def script(i: int) -> List[Track]:
        if i >= frames:
            return []
        f = i / max(frames - 1, 1)
        x = start[0] + (end[0] - start[0]) * f
        y = start[1] + (end[1] - start[1]) * f
        return [Track(track_id=track_id, cls=cls, conf=0.9, box=box_at(x, y))]

    return script


def combine(*scripts):
    def script(i: int) -> List[Track]:
        out: List[Track] = []
        for s in scripts:
            out.extend(s(i))
        return out

    return script


class ScriptedTracker:
    """Returns pre-scripted tracks, one call per frame, ignoring the pixels."""

    def __init__(self, script: Callable[[int], List[Track]]) -> None:
        self.script = script
        self.calls = 0

    def update(self, frame: np.ndarray) -> List[Track]:
        tracks = self.script(self.calls)
        self.calls += 1
        return tracks


def blank_frame() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def write_video(path, frames: int, fps: float = 10.0) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert writer.isOpened()
    for i in range(frames):
        img = blank_frame()
        cv2.putText(img, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(img)
    writer.release()
