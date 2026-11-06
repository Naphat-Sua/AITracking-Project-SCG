"""Draw boxes, IDs, movement trails, counting lines and zones on a frame."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .counting import LineCounter, ZoneMonitor
from .geometry import to_pixels
from .tracks import Track

LINE_COLOR = (0, 215, 255)  # BGR amber
ZONE_COLOR = (255, 160, 40)  # BGR blue
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (32, 32, 32)


def color_for(track_id: int) -> Tuple[int, int, int]:
    """Stable, well-spread colour per track id (golden-ratio hue walk)."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 179)
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def _label(img: np.ndarray, text: str, org: Tuple[int, int], color, scale: float = 0.5) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


class Annotator:
    def __init__(self, trail: int = 30, anchor: str = "bottom") -> None:
        self.trail = trail
        self.anchor = anchor
        self._history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=max(trail, 1)))

    def draw(self, frame: np.ndarray, tracks: Iterable[Track], lines: List[LineCounter], zones: List[ZoneMonitor],
             header: List[str], t: float) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]

        # zones: translucent fill + outline
        if zones:
            overlay = img.copy()
            for zone in zones:
                pts = np.array(to_pixels(zone.polygon, w, h), dtype=np.int32)
                cv2.fillPoly(overlay, [pts], ZONE_COLOR)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
            for zone in zones:
                pts = np.array(to_pixels(zone.polygon, w, h), dtype=np.int32)
                cv2.polylines(img, [pts], True, ZONE_COLOR, 2, cv2.LINE_AA)
                snap = zone.snapshot(t)
                x, y = pts.min(axis=0)
                _label(img, f"{zone.name}: {snap['occupancy']} inside", (int(x) + 4, int(y) + 22), ZONE_COLOR)

        # counting lines
        for line in lines:
            (x1, y1), (x2, y2) = to_pixels(line.config.points, w, h)
            cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR, 3, cv2.LINE_AA)
            snap = line.snapshot()
            # label at the first end point (kept inside the frame) so it does not cover people mid-line
            text = f"{line.name}  in {snap['in']}  out {snap['out']}"
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            lx = min(max(x1 + 6, 0), max(w - tw - 8, 0))
            ly = min(max(y1 + 24, 24), h - 4)
            _label(img, text, (lx, ly), LINE_COLOR, 0.55)

        # tracked objects
        for track in tracks:
            color = color_for(track.track_id)
            b = track.box
            p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
            cv2.rectangle(img, p1, p2, color, 2)
            _label(img, f"#{track.track_id} {track.cls} {track.conf:.2f}", (p1[0], p1[1] - 2), color, 0.45)
            ax, ay = track.box.anchor(self.anchor)
            history = self._history[track.track_id]
            history.append((int(ax), int(ay)))
            if self.trail > 1 and len(history) > 1:
                cv2.polylines(img, [np.array(history, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(img, (int(ax), int(ay)), 4, color, -1)

        # header panel
        if header:
            line_h = 22
            panel_h = 10 + line_h * len(header)
            panel_w = 12 + max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for s in header)
            sub = img[0:min(panel_h, h), 0:min(panel_w, w)]
            cv2.addWeighted(sub, 0.35, np.full_like(sub, PANEL_COLOR), 0.65, 0, sub)
            for i, text in enumerate(header):
                cv2.putText(img, text, (6, 24 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1,
                            cv2.LINE_AA)
        return img

    def forget(self, active_ids: Iterable[int]) -> None:
        """Drop trails of objects that are no longer tracked."""
        keep = set(active_ids)
        for tid in list(self._history):
            if tid not in keep:
                del self._history[tid]
