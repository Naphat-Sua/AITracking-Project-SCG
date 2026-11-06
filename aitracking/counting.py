"""Turn per-frame tracks into business numbers: line in/out counts and zone occupancy.

This module is pure Python (no OpenCV / YOLO) so it is fast to unit test.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from .config import LineConfig, ZoneConfig
from .events import Event
from .geometry import Point, point_in_polygon, segments_intersect, side_of_line
from .tracks import Track

# Forget a track's state after it has been missing for this many frames.
STALE_AFTER_FRAMES = 300


def _norm(point: Point, width: int, height: int) -> Point:
    return (point[0] / width, point[1] / height)


class LineCounter:
    """Counts tracks crossing one line, per direction and per class."""

    def __init__(self, config: LineConfig, anchor: str = "bottom", count_once: bool = True) -> None:
        self.config = config
        self.name = config.name
        self.p1, self.p2 = config.points
        self.anchor = anchor
        self.count_once = count_once
        self.counts: Dict[str, Dict[str, int]] = {"in": defaultdict(int), "out": defaultdict(int)}
        # last point that was clearly on one side of the line, per track
        self._last: Dict[int, Tuple[int, Point]] = {}
        self._last_seen: Dict[int, int] = {}
        self._counted: Set[int] = set()

    def update(self, tracks: Iterable[Track], width: int, height: int, frame: int, t: float) -> List[Event]:
        events: List[Event] = []
        for track in tracks:
            point = _norm(track.box.anchor(self.anchor), width, height)
            side = side_of_line(self.p1, self.p2, point)
            self._last_seen[track.track_id] = frame
            if side == 0:
                continue  # exactly on the line - wait until it is clearly across
            previous = self._last.get(track.track_id)
            self._last[track.track_id] = (side, point)
            if previous is None or previous[0] == side:
                continue
            if not segments_intersect(previous[1], point, self.p1, self.p2):
                continue  # changed side by going around the end of the line
            if self.count_once and track.track_id in self._counted:
                continue
            direction = "in" if previous[0] == 1 else "out"
            self.counts[direction][track.cls] += 1
            self._counted.add(track.track_id)
            events.append(Event(kind="line_cross", name=self.name, track_id=track.track_id, cls=track.cls,
                                frame=frame, t=round(t, 3), direction=direction))
        self._prune(frame)
        return events

    def _prune(self, frame: int) -> None:
        stale = [tid for tid, seen in self._last_seen.items() if frame - seen > STALE_AFTER_FRAMES]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._last.pop(tid, None)
            self._counted.discard(tid)

    def snapshot(self) -> dict:
        totals = {d: sum(c.values()) for d, c in self.counts.items()}
        return {
            "name": self.name,
            "points": [list(p) for p in self.config.points],
            "in": totals["in"],
            "out": totals["out"],
            "net": totals["in"] - totals["out"],
            "by_class": {d: dict(c) for d, c in self.counts.items()},
        }


class ZoneMonitor:
    """Tracks which objects are inside a polygon zone and how long they stay."""

    def __init__(self, config: ZoneConfig, anchor: str = "bottom", lost_after_frames: int = 30) -> None:
        self.config = config
        self.name = config.name
        self.polygon = config.points
        self.anchor = anchor
        self.lost_after_frames = lost_after_frames
        self._inside: Dict[int, Tuple[float, str]] = {}  # track_id -> (enter time, class)
        self._last_seen: Dict[int, Tuple[int, float]] = {}  # track_id -> (frame, t)
        self.entries: Dict[str, int] = defaultdict(int)
        self._dwell_total = 0.0
        self._dwell_count = 0

    def update(self, tracks: Iterable[Track], width: int, height: int, frame: int, t: float) -> List[Event]:
        events: List[Event] = []
        seen_now = set()
        for track in tracks:
            seen_now.add(track.track_id)
            self._last_seen[track.track_id] = (frame, t)
            inside = point_in_polygon(_norm(track.box.anchor(self.anchor), width, height), self.polygon)
            was_inside = track.track_id in self._inside
            if inside and not was_inside:
                self._inside[track.track_id] = (t, track.cls)
                self.entries[track.cls] += 1
                events.append(Event(kind="zone_enter", name=self.name, track_id=track.track_id, cls=track.cls,
                                    frame=frame, t=round(t, 3)))
            elif not inside and was_inside:
                events.append(self._exit(track.track_id, frame, t))

        # objects that vanished while inside the zone (left the camera, occluded too long)
        for tid in list(self._inside):
            if tid in seen_now:
                continue
            last_frame, last_t = self._last_seen.get(tid, (frame, t))
            if frame - last_frame > self.lost_after_frames:
                events.append(self._exit(tid, frame, last_t))

        for tid, (seen_frame, _) in list(self._last_seen.items()):
            if frame - seen_frame > STALE_AFTER_FRAMES and tid not in self._inside:
                del self._last_seen[tid]
        return events

    def _exit(self, track_id: int, frame: int, t: float) -> Event:
        enter_t, cls = self._inside.pop(track_id)
        dwell = max(0.0, t - enter_t)
        self._dwell_total += dwell
        self._dwell_count += 1
        return Event(kind="zone_exit", name=self.name, track_id=track_id, cls=cls, frame=frame,
                     t=round(t, 3), dwell_s=round(dwell, 2))

    def occupancy(self) -> Dict[str, int]:
        result: Dict[str, int] = defaultdict(int)
        for _, cls in self._inside.values():
            result[cls] += 1
        return dict(result)

    def snapshot(self, t: float) -> dict:
        occupancy = self.occupancy()
        current_dwell = [t - enter for enter, _ in self._inside.values()]
        return {
            "name": self.name,
            "points": [list(p) for p in self.polygon],
            "occupancy": sum(occupancy.values()),
            "occupancy_by_class": occupancy,
            "entries": sum(self.entries.values()),
            "entries_by_class": dict(self.entries),
            "avg_dwell_s": round(self._dwell_total / self._dwell_count, 2) if self._dwell_count else None,
            "longest_current_dwell_s": round(max(current_dwell), 2) if current_dwell else None,
        }
