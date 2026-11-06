"""The processing loop: read frame -> detect & track -> count -> draw -> output.

``Pipeline.process`` handles a single frame and is what the tests drive.
``Pipeline.run`` reads a whole source and writes the optional outputs (video
file, live window, latest JPEG for the web dashboard).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import cv2
import numpy as np

from .annotate import Annotator
from .config import TrackingConfig
from .counting import LineCounter, ZoneMonitor
from .events import EventLog
from .tracker import Tracker
from .tracks import Track

log = logging.getLogger(__name__)

TRAIL_FORGET_FRAMES = 30


class Pipeline:
    def __init__(self, config: TrackingConfig, tracker: Optional[Tracker] = None,
                 events: Optional[EventLog] = None) -> None:
        self.config = config.validate()
        self._tracker = tracker
        self.events = events or EventLog()
        self.lines = [LineCounter(lc, anchor=config.anchor, count_once=config.count_once) for lc in config.lines]
        self.zones = [ZoneMonitor(zc, anchor=config.anchor) for zc in config.zones]
        self.annotator = Annotator(trail=config.trail, anchor=config.anchor)

        self._lock = threading.Lock()
        self._unique: Dict[str, Set[int]] = defaultdict(set)
        self._current: Dict[str, int] = {}
        self._last_seen: Dict[int, int] = {}
        self._frames = 0
        self._t = 0.0
        self._fps = 0.0
        self._last_tick: Optional[float] = None
        self._dropped = 0
        self._started = datetime.now().astimezone().isoformat(timespec="seconds")
        self._status = "starting"
        self._error: Optional[str] = None
        self._frame_size: Optional[List[int]] = None
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._new_frame = threading.Condition(self._lock)

    # ---------------------------------------------------------------- tracker
    @property
    def tracker(self) -> Tracker:
        if self._tracker is None:
            from .tracker import YoloTracker

            c = self.config
            self._tracker = YoloTracker(model=c.model, tracker=c.tracker, classes=c.classes, conf=c.conf,
                                        iou=c.iou, imgsz=c.imgsz, device=c.device)
        return self._tracker

    # ------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> np.ndarray:
        """Track, count and annotate one frame. Returns the annotated frame."""
        h, w = frame.shape[:2]
        now = time.monotonic()
        if self._last_tick is not None:
            inst = 1.0 / max(now - self._last_tick, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_tick = now
        tracks: List[Track] = self.tracker.update(frame)

        new_events = []
        for line in self.lines:
            new_events.extend(line.update(tracks, w, h, frame_idx, t))
        for zone in self.zones:
            new_events.extend(zone.update(tracks, w, h, frame_idx, t))
        self.events.add(new_events)

        current: Dict[str, int] = defaultdict(int)
        with self._lock:
            for track in tracks:
                self._unique[track.cls].add(track.track_id)
                self._last_seen[track.track_id] = frame_idx
                current[track.cls] += 1
            self._current = dict(current)
            self._frames = frame_idx + 1
            self._t = t
            self._frame_size = [w, h]
            recent_ids = [tid for tid, seen in self._last_seen.items() if frame_idx - seen <= TRAIL_FORGET_FRAMES]
            self._last_seen = {tid: self._last_seen[tid] for tid in recent_ids}
        self.annotator.forget(recent_ids)

        fps_text = f"{self._fps:4.1f} FPS" if self._fps else "-- FPS"
        header = [f"AITracking  |  {len(tracks)} tracked  |  {fps_text}"]
        for line in self.lines:
            s = line.snapshot()
            header.append(f"{line.name}: in {s['in']}  out {s['out']}  net {s['net']}")
        for zone in self.zones:
            header.append(f"{zone.name}: {zone.snapshot(t)['occupancy']} inside")
        return self.annotator.draw(frame, tracks, self.lines, self.zones, header, t)

    # -------------------------------------------------------------- run loop
    def run(self, output: Optional[str] = None, show: bool = False, publish_jpeg: bool = False,
            stop_event: Optional[threading.Event] = None,
            on_progress: Optional[Callable[[int, Optional[int]], None]] = None) -> dict:
        from .sources import VideoSource

        source = VideoSource(self.config.source, loop=self.config.loop)
        writer = None
        fps_src = source.fps
        start = time.monotonic()
        frame_idx = 0
        self._set_status("running")
        try:
            # load the model before the first frame so the FPS figure is honest
            self.tracker  # noqa: B018
            while stop_event is None or not stop_event.is_set():
                if self.config.max_frames is not None and frame_idx >= self.config.max_frames:
                    break
                ok, frame = source.read()
                if not ok:
                    break
                t = (time.monotonic() - start) if source.live else frame_idx / fps_src
                annotated = self.process(frame, frame_idx, t)

                if output:
                    if writer is None:
                        writer = _open_writer(output, fps_src, annotated.shape[1], annotated.shape[0])
                    writer.write(annotated)
                if publish_jpeg:
                    ok_jpg, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_jpg:
                        with self._new_frame:
                            self._jpeg = buf.tobytes()
                            self._jpeg_seq += 1
                            self._new_frame.notify_all()
                if show:
                    cv2.imshow("AITracking (press q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                self._dropped = source.dropped
                if on_progress:
                    on_progress(frame_idx + 1, source.frame_count)
                frame_idx += 1
            self._set_status("finished")
        except Exception as exc:
            self._set_status("error", str(exc))
            raise
        finally:
            source.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            self.events.close()
        return self.snapshot()

    # --------------------------------------------------------------- outputs
    def _set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._new_frame:
            self._status = status
            self._error = error
            self._new_frame.notify_all()

    def wait_jpeg(self, last_seq: int, timeout: float = 5.0):
        """Block until a newer JPEG than ``last_seq`` exists. Returns (seq, bytes or None)."""
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._jpeg_seq > last_seq or self._status in ("finished", "error"),
                                     timeout=timeout)
            return self._jpeg_seq, self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            unique = {cls: len(ids) for cls, ids in self._unique.items()}
            data = {
                "status": self._status,
                "error": self._error,
                "source": str(self.config.source),
                "started_at": self._started,
                "frames": self._frames,
                "elapsed_s": round(self._t, 2),
                "fps": round(self._fps, 1),
                "dropped_frames": self._dropped,
                "frame_size": self._frame_size,
                "current": dict(self._current),
                "current_total": sum(self._current.values()),
                "unique": unique,
                "unique_total": sum(unique.values()),
            }
            t = self._t
        data["lines"] = [line.snapshot() for line in self.lines]
        data["zones"] = [zone.snapshot(t) for zone in self.zones]
        data["events_total"] = self.events.total
        return data


def _open_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out.suffix.lower() in (".mp4", ".m4v", ".mov") else "XVID"))
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out}")
    return writer
