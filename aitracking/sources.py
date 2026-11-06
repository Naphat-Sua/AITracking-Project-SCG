"""Open a video source: a file, a webcam index, or a network stream (RTSP / HTTP)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .config import ConfigError

log = logging.getLogger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live(source: Union[int, str]) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(STREAM_PREFIXES)


class VideoSource:
    """Thin wrapper around ``cv2.VideoCapture`` that

    * fails early with a clear message when a file does not exist,
    * for cameras / streams, always hands out the *newest* frame: a background
      thread keeps reading, so when detection is slower than the camera the
      dashboard stays real-time instead of drifting further and further behind,
    * reconnects to cameras / streams when they drop, and
    * can loop a video file forever (handy for a demo dashboard).

    Video files are read frame by frame without skipping, so counts on a
    recording are reproducible.
    """

    def __init__(self, source: Union[int, str], loop: bool = False, reconnect_delay: float = 2.0,
                 max_reconnects: Optional[int] = None, live: Optional[bool] = None) -> None:
        self.source = source
        self.live = is_live(source) if live is None else live
        self.loop = loop and not self.live
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        if not is_live(source) and not Path(str(source)).is_file():
            raise ConfigError(f"video file not found: {source}")
        self._cap = self._open()
        self._fps = self._read_fps(self._cap)

        # latest-frame reader for live sources
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0  # frames grabbed
        self._taken = 0  # frames handed out
        self._taken_seq = 0  # seq of the last frame handed out
        self._ended = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.live:
            self._thread = threading.Thread(target=self._reader, name="aitracking-grabber", daemon=True)
            self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            hint = " (is the camera connected and not used by another app?)" if isinstance(self.source, int) else ""
            raise ConfigError(f"could not open video source: {self.source}{hint}")
        return cap

    @staticmethod
    def _read_fps(cap: cv2.VideoCapture) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # cameras/streams often report 0 or nonsense
        return float(fps) if fps and 1 <= fps <= 240 else 25.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> Optional[int]:
        if self.live or self._cap is None:
            return None
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None

    @property
    def dropped(self) -> int:
        """Live sources: frames skipped because detection was slower than the camera."""
        return max(0, self._seq - self._taken)

    # ------------------------------------------------------------------ files
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.live:
            return self._read_latest()
        ok, frame = self._cap.read()
        if ok:
            return True, frame
        if self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            return (True, frame) if ok else (False, None)
        return False, None

    # ------------------------------------------------------------- live streams
    def _read_latest(self, timeout: float = 30.0) -> Tuple[bool, Optional[np.ndarray]]:
        with self._cond:
            last = self._taken_seq
            self._cond.wait_for(lambda: self._seq > last or self._ended, timeout=timeout)
            if self._seq > last:
                self._taken_seq = self._seq
                self._taken += 1
                return True, self._frame
            return False, None

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self._stop.is_set() or not self._reconnect():
                    break
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()
        with self._cond:
            self._ended = True
            self._cond.notify_all()

    def _reconnect(self) -> bool:
        attempt = 0
        while not self._stop.is_set() and (self.max_reconnects is None or attempt < self.max_reconnects):
            attempt += 1
            log.warning("video source %s dropped, reconnecting (attempt %d)...", self.source, attempt)
            self._cap.release()
            if self._stop.wait(self.reconnect_delay):
                return False
            try:
                self._cap = self._open()
            except ConfigError:
                continue
            log.info("reconnected to %s", self.source)
            return True
        return False

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._cap is not None:
            self._cap.release()
