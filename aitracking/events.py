"""Events produced by the counters (line crossings, zone enter/exit) and a log for them.

The log keeps the most recent events in memory (for the dashboard) and can
append every event to a ``.csv`` or ``.jsonl`` file as it happens, so nothing
is lost if the process is stopped.
"""

from __future__ import annotations

import csv
import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Union


@dataclass(frozen=True)
class Event:
    kind: str  # line_cross | zone_enter | zone_exit
    name: str  # name of the line or zone
    track_id: int
    cls: str
    frame: int
    t: float  # seconds since the start of the run (video time for files)
    direction: str = ""  # line_cross: in | out
    dwell_s: Optional[float] = None  # zone_exit: seconds spent inside
    wall_time: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


EVENT_FIELDS = [f.name for f in fields(Event)]


class EventLog:
    def __init__(self, path: Optional[Union[str, Path]] = None, keep: int = 500) -> None:
        self._recent: Deque[Event] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self._total = 0
        self._path = Path(path) if path else None
        self._fh = None
        self._writer = None
        if self._path:
            self._open()

    def _open(self) -> None:
        suffix = self._path.suffix.lower()
        if suffix not in (".csv", ".jsonl"):
            raise ValueError(f"event file must end in .csv or .jsonl, got {self._path.name}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists() or self._path.stat().st_size == 0
        self._fh = self._path.open("a", newline="", encoding="utf-8")
        if suffix == ".csv":
            self._writer = csv.DictWriter(self._fh, fieldnames=EVENT_FIELDS)
            if new_file:
                self._writer.writeheader()

    def add(self, events: List[Event]) -> None:
        if not events:
            return
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            for event in events:
                if not event.wall_time:
                    event = Event(**{**event.to_dict(), "wall_time": now})
                self._recent.append(event)
                self._total += 1
                if self._fh is not None:
                    row = event.to_dict()
                    if self._writer is not None:
                        self._writer.writerow(row)
                    else:
                        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self._fh is not None:
                self._fh.flush()

    def recent(self, limit: int = 50) -> List[Event]:
        with self._lock:
            items = list(self._recent)
        return items[-limit:][::-1]  # newest first

    @property
    def total(self) -> int:
        return self._total

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
                self._writer = None
