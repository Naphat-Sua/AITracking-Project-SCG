"""Object detection + multi-object tracking.

``YoloTracker`` wraps an Ultralytics YOLO model with its built-in ByteTrack /
BoT-SORT tracker, so every detected object keeps the same ID from frame to
frame. Anything with an ``update(frame) -> list[Track]`` method can be used
instead (the tests use a scripted fake).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from .config import ConfigError
from .geometry import Box
from .tracks import Track


class Tracker(Protocol):
    def update(self, frame: np.ndarray) -> List[Track]:
        ...


class YoloTracker:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        classes: Optional[Sequence[str]] = None,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The 'ultralytics' package is required for detection. Install it with:  pip install ultralytics"
            ) from exc

        # Model weights such as yolo11n.pt are downloaded automatically on first use.
        self.model = YOLO(model)
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.class_ids = self._resolve_classes(classes)
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def _resolve_classes(self, classes: Optional[Sequence[str]]) -> Optional[List[int]]:
        if not classes:
            return None
        by_name = {name.lower(): idx for idx, name in self.names.items()}
        ids, unknown = [], []
        for name in classes:
            key = str(name).strip().lower()
            if key.isdigit() and int(key) in self.names:
                ids.append(int(key))
            elif key in by_name:
                ids.append(by_name[key])
            else:
                unknown.append(name)
        if unknown:
            available = ", ".join(sorted(self.names.values()))
            raise ConfigError(f"unknown class name(s): {', '.join(unknown)}. This model knows: {available}")
        return ids

    def update(self, frame: np.ndarray) -> List[Track]:
        kwargs = dict(persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                      classes=self.class_ids, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        result = self.model.track(frame, **kwargs)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        cls_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        return [
            Track(track_id=int(tid), cls=self.names.get(int(c), str(c)), conf=float(cf), box=Box(*map(float, b)))
            for tid, c, cf, b in zip(ids, cls_ids, confs, xyxy)
        ]
