"""Configuration for a tracking run: video source, detector and counting rules.

A config can come from a YAML file (see ``configs/example.yaml``), from command
line flags, or both - flags override values from the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

Point = Tuple[float, float]


class ConfigError(ValueError):
    """Raised for an invalid configuration, with a message meant for the user."""


def _check_normalized(points: Sequence[Point], what: str) -> None:
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConfigError(
                f"{what}: point ({x}, {y}) is outside 0..1. Coordinates are fractions of the frame "
                f"width/height, e.g. 0.5 = middle of the frame."
            )


def _as_points(raw: Any, what: str) -> List[Point]:
    try:
        points = [(float(p[0]), float(p[1])) for p in raw]
    except (TypeError, ValueError, IndexError) as exc:
        raise ConfigError(f"{what}: expected a list of [x, y] pairs, got {raw!r}") from exc
    _check_normalized(points, what)
    return points


@dataclass
class LineConfig:
    """A counting line. Objects crossing it are counted per direction.

    Direction ``in`` means the object crossed towards the left-hand side of
    someone walking along the line from ``points[0]`` to ``points[1]``:

    * vertical line drawn top -> bottom:   moving left -> right is ``in``
    * horizontal line drawn left -> right: moving bottom -> top is ``in``

    Swap the two points to swap in/out.
    """

    name: str
    points: List[Point]

    def __post_init__(self) -> None:
        self.points = _as_points(self.points, f"line '{self.name}'")
        if len(self.points) != 2:
            raise ConfigError(f"line '{self.name}': needs exactly 2 points, got {len(self.points)}")
        if self.points[0] == self.points[1]:
            raise ConfigError(f"line '{self.name}': the two points must differ")


@dataclass
class ZoneConfig:
    """A polygon zone. Reports how many objects are inside and how long they stay."""

    name: str
    points: List[Point]

    def __post_init__(self) -> None:
        self.points = _as_points(self.points, f"zone '{self.name}'")
        if len(self.points) < 3:
            raise ConfigError(f"zone '{self.name}': needs at least 3 points, got {len(self.points)}")


@dataclass
class TrackingConfig:
    source: Union[int, str] = 0
    model: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    classes: Optional[List[str]] = None  # None = every class the model knows
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: Optional[str] = None  # None = auto (GPU if available)
    anchor: str = "bottom"  # point of the box used for counting: bottom | center
    lines: List[LineConfig] = field(default_factory=list)
    zones: List[ZoneConfig] = field(default_factory=list)
    count_once: bool = True  # count each track at most once per line
    trail: int = 30  # frames of movement history drawn behind each object
    loop: bool = False  # restart a video file when it ends (useful for demos)
    max_frames: Optional[int] = None

    def validate(self) -> "TrackingConfig":
        if not 0.0 < self.conf <= 1.0:
            raise ConfigError(f"conf must be in (0, 1], got {self.conf}")
        if not 0.0 < self.iou <= 1.0:
            raise ConfigError(f"iou must be in (0, 1], got {self.iou}")
        if self.anchor not in ("bottom", "center"):
            raise ConfigError(f"anchor must be 'bottom' or 'center', got {self.anchor!r}")
        if self.imgsz < 32:
            raise ConfigError(f"imgsz too small: {self.imgsz}")
        names = [item.name for item in [*self.lines, *self.zones]]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"line/zone names must be unique, duplicated: {', '.join(dupes)}")
        return self

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackingConfig":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown config keys: {', '.join(unknown)}")

        lines = [_named(LineConfig, item, "line", i) for i, item in enumerate(data.pop("lines", None) or [])]
        zones = [_named(ZoneConfig, item, "zone", i) for i, item in enumerate(data.pop("zones", None) or [])]
        if "source" in data:
            data["source"] = parse_source(data["source"])
        if "classes" in data and isinstance(data["classes"], str):
            data["classes"] = parse_classes(data["classes"])
        return cls(lines=lines, zones=zones, **data).validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrackingConfig":
        import yaml

        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping of keys")
        return cls.from_dict(data)


def _named(kind, item: Any, label: str, index: int):
    if not isinstance(item, dict) or "points" not in item:
        raise ConfigError(f"{label} #{index + 1}: expected {{name: ..., points: [[x, y], ...]}}")
    return kind(name=str(item.get("name") or f"{label}-{index + 1}"), points=item["points"])


# ---------------------------------------------------------------- CLI parsing
def parse_source(value: Union[int, str]) -> Union[int, str]:
    """``"0"`` -> webcam 0; anything else (file path, rtsp://, http://) stays a string."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def parse_classes(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [c.strip() for c in value.split(",") if c.strip()]
    return items or None


def parse_line_arg(text: str, index: int) -> LineConfig:
    """Parse ``[name=]x1,y1,x2,y2`` e.g. ``entrance=0.5,0,0.5,1``."""
    name, _, coords = text.rpartition("=")
    try:
        nums = [float(v) for v in coords.split(",")]
    except ValueError as exc:
        raise ConfigError(f"--line {text!r}: expected [name=]x1,y1,x2,y2 with numbers") from exc
    if len(nums) != 4:
        raise ConfigError(f"--line {text!r}: expected 4 numbers x1,y1,x2,y2, got {len(nums)}")
    return LineConfig(name=name or f"line-{index + 1}", points=[(nums[0], nums[1]), (nums[2], nums[3])])


def parse_zone_arg(text: str, index: int) -> ZoneConfig:
    """Parse ``[name=]x1,y1;x2,y2;x3,y3...`` e.g. ``dock=0.1,0.5;0.5,0.5;0.5,1;0.1,1``."""
    name, _, coords = text.rpartition("=")
    try:
        points = [tuple(float(v) for v in pair.split(",")) for pair in coords.split(";") if pair.strip()]
    except ValueError as exc:
        raise ConfigError(f"--zone {text!r}: expected [name=]x1,y1;x2,y2;x3,y3 with numbers") from exc
    if any(len(p) != 2 for p in points):
        raise ConfigError(f"--zone {text!r}: every point needs exactly x,y")
    return ZoneConfig(name=name or f"zone-{index + 1}", points=points)
