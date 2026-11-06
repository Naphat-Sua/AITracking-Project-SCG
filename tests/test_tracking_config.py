from pathlib import Path

import pytest

from aitracking.cli import build_parser, config_from_args
from aitracking.config import ConfigError, LineConfig, TrackingConfig, parse_line_arg, parse_source, parse_zone_arg

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"


def test_example_config_loads():
    cfg = TrackingConfig.from_yaml(EXAMPLE)
    assert cfg.source == 0
    assert cfg.lines[0].name == "entrance"
    assert cfg.zones[0].name == "waiting-area"
    assert "person" in cfg.classes


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2
    assert parse_source("rtsp://cam/1") == "rtsp://cam/1"
    assert parse_source("video.mp4") == "video.mp4"


def test_parse_line_and_zone_args():
    line = parse_line_arg("door=0.5,0,0.5,1", 0)
    assert line.name == "door" and line.points == [(0.5, 0.0), (0.5, 1.0)]
    assert parse_line_arg("0,0.5,1,0.5", 1).name == "line-2"
    zone = parse_zone_arg("dock=0.1,0.5;0.5,0.5;0.5,1", 0)
    assert zone.name == "dock" and len(zone.points) == 3


@pytest.mark.parametrize("bad", ["0.5,0,0.5", "a,b,c,d", "0.5,0,1.5,1"])
def test_bad_line_args(bad):
    with pytest.raises(ConfigError):
        parse_line_arg(bad, 0)


def test_invalid_configs():
    with pytest.raises(ConfigError, match="unknown config keys"):
        TrackingConfig.from_dict({"sauce": 0})
    with pytest.raises(ConfigError, match="exactly 2 points"):
        LineConfig(name="x", points=[(0, 0), (1, 1), (0.5, 0.5)])
    with pytest.raises(ConfigError, match="unique"):
        TrackingConfig.from_dict({"lines": [{"name": "a", "points": [[0, 0], [1, 1]]}],
                                  "zones": [{"name": "a", "points": [[0, 0], [1, 0], [1, 1]]}]})
    with pytest.raises(ConfigError, match="conf"):
        TrackingConfig(conf=0).validate()


def test_cli_flags_override_config_file():
    args = build_parser().parse_args(["run", "--config", str(EXAMPLE), "--source", "clip.mp4", "--classes", "car",
                                      "--line", "road=0,0.5,1,0.5", "--conf", "0.5"])
    cfg = config_from_args(args)
    assert cfg.source == "clip.mp4"
    assert cfg.classes == ["car"]
    assert cfg.conf == 0.5
    assert [line.name for line in cfg.lines] == ["road"]
    assert [z.name for z in cfg.zones] == ["waiting-area"]  # kept from the file
