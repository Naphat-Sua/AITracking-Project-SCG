import csv
import json

import cv2
import pytest
from tracking_helpers import H, W, ScriptedTracker, blank_frame, combine, walker, write_video

from aitracking.cli import main
from aitracking.config import ConfigError, LineConfig, TrackingConfig, ZoneConfig
from aitracking.events import EventLog
from aitracking.pipeline import Pipeline


def make_config(**kw):
    base = dict(lines=[LineConfig(name="door", points=[(0.5, 0.0), (0.5, 1.0)])],
                zones=[ZoneConfig(name="left", points=[(0, 0), (0.3, 0), (0.3, 1), (0, 1)])])
    base.update(kw)
    return TrackingConfig(**base)


SCRIPT = combine(walker(1, (60, 300), (580, 300), 30), walker(2, (580, 420), (60, 420), 30, cls="car"))


def test_process_counts_and_annotates():
    pipe = Pipeline(make_config(), tracker=ScriptedTracker(SCRIPT))
    out = None
    for i in range(30):
        out = pipe.process(blank_frame(), i, i / 10)
    assert out.shape == (H, W, 3)
    assert out.any(), "annotations should draw on the blank frame"

    snap = pipe.snapshot()
    assert snap["unique"] == {"person": 1, "car": 1}
    line = snap["lines"][0]
    assert (line["in"], line["out"]) == (1, 1)
    zone = snap["zones"][0]
    assert zone["entries"] == 2
    kinds = sorted(e.kind for e in pipe.events.recent(100))
    assert kinds.count("line_cross") == 2 and kinds.count("zone_enter") == 2


def test_run_writes_video_events_and_summary(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video, frames=30)
    events_path = tmp_path / "events.csv"
    out_path = tmp_path / "out.mp4"

    pipe = Pipeline(make_config(source=str(video)), tracker=ScriptedTracker(SCRIPT), events=EventLog(events_path))
    summary = pipe.run(output=str(out_path))

    assert summary["status"] == "finished"
    assert summary["frames"] == 30
    assert summary["lines"][0]["in"] == 1

    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()

    with events_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["kind"] for r in rows} >= {"line_cross", "zone_enter"}
    assert all(r["wall_time"] for r in rows)


def test_jsonl_event_log(tmp_path):
    path = tmp_path / "events.jsonl"
    pipe = Pipeline(make_config(), tracker=ScriptedTracker(SCRIPT), events=EventLog(path))
    for i in range(30):
        pipe.process(blank_frame(), i, i / 10)
    pipe.events.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(r["kind"] == "line_cross" and r["direction"] == "in" for r in rows)


def test_max_frames_and_loop(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video, frames=10)
    pipe = Pipeline(make_config(source=str(video), loop=True, max_frames=25), tracker=ScriptedTracker(SCRIPT))
    assert pipe.run()["frames"] == 25


def test_missing_video_is_a_clear_error(tmp_path):
    pipe = Pipeline(make_config(source=str(tmp_path / "nope.mp4")), tracker=ScriptedTracker(SCRIPT))
    with pytest.raises(ConfigError, match="not found"):
        pipe.run()


def test_cli_reports_config_errors(capsys):
    assert main(["run", "--source", "does-not-exist.mp4"]) == 2
    assert "not found" in capsys.readouterr().err
