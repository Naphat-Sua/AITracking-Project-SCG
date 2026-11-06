import time

from fastapi.testclient import TestClient
from tracking_helpers import ScriptedTracker, combine, walker, write_video

from aitracking.config import LineConfig, TrackingConfig
from aitracking.events import EventLog
from aitracking.pipeline import Pipeline
from aitracking.server import create_app

SCRIPT = combine(walker(1, (60, 300), (580, 300), 20), walker(2, (580, 420), (60, 420), 20, cls="car"))


def test_dashboard_and_api_end_to_end(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video, frames=20)
    events_path = tmp_path / "events.csv"
    config = TrackingConfig(source=str(video), lines=[LineConfig(name="door", points=[(0.5, 0), (0.5, 1)])])
    pipe = Pipeline(config, tracker=ScriptedTracker(SCRIPT), events=EventLog(events_path))

    with TestClient(create_app(pipe)) as client:
        deadline = time.time() + 20
        while pipe.snapshot()["status"] not in ("finished", "error") and time.time() < deadline:
            time.sleep(0.05)

        stats = client.get("/api/stats").json()
        assert stats["status"] == "finished", stats
        assert stats["frames"] == 20
        assert stats["lines"][0]["in"] == 1 and stats["lines"][0]["out"] == 1

        events = client.get("/api/events?limit=10").json()
        assert {e["direction"] for e in events if e["kind"] == "line_cross"} == {"in", "out"}

        page = client.get("/")
        assert page.status_code == 200 and "AITracking" in page.text

        snap = client.get("/snapshot.jpg")
        assert snap.status_code == 200 and snap.content[:2] == b"\xff\xd8"  # JPEG magic

        dl = client.get("/api/events/file")
        assert dl.status_code == 200 and b"line_cross" in dl.content

        assert client.get("/healthz").json()["status"] == "finished"


def test_snapshot_503_before_first_frame():
    pipe = Pipeline(TrackingConfig(), tracker=ScriptedTracker(SCRIPT))
    with TestClient(create_app(pipe, start_worker=False)) as client:
        assert client.get("/snapshot.jpg").status_code == 503
        assert client.get("/api/events/file").status_code == 404
        assert client.get("/api/stats").json()["status"] == "starting"
