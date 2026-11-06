import time

import pytest
from tracking_helpers import write_video

from aitracking.config import ConfigError
from aitracking.sources import VideoSource, is_live


def test_is_live():
    assert is_live(0)
    assert is_live("rtsp://10.0.0.5/stream")
    assert is_live("HTTP://cam/video.mjpg")
    assert not is_live("clip.mp4")


def test_file_source_reads_every_frame_and_loops(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video, frames=5)
    src = VideoSource(str(video))
    assert src.frame_count == 5 and not src.live
    assert sum(1 for _ in iter(lambda: src.read()[0] or None, None)) == 5
    src.release()

    looping = VideoSource(str(video), loop=True)
    assert all(looping.read()[0] for _ in range(12))
    looping.release()


def test_live_source_hands_out_latest_frame_and_drops_backlog(tmp_path):
    """A slow consumer of a live stream gets fresh frames; old ones are dropped, not queued."""
    video = tmp_path / "in.mp4"
    write_video(video, frames=200)
    src = VideoSource(str(video), live=True, max_reconnects=0)  # treat the file like a camera
    ok, first = src.read()
    assert ok and first is not None
    time.sleep(0.3)  # simulate slow detection while the grabber keeps reading
    got = 1
    while src.read()[0]:
        got += 1
    src.release()
    assert got < 200, "a slow consumer should skip frames instead of building a backlog"
    assert src.dropped > 0


def test_missing_file_and_unopenable_stream():
    with pytest.raises(ConfigError, match="not found"):
        VideoSource("no-such-file.mp4")
