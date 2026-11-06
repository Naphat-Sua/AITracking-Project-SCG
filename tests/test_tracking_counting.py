from tracking_helpers import H, W, combine, walker

from aitracking.config import LineConfig, ZoneConfig
from aitracking.counting import LineCounter, ZoneMonitor

VERTICAL = LineConfig(name="door", points=[(0.5, 0.0), (0.5, 1.0)])


def run_counter(counter, script, frames, fps=10.0):
    events = []
    for i in range(frames):
        events.extend(counter.update(script(i), W, H, i, i / fps))
    return events


def test_left_to_right_is_in_and_right_to_left_is_out():
    counter = LineCounter(VERTICAL)
    script = combine(walker(1, (100, 300), (540, 300), 20), walker(2, (540, 400), (100, 400), 20, cls="car"))
    events = run_counter(counter, script, 20)

    assert sorted((e.track_id, e.direction) for e in events) == [(1, "in"), (2, "out")]
    snap = counter.snapshot()
    assert snap["in"] == 1 and snap["out"] == 1 and snap["net"] == 0
    assert snap["by_class"] == {"in": {"person": 1}, "out": {"car": 1}}


def test_swapping_points_swaps_direction():
    counter = LineCounter(LineConfig(name="door", points=[(0.5, 1.0), (0.5, 0.0)]))
    events = run_counter(counter, walker(1, (100, 300), (540, 300), 20), 20)
    assert [e.direction for e in events] == ["out"]


def test_count_once_ignores_jitter_back_and_forth():
    def jitter(i):
        x = 300 if i % 2 == 0 else 340  # flips across x=320 every frame
        return walker(1, (x, 300), (x, 300), 1)(0)

    once = LineCounter(VERTICAL, count_once=True)
    assert len(run_counter(once, jitter, 10)) == 1

    every = LineCounter(VERTICAL, count_once=False)
    assert len(run_counter(every, jitter, 10)) == 9


def test_passing_around_the_end_of_a_short_line_is_not_counted():
    short = LineCounter(LineConfig(name="gate", points=[(0.5, 0.0), (0.5, 0.4)]))  # top 40% of the frame only
    events = run_counter(short, walker(1, (100, 400), (540, 400), 20), 20)  # walks across the bottom
    assert events == []


def test_horizontal_line_bottom_to_top_is_in():
    counter = LineCounter(LineConfig(name="road", points=[(0.0, 0.5), (1.0, 0.5)]))
    events = run_counter(counter, walker(7, (320, 460), (320, 60), 20, cls="truck"), 20)
    assert [(e.direction, e.cls) for e in events] == [("in", "truck")]


def test_zone_enter_exit_and_dwell():
    zone = ZoneMonitor(ZoneConfig(name="dock", points=[(0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.4, 1.0)]))
    # walks through the middle 20% of the frame: inside from x=256 to x=384
    events = run_counter(zone, walker(3, (100, 300), (540, 300), 23), 23, fps=10.0)
    kinds = [e.kind for e in events]
    assert kinds == ["zone_enter", "zone_exit"]
    exit_event = events[1]
    assert exit_event.dwell_s is not None and 0.4 <= exit_event.dwell_s <= 0.8
    snap = zone.snapshot(t=2.3)
    assert snap["occupancy"] == 0 and snap["entries"] == 1 and snap["avg_dwell_s"] == exit_event.dwell_s


def test_zone_reports_exit_when_track_disappears_inside():
    zone = ZoneMonitor(ZoneConfig(name="dock", points=[(0, 0), (1, 0), (1, 1), (0, 1)]), lost_after_frames=5)
    still = walker(9, (320, 300), (320, 300), 10)  # stands inside for 10 frames then vanishes
    events = run_counter(zone, still, 20)
    assert [e.kind for e in events] == ["zone_enter", "zone_exit"]
    assert zone.snapshot(t=2.0)["occupancy"] == 0


def test_zone_occupancy_by_class():
    zone = ZoneMonitor(ZoneConfig(name="all", points=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    script = combine(walker(1, (100, 300), (100, 300), 5), walker(2, (200, 300), (200, 300), 5, cls="car"))
    run_counter(zone, script, 3)
    assert zone.occupancy() == {"person": 1, "car": 1}
