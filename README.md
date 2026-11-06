# AITracking

**Real-time AI object tracking and counting for CCTV, webcams and video files.**
AITracking detects people and vehicles with YOLO, gives every object a persistent
ID with ByteTrack, counts who crosses a line (in / out), measures how many are
inside a zone and how long they stay, and shows it all on a live web dashboard
with a JSON API and CSV event log.

Built for SCG Digital Office. Runs on an ordinary laptop CPU and uses an NVIDIA GPU automatically when one is present.

```
camera / RTSP / video ─► YOLO detection ─► ByteTrack IDs ─► line & zone counting ─► dashboard · API · CSV · MP4
```

![AITracking dashboard: live video with tracked people, a counting line and a zone, plus counters and an event feed](docs/dashboard.jpg)

## Features

- **Any source** – video files, USB webcams (`0`), IP cameras over RTSP/HTTP; streams reconnect automatically when they drop.
- **Multi-object tracking** – stable IDs per object with ByteTrack (or BoT-SORT), movement trails on screen.
- **Line counting** – in / out / net per line and per class (person, car, truck…), each object counted once.
- **Zones** – live occupancy, total entries and average time spent inside (dwell time) for any polygon.
- **Live dashboard** – annotated video plus counters and an event feed at `http://localhost:8000`.
- **Data out** – REST/JSON API, every event appended to `.csv` or `.jsonl`, annotated `.mp4` recording, final JSON summary.
- **Resolution independent config** – lines and zones are written as 0..1 fractions of the frame in one YAML file.
- **Custom models** – drop in any Ultralytics YOLO weights (e.g. a model trained on helmets, forklifts or pallets).
- **Docker image** and a test suite that runs in CI without a GPU or model download.

## Quick start

Requires Python 3.9 – 3.12.

```bash
git clone https://github.com/Naphat-Sua/AITracking-Project-SCG.git
cd AITracking-Project-SCG
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only PyTorch is a much smaller download; skip this line if you have an NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The default model (`yolo11n.pt`, ~5 MB) is downloaded automatically on first run.

**Live dashboard from a webcam**

```bash
aitrack serve --source 0 --config configs/example.yaml
# open http://localhost:8000
```

**IP camera (RTSP)**

```bash
aitrack serve --source "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101" \
              --config configs/example.yaml --events data/events.csv --host 0.0.0.0
```

**Process a recorded video and save the results**

```bash
aitrack run --source entrance.mp4 --classes person \
            --line "door=0.5,0,0.5,1" --zone "queue=0.05,0.55;0.4,0.55;0.4,0.95;0.05,0.95" \
            --output entrance-tracked.mp4 --events events.csv --summary summary.json
```

`aitrack run` prints a JSON summary when it finishes. This is the real (shortened)
output for OpenCV's 795-frame pedestrian sample
[`vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
with a vertical line through the middle and a zone on the left:

```json
{
  "status": "finished", "frames": 795, "fps": 4.4,
  "unique": {"person": 34}, "unique_total": 34,
  "lines": [{"name": "middle", "in": 4, "out": 16, "net": -12}],
  "zones": [{"name": "left-side", "occupancy": 1, "entries": 14, "avg_dwell_s": 4.34}],
  "events_total": 47
}
```

Add `--show` to watch a live window while it runs (press `q` to stop), and
`python -m aitracking ...` works the same as `aitrack ...`.

## Configuration

Everything can be set in a YAML file ([`configs/example.yaml`](configs/example.yaml));
command-line flags override the file.

```yaml
source: 0                         # file path, webcam index or rtsp:// URL
model: yolo11n.pt                 # yolo11s.pt / yolo11m.pt = more accurate, slower
classes: [person, car, truck, motorcycle]
conf: 0.35
lines:
  - name: entrance
    points: [[0.5, 0.05], [0.5, 0.95]]          # x, y as fractions of the frame
zones:
  - name: waiting-area
    points: [[0.05, 0.55], [0.4, 0.55], [0.4, 0.95], [0.05, 0.95]]
```

**Which way is "in"?** Imagine walking along the line from its first point to its
second: crossing towards your left-hand side is `in`. In practice:

| Line drawn | `in` means moving |
| --- | --- |
| top → bottom (vertical) | left → right |
| left → right (horizontal) | bottom → top |

Swap the two points to flip it. Objects are counted at the bottom-centre of their
box (where they touch the floor); use `anchor: center` for top-down cameras.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--source`, `-s` | video file, webcam index, or stream URL | `0` |
| `--config`, `-c` | YAML config file | – |
| `--model`, `-m` | YOLO weights | `yolo11n.pt` |
| `--classes` | comma separated class names | all |
| `--conf` | minimum detection confidence | `0.35` |
| `--imgsz` | inference size (lower = faster) | `640` |
| `--device` | `cpu`, `0` (GPU 0), `mps` | auto |
| `--tracker` | `bytetrack.yaml` or `botsort.yaml` | `bytetrack.yaml` |
| `--line` | `[name=]x1,y1,x2,y2` – repeatable | – |
| `--zone` | `[name=]x1,y1;x2,y2;x3,y3…` – repeatable | – |
| `--events`, `-e` | append events to `.csv` / `.jsonl` | – |
| `--loop` | restart a video file at the end (demos) | off |
| `--max-frames` | stop after N frames | – |
| `run --output`, `-o` | save annotated video | – |
| `run --summary` | save final counts as JSON | – |
| `serve --host` / `--port` | dashboard address | `127.0.0.1:8000` |

## Dashboard & API

`aitrack serve` starts the dashboard and these endpoints:

| Endpoint | Returns |
| --- | --- |
| `GET /` | dashboard page |
| `GET /stream.mjpg` | live annotated video (embed it in any `<img>` tag) |
| `GET /snapshot.jpg` | latest annotated frame |
| `GET /api/stats` | tracked now, unique objects, FPS, line counts, zone occupancy |
| `GET /api/events?limit=50` | most recent events, newest first |
| `GET /api/events/file` | download the full event log (needs `--events`) |
| `GET /healthz` | liveness probe |
| `GET /docs` | interactive API docs |

An event looks like this (also the CSV columns):

```json
{"kind": "line_cross", "name": "entrance", "track_id": 12, "cls": "person", "frame": 431,
 "t": 17.24, "direction": "in", "dwell_s": null, "wall_time": "2026-11-06T09:15:02+07:00"}
```

`kind` is `line_cross`, `zone_enter` or `zone_exit` (`zone_exit` carries `dwell_s`).

> The dashboard has no login. It binds to `127.0.0.1` by default; only use
> `--host 0.0.0.0` on a trusted network or behind a reverse proxy with authentication.

## Docker

```bash
docker build -t aitracking .
docker run --rm -p 8000:8000 -v "$PWD/data:/data" aitracking \
  serve --host 0.0.0.0 --source "rtsp://user:pass@camera/stream" \
        --config /app/configs/example.yaml --events /data/events.csv
```

The image uses CPU PyTorch and has the default weights baked in, so it starts offline.

## Tips for real deployments

- **Speed:** the model is the bottleneck. Measured on a low-power laptop CPU (Intel i5-10210U), `yolo11n.pt` takes ~170 ms per frame at `--imgsz 640` (~6 FPS) and ~125 ms at `--imgsz 480` (~8 FPS); desktop CPUs are several times faster and an NVIDIA GPU reaches real-time easily. That is enough for counting people walking past a door. For live cameras AITracking always processes the *newest* frame and skips the backlog, so the dashboard stays real-time even when detection is slower than the camera (`dropped_frames` in `/api/stats` shows how many were skipped). Run one `aitrack serve` process per camera on different ports; use a GPU for `yolo11s/m` or many cameras.
- **Accuracy:** mount cameras high and angled down, put the counting line where people walk across it, not along it, and keep it away from the frame edges so objects are tracked for a few frames before they cross.
- **Custom objects:** train with Ultralytics (`yolo detect train data=my.yaml model=yolo11n.pt`) and pass `--model runs/detect/train/weights/best.pt --classes helmet,no-helmet`.
- **Privacy:** only counts and events are stored, never images, unless you pass `--output`. Follow your site's CCTV / PDPA policy before recording.

## Project layout

```
aitracking/            the application
  cli.py               `aitrack run` / `aitrack serve`
  pipeline.py          frame loop: track -> count -> draw -> outputs
  tracker.py           YOLO + ByteTrack wrapper (the only Ultralytics-specific file)
  counting.py          line in/out counter and zone occupancy / dwell time
  geometry.py          line / polygon math
  events.py            event model and CSV / JSONL log
  server.py, static/   FastAPI dashboard and API
configs/               example config
lessons/               the original machine-learning lessons (regression, CNN, RNN, GAN, ...)
tests/                 pytest suite for the app and the lessons
```

## Development

```bash
pip install -e ".[dev]" -r lessons/requirements.txt
ruff check .
pytest
```

The app tests use a scripted fake tracker and synthetic videos, so they need no
GPU and no model download. CI runs lint and tests on Python 3.9 – 3.12 for every
push and pull request.

## Machine-learning lessons

The repository started as a collection of hands-on ML / deep-learning lessons.
They are still here, cleaned up and tested, in [`lessons/`](lessons/README.md).

## Licence

AITracking uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is **AGPL-3.0**. Internal use is fine; exposing a modified version to
outside users or shipping it in a product requires either publishing the source
or an Ultralytics Enterprise licence. See [`NOTICES.md`](NOTICES.md) for details,
third-party attributions, and the note on this repository's own licence.
