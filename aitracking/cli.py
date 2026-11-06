"""Command line interface.

    aitrack run   --source video.mp4 --line entrance=0.5,0,0.5,1 --output out.mp4 --events events.csv
    aitrack serve --source rtsp://user:pass@camera/stream --config configs/example.yaml --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from . import __version__
from .config import ConfigError, TrackingConfig, parse_classes, parse_line_arg, parse_source, parse_zone_arg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="YAML config file (flags below override it)")
    p.add_argument("--source", "-s", help="video file, webcam index (0) or stream URL (rtsp://, http://)")
    p.add_argument("--model", "-m", help="YOLO weights, e.g. yolo11n.pt (default), yolo11s.pt, or your own best.pt")
    p.add_argument("--tracker", help="bytetrack.yaml (default) or botsort.yaml")
    p.add_argument("--classes", help="comma separated class names to track, e.g. person,car,truck")
    p.add_argument("--conf", type=float, help="minimum detection confidence (default 0.35)")
    p.add_argument("--imgsz", type=int, help="inference size in pixels (default 640)")
    p.add_argument("--device", help="cpu, 0 (first GPU), mps ... (default: auto)")
    p.add_argument("--line", action="append", default=[], metavar="[NAME=]X1,Y1,X2,Y2",
                   help="counting line in 0..1 frame coordinates; repeatable")
    p.add_argument("--zone", action="append", default=[], metavar="[NAME=]X1,Y1;X2,Y2;X3,Y3...",
                   help="polygon zone in 0..1 frame coordinates; repeatable")
    p.add_argument("--anchor", choices=["bottom", "center"], help="box point used for counting (default bottom)")
    p.add_argument("--events", "-e", help="append every event to this .csv or .jsonl file")
    p.add_argument("--max-frames", type=int, help="stop after N frames")
    p.add_argument("--loop", action="store_true", default=None, help="restart a video file when it ends")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aitrack", description="Real-time object detection, tracking and counting.")
    parser.add_argument("--version", action="version", version=f"aitracking {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a video/camera and save the results")
    _add_common(run)
    run.add_argument("--output", "-o", help="save the annotated video here (e.g. out.mp4)")
    run.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    run.add_argument("--summary", help="write the final counts as JSON to this file")

    srv = sub.add_parser("serve", help="run the live web dashboard + JSON API")
    _add_common(srv)
    srv.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to allow other machines)")
    srv.add_argument("--port", type=int, default=8000)
    return parser


def config_from_args(args: argparse.Namespace) -> TrackingConfig:
    base = TrackingConfig.from_yaml(args.config) if args.config else TrackingConfig()
    overrides = {
        "source": parse_source(args.source) if args.source is not None else None,
        "model": args.model,
        "tracker": args.tracker,
        "classes": parse_classes(args.classes),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "anchor": args.anchor,
        "max_frames": args.max_frames,
        "loop": args.loop,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    if args.line:
        base.lines = [parse_line_arg(text, i) for i, text in enumerate(args.line)]
    if args.zone:
        base.zones = [parse_zone_arg(text, i) for i, text in enumerate(args.zone)]
    return base.validate()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = config_from_args(args)
        from .events import EventLog
        from .pipeline import Pipeline

        pipeline = Pipeline(config, events=EventLog(args.events))
        if args.command == "serve":
            from .server import serve

            if args.host not in ("127.0.0.1", "localhost"):
                logging.warning("dashboard is reachable from other machines on %s:%d - it has no login, "
                                "keep it on a trusted network", args.host, args.port)
            print(f"Dashboard: http://{'localhost' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}")
            serve(pipeline, host=args.host, port=args.port)
            return 0

        summary = pipeline.run(output=args.output, show=args.show, on_progress=_progress)
        sys.stderr.write("\n")
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        print(text)
        if args.summary:
            with open(args.summary, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _progress(done: int, total: Optional[int]) -> None:
    if done % 25 and done != total:
        return
    msg = f"\rprocessed {done}/{total} frames" if total else f"\rprocessed {done} frames"
    sys.stderr.write(msg)
    sys.stderr.flush()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
