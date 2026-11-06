"""Web dashboard: live annotated video (MJPEG) + JSON API with counts and events.

Endpoints
---------
GET /                 dashboard page
GET /stream.mjpg      live annotated video
GET /snapshot.jpg     latest annotated frame
GET /api/stats        current numbers (tracked now, unique, line counts, zones, FPS)
GET /api/events       most recent events, newest first (``?limit=50``)
GET /api/events/file  download the full event log file (when --events was given)
GET /healthz          liveness probe for Docker / Kubernetes
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from importlib import resources
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from .pipeline import Pipeline

log = logging.getLogger(__name__)

BOUNDARY = "frame"


def _dashboard_html() -> str:
    return resources.files("aitracking").joinpath("static/index.html").read_text(encoding="utf-8")


def create_app(pipeline: Pipeline, start_worker: bool = True) -> FastAPI:
    stop_event = threading.Event()
    worker: Optional[threading.Thread] = None

    def _work() -> None:
        try:
            pipeline.run(publish_jpeg=True, stop_event=stop_event)
        except Exception:  # status/error is recorded on the pipeline and shown in the dashboard
            log.exception("tracking pipeline stopped with an error")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal worker
        if start_worker:
            worker = threading.Thread(target=_work, name="aitracking-pipeline", daemon=True)
            worker.start()
        yield
        stop_event.set()
        if worker is not None:
            worker.join(timeout=10)

    app = FastAPI(title="AITracking", version=_version(), lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.stop_event = stop_event

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _dashboard_html()

    @app.get("/healthz")
    def healthz() -> dict:
        snap = pipeline.snapshot()
        return {"status": snap["status"], "frames": snap["frames"]}

    @app.get("/api/stats")
    def stats() -> JSONResponse:
        return JSONResponse(pipeline.snapshot())

    @app.get("/api/events")
    def events(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse([e.to_dict() for e in pipeline.events.recent(limit)])

    @app.get("/api/events/file")
    def events_file() -> FileResponse:
        path = pipeline.events.path
        if path is None or not path.exists():
            raise HTTPException(404, "no event file - start the server with --events events.csv")
        media = "text/csv" if path.suffix.lower() == ".csv" else "application/x-ndjson"
        return FileResponse(path, media_type=media, filename=path.name)

    @app.get("/snapshot.jpg")
    def snapshot() -> Response:
        _, jpeg = pipeline.wait_jpeg(-1, timeout=0)
        if jpeg is None:
            raise HTTPException(503, "no frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        def frames() -> Iterator[bytes]:
            seq = -1
            while not stop_event.is_set():
                new_seq, jpeg = pipeline.wait_jpeg(seq, timeout=2.0)
                if jpeg is None or new_seq == seq:
                    if pipeline.snapshot()["status"] in ("finished", "error"):
                        return
                    continue
                seq = new_seq
                yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    return app


def _version() -> str:
    from . import __version__

    return __version__


def serve(pipeline: Pipeline, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(pipeline), host=host, port=port, log_level="info")
