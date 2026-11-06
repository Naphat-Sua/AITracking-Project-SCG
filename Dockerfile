# CPU image for the AITracking dashboard.
#   docker build -t aitracking .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" aitracking \
#       serve --source rtsp://user:pass@camera/stream --config /app/configs/example.yaml --events /data/events.csv
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image ~4x smaller than the default CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY aitracking ./aitracking
COPY configs ./configs
RUN pip install . \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"  # bake the default weights into the image

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
ENTRYPOINT ["aitrack"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--config", "/app/configs/example.yaml"]
