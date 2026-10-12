FROM python:3.12-slim

# ffmpeg: voice-note transcription; DejaVu fonts: matplotlib in a headless container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY healthcheck.py ./
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/workspace /app/data \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    WORKSPACE_DIR=/app/workspace \
    DATA_DIR=/app/data \
    HEARTBEAT=/app/data/heartbeat

CMD ["python", "-m", "tgagent"]
