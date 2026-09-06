FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/video_sql_v2

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY skills ./skills

ENTRYPOINT [".venv/bin/python"]
