FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/video_sql_v2
COPY pyproject.toml uv.lock ./
ENV UV_CACHE_DIR=/raid/uv-cache
ENV HF_HOME=/raid/huggingface
RUN uv sync --frozen
COPY . .

ENTRYPOINT ["uv", "run", "python", "-m", "src.training"]

