"""Embed every MM-Lifelong week source file with qwen3-vl-embedding-8b on the verification embedder; one npz per source, resumable."""

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/egolife-week-shared/magic_index/visual_vectors"
MEDIA_MAP = Path("/raid/wg25r/video_sql_v2/data/runtime/week/media-map.json")
EMBED_URL = "http://127.0.0.1:8015/v1/embeddings"
CONCURRENCY = 8


async def embed_one(client, semaphore, index, source):
    target = OUT / f"{index:06d}.npz"
    if target.exists():
        return
    container_path = source["path"].replace("/raid/huggingface", "/hf_cache")
    payload = {"model": "qwen3-vl-embedding-8b", "input": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": f"file://{container_path}"}}]}]}
    async with semaphore:
        response = await client.post(EMBED_URL, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    if len(data) != 1 or data[0]["index"] != 0:
        raise RuntimeError(f"invalid embedding response indexes for {source['path']}: {data}")
    vector = np.asarray(data[0]["embedding"], dtype=np.float32)
    if vector.shape != (4096,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid vector for {source['path']}: {vector.shape}")
    partial = target.with_suffix(".partial.npz")
    np.savez(partial, row_ids=np.asarray([Path(source["path"]).name]), vectors=vector[None, :],
             timeline_start_s=np.asarray([source["timeline_start_s"]]), timeline_end_s=np.asarray([source["timeline_end_s"]]))
    partial.rename(target)
    print(f"embedded {index:06d} {Path(source['path']).name} norm={float(np.linalg.norm(vector)):.4f}", flush=True)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads(MEDIA_MAP.read_text())["sources"]
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=1800) as client:
        async with asyncio.TaskGroup() as group:
            for index, source in enumerate(sources):
                group.create_task(embed_one(client, semaphore, index, source))
    done = sorted(OUT.glob("[0-9]*.npz"))
    if len(done) != len(sources):
        raise RuntimeError(f"{len(done)} vectors for {len(sources)} sources")
    print(f"all {len(done)} week source files embedded", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
