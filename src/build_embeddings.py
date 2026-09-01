import argparse
import asyncio
import base64
import json
from pathlib import Path

import duckdb
import httpx
import numpy as np


MODEL = "qwen3-vl-embedding-8b"
TEXT_BATCH_SIZE = 512
VIDEO_CONCURRENCY = 64
EMBEDDING_DIMENSION = 4096


async def embed_video(client: httpx.AsyncClient, row: tuple, clip_root: Path, shard_dir: Path) -> None:
    row_id, video_id, _, _, _ = row
    video_shard_dir = shard_dir / video_id
    video_shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = video_shard_dir / f"{int(row_id.rsplit(':', 1)[1]):06d}.npz"
    if shard_path.exists():
        with np.load(shard_path, allow_pickle=False) as shard:
            if shard["row_ids"].tolist() != [row_id]:
                raise RuntimeError(f"Shard row ID mismatch in {shard_path}")
            vector = shard["vectors"]
        if vector.shape != (1, EMBEDDING_DIMENSION) or not np.isfinite(vector).all():
            raise RuntimeError(f"Invalid embedding shard {shard_path}")
        return
    clip_path = clip_root / video_id / f"{int(row_id.rsplit(':', 1)[1]):06d}.mp4"
    video_url = "data:video/mp4;base64," + base64.b64encode(clip_path.read_bytes()).decode("ascii")
    payload = {"model": MODEL, "input": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": video_url}}]}]}
    try:
        response = await client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        print(json.dumps({"kind": "video", "row_id": row_id, "clip_path": str(clip_path), "status_code": error.response.status_code, "response": error.response.text}))
        raise
    except httpx.RequestError as error:
        print(json.dumps({"kind": "video", "row_id": row_id, "clip_path": str(clip_path), "error": str(error)}))
        raise
    data = response.json()["data"]
    if len(data) != 1 or data[0]["index"] != 0:
        raise RuntimeError(f"Unexpected embedding response for {row_id}")
    vector = np.asarray(data[0]["embedding"], dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
        raise RuntimeError(f"Invalid video embedding for {row_id}")
    np.savez_compressed(shard_path, row_ids=np.asarray([row_id]), vectors=vector[None, :])


async def build_video_embeddings(rows: list[tuple], base_url: str, clip_root: Path, shard_dir: Path) -> None:
    limits = httpx.Limits(max_connections=VIDEO_CONCURRENCY, max_keepalive_connections=VIDEO_CONCURRENCY)
    async with httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits) as client:
        for offset in range(0, len(rows), VIDEO_CONCURRENCY):
            batch = rows[offset:offset + VIDEO_CONCURRENCY]
            await asyncio.gather(*(embed_video(client, row, clip_root, shard_dir) for row in batch))
            print(json.dumps({"table": "visual_clips", "embedded": min(offset + VIDEO_CONCURRENCY, len(rows)), "total": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--kind", choices=["text", "video"], required=True)
    args = parser.parse_args()

    args.shard_root.mkdir(parents=True, exist_ok=True)
    args.index_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(args.database), read_only=True)
    queries = {
        "episodes": "SELECT row_id, video_id, start_s, end_s, text FROM episodes ORDER BY CAST(video_id AS INTEGER), start_s, row_id",
        "entities": "SELECT row_id, video_id, first_seen_s, last_seen_s, name FROM entities ORDER BY CAST(video_id AS INTEGER), first_seen_s, row_id",
        "semantic_triples": "SELECT row_id, video_id, snapshot_s, snapshot_s, text FROM semantic_triples ORDER BY CAST(video_id AS INTEGER), snapshot_s, row_id",
        "openie_triples": "SELECT o.row_id, o.video_id, e.start_s, e.end_s, o.text FROM openie_triples o JOIN episodes e ON e.row_id = o.episode_row_id ORDER BY CAST(o.video_id AS INTEGER), e.start_s, o.row_id",
        "topic_facts": "SELECT row_id, video_id, time_s, time_s, fact FROM topic_facts ORDER BY CAST(video_id AS INTEGER), time_s, row_id",
        "event_steps": "SELECT row_id, video_id, start_s, end_s, description FROM event_steps ORDER BY CAST(video_id AS INTEGER), start_s, row_id",
    }

    if args.kind == "text":
        client = httpx.Client(base_url=args.base_url, timeout=None)
        for table, query in queries.items():
            rows = connection.execute(query).fetchall()
            shard_dir = args.shard_root / "text" / table
            shard_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(0, len(rows), TEXT_BATCH_SIZE):
                batch = rows[offset:offset + TEXT_BATCH_SIZE]
                shard_path = shard_dir / f"{offset // TEXT_BATCH_SIZE:06d}.npz"
                row_ids = [row[0] for row in batch]
                if shard_path.exists():
                    with np.load(shard_path, allow_pickle=False) as shard:
                        if shard["row_ids"].tolist() != row_ids:
                            raise RuntimeError(f"Shard row IDs mismatch in {shard_path}")
                        vectors = shard["vectors"]
                    if vectors.shape != (len(batch), EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
                        raise RuntimeError(f"Invalid embedding shard {shard_path}")
                    continue
                payload = {"model": MODEL, "input": [row[4] for row in batch]}
                try:
                    response = client.post("/v1/embeddings", json=payload)
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    print(json.dumps({"kind": "text", "table": table, "offset": offset, "rows": len(batch), "status_code": error.response.status_code, "response": error.response.text}))
                    raise
                except httpx.RequestError as error:
                    print(json.dumps({"kind": "text", "table": table, "offset": offset, "rows": len(batch), "error": str(error)}))
                    raise
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise RuntimeError(f"Unexpected embedding response for {table} offset {offset}")
                vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
                if vectors.shape != (len(batch), EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
                    raise RuntimeError(f"Invalid text embeddings for {table} offset {offset}")
                np.savez_compressed(shard_path, row_ids=np.asarray(row_ids), vectors=vectors)
            print(json.dumps({"table": table, "embedded": len(rows), "batch_size": TEXT_BATCH_SIZE}))
        client.close()
        tables = list(queries)
    else:
        rows = connection.execute("SELECT row_id, video_id, start_s, end_s, text FROM visual_clips ORDER BY CAST(video_id AS INTEGER), start_s, row_id").fetchall()
        shard_dir = args.shard_root / "video" / "visual_clips"
        shard_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(build_video_embeddings(rows, args.base_url, args.clip_root, shard_dir))
        tables = ["visual_clips"]
    for table in tables:
        if table == "visual_clips":
            rows = connection.execute("SELECT row_id, video_id FROM visual_clips ORDER BY CAST(video_id AS INTEGER), start_s, row_id").fetchall()
            shard_dir = args.shard_root / "video" / table
            shard_paths = [shard_dir / video_id / f"{int(row_id.rsplit(':', 1)[1]):06d}.npz" for row_id, video_id in rows]
        else:
            rows = connection.execute(queries[table]).fetchall()
            shard_dir = args.shard_root / "text" / table
            shard_paths = [shard_dir / f"{index // TEXT_BATCH_SIZE:06d}.npz" for index in range(0, len(rows), TEXT_BATCH_SIZE)]
        expected_ids = [row[0] for row in rows]
        row_id_array = np.lib.format.open_memmap(args.index_root / f"{table}.row_ids.npy", mode="w+", dtype=f"<U{max(len(row_id) for row_id in expected_ids)}", shape=(len(expected_ids),))
        vector_array = np.lib.format.open_memmap(args.index_root / f"{table}.vectors.npy", mode="w+", dtype=np.float32, shape=(len(expected_ids), EMBEDDING_DIMENSION))
        position = 0
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                shard_ids = shard["row_ids"]
                shard_vectors = shard["vectors"]
            if shard_ids.tolist() != expected_ids[position:position + len(shard_ids)]:
                raise RuntimeError(f"Consolidation row ID mismatch in {shard_path}")
            if shard_vectors.shape != (len(shard_ids), EMBEDDING_DIMENSION) or not np.isfinite(shard_vectors).all():
                raise RuntimeError(f"Invalid consolidation shard {shard_path}")
            row_id_array[position:position + len(shard_ids)] = shard_ids
            vector_array[position:position + len(shard_ids)] = shard_vectors
            position += len(shard_ids)
        if position != len(expected_ids):
            raise RuntimeError(f"Consolidation count mismatch for {table}: expected={len(expected_ids)}, actual={position}")
        row_id_array.flush()
        vector_array.flush()
        print(json.dumps({"table": table, "index_rows": len(expected_ids), "dimension": EMBEDDING_DIMENSION}))
    connection.close()
