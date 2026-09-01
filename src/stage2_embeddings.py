import argparse
import asyncio
import json
from pathlib import Path

import duckdb
import httpx
import numpy as np
import wandb


MODEL = "qwen3-vl-embedding-8b"
WORKERS = 32
EMBEDDING_DIMENSION = 4096


class EmbeddingOutputError(RuntimeError):
    pass


def load_shard(path: Path, expected_row_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as shard:
        row_ids = shard["row_ids"]
        vectors = shard["vectors"]
    if row_ids.tolist() != [expected_row_id]:
        raise RuntimeError(f"Embedding shard row ID mismatch in {path}")
    if vectors.shape != (1, EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid embedding shard in {path}")
    return vectors[0]


async def build_table(table: str, rows: list[tuple], base_url: str, shard_root: Path, index_root: Path, run_root: Path, kind: str) -> int:
    table_shard_root = shard_root / kind / table
    table_shard_root.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(max_connections=WORKERS, max_keepalive_connections=WORKERS)
    semaphore = asyncio.Semaphore(WORKERS)
    async with httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits) as client:
        async def embed_one(position: int, row: tuple) -> None:
            row_id = row[0]
            shard_path = table_shard_root / f"{position:08d}.npz"
            if shard_path.exists():
                load_shard(shard_path, row_id)
                return
            if kind == "text":
                payload = {"model": MODEL, "input": [row[4]]}
            else:
                clip_path = (Path.cwd() / row[5]).resolve()
                payload = {"model": MODEL, "input": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": clip_path.as_uri()}}]}]}
            last_error = None
            for attempt in range(1, 6):
                trace_path = run_root / "traces" / "embeddings" / kind / table / f"{position:08d}" / f"attempt-{attempt}.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    async with semaphore:
                        response = await client.post("/v1/embeddings", json=payload)
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    last_error = error
                    trace_path.write_text(json.dumps({"request": payload, "status_code": error.response.status_code, "response": error.response.text}, ensure_ascii=False, indent=2))
                    print(json.dumps({"stage": "embedding", "kind": kind, "table": table, "row_id": row_id, "attempt": attempt, "status_code": error.response.status_code, "response": error.response.text}))
                    continue
                except httpx.RequestError as error:
                    last_error = error
                    trace_path.write_text(json.dumps({"request": payload, "error": str(error)}, ensure_ascii=False, indent=2))
                    print(json.dumps({"stage": "embedding", "kind": kind, "table": table, "row_id": row_id, "attempt": attempt, "error": str(error)}))
                    continue
                try:
                    data = response.json()["data"]
                    if len(data) != 1 or data[0]["index"] != 0:
                        raise EmbeddingOutputError(f"Unexpected embedding response for {row_id}")
                    vector = np.asarray(data[0]["embedding"], dtype=np.float32)
                    if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
                        raise EmbeddingOutputError(f"Invalid embedding response for {row_id}")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, EmbeddingOutputError) as error:
                    last_error = error
                    trace_path.write_text(json.dumps({"request": payload, "response": response.text, "error": str(error)}, ensure_ascii=False, indent=2))
                    print(json.dumps({"stage": "embedding", "kind": kind, "table": table, "row_id": row_id, "attempt": attempt, "error": str(error)}))
                    continue
                np.savez_compressed(shard_path, row_ids=np.asarray([row_id]), vectors=vector[None, :])
                trace_path.write_text(json.dumps({"request": payload, "response": {"index": data[0]["index"], "dimension": len(data[0]["embedding"]), "norm": float(np.linalg.norm(vector)), "shard": str(shard_path)}}, ensure_ascii=False, indent=2))
                return
            raise RuntimeError(f"Embedding failed after 5 attempts for {row_id}: {last_error}")
        tasks = [asyncio.create_task(embed_one(position, row)) for position, row in enumerate(rows)]
        for completed, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if completed % 500 == 0 or completed == len(tasks):
                print(json.dumps({"stage": "embedding", "kind": kind, "table": table, "completed": completed, "total": len(tasks)}))

    row_ids = [row[0] for row in rows]
    row_id_array = np.lib.format.open_memmap(index_root / f"{table}.row_ids.npy", mode="w+", dtype=f"<U{max(len(row_id) for row_id in row_ids)}", shape=(len(row_ids),))
    vector_array = np.lib.format.open_memmap(index_root / f"{table}.vectors.npy", mode="w+", dtype=np.float32, shape=(len(row_ids), EMBEDDING_DIMENSION))
    for position, row_id in enumerate(row_ids):
        vector = load_shard(table_shard_root / f"{position:08d}.npz", row_id)
        row_id_array[position] = row_id
        vector_array[position] = vector
    row_id_array.flush()
    vector_array.flush()
    return len(rows)


async def build_embeddings(database_path: Path, base_url: str, shard_root: Path, index_root: Path, run_root: Path, kind: str) -> dict[str, int]:
    shard_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path), read_only=True)
    order = "CASE video_id WHEN 'day' THEN 0 WHEN 'week' THEN 1 WHEN 'month' THEN 2 END"
    text_queries = {
        "episodes": f"SELECT row_id, video_id, start_s, end_s, text FROM episodes ORDER BY {order}, start_s, row_id",
        "entities": f"SELECT row_id, video_id, first_seen_s, last_seen_s, name FROM entities ORDER BY {order}, first_seen_s, row_id",
        "semantic_triples": f"SELECT row_id, video_id, snapshot_s, snapshot_s, text FROM semantic_triples ORDER BY {order}, snapshot_s, row_id",
        "topic_facts": f"SELECT row_id, video_id, time_s, time_s, fact FROM topic_facts ORDER BY {order}, time_s, row_id",
        "event_steps": f"SELECT row_id, video_id, start_s, end_s, description FROM event_steps ORDER BY {order}, start_s, row_id",
        "object_states": f"SELECT row_id, video_id, start_s, end_s, text FROM object_states ORDER BY {order}, start_s, row_id",
    }
    metrics = {}
    if kind == "text":
        for table, query in text_queries.items():
            rows = connection.execute(query).fetchall()
            if not rows:
                raise RuntimeError(f"Searchable table is empty: {table}")
            metrics[table] = await build_table(table, rows, base_url, shard_root, index_root, run_root, "text")
    else:
        rows = connection.execute(f"SELECT row_id, video_id, start_s, end_s, text, filename FROM visual_clips ORDER BY {order}, start_s, row_id").fetchall()
        if not rows:
            raise RuntimeError("Searchable table is empty: visual_clips")
        metrics["visual_clips"] = await build_table("visual_clips", rows, base_url, shard_root, index_root, run_root, "video")
    connection.close()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kind", choices=["text", "video"], required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    args = parser.parse_args()
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name, job_type=f"stage2-embedding-{args.kind}", config={"database": str(args.database), "base_url": args.base_url, "shard_root": str(args.shard_root), "index_root": str(args.index_root), "run_root": str(args.run_root), "kind": args.kind, "model": MODEL, "workers": WORKERS, "dimension": EMBEDDING_DIMENSION})
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    metrics = asyncio.run(build_embeddings(args.database, args.base_url, args.shard_root, args.index_root, args.run_root, args.kind))
    run.log(metrics)
    run.finish()
    print(json.dumps({"stage": f"embedding-{args.kind}", "status": "completed", **metrics}))
