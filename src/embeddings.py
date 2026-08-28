import argparse
import asyncio
from pathlib import Path

import duckdb
import httpx
import numpy as np

TEXT_TABLES = {
    "asr_segments": "text",
    "captions": "text",
    "entities": "entity",
    "episodic_triples": "text",
    "semantic_triples": "text",
    "activities": "text",
    "event_chains": "text",
    "event_steps": "text",
    "topic_chains": "entity",
    "topic_facts": "text",
}
VIDEO_REQUEST_BATCH_SIZE = 1


async def request_embeddings(client: httpx.AsyncClient, endpoint: str, model: str, conversations: list[list[dict]], row_ids: list[str], media_io_kwargs: dict | None) -> list[tuple[str, list[float]]]:
    print(f"embedding_batch rows={len(row_ids)} first_row_id={row_ids[0]} last_row_id={row_ids[-1]}")
    body = {"model": model, "messages": conversations, "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True}
    if media_io_kwargs is not None:
        body["media_io_kwargs"] = media_io_kwargs
    response = await client.post(endpoint, json=body)
    response.raise_for_status()
    items = sorted(response.json()["data"], key=lambda item: item["index"])
    if [item["index"] for item in items] != list(range(len(row_ids))):
        raise RuntimeError(f"embedding response indexes do not match batch rows: {[item['index'] for item in items]}")
    return [(row_id, item["embedding"]) for row_id, item in zip(row_ids, items, strict=True)]


async def embed_text(database_path: Path, endpoint: str, model: str, batch_size: int, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = []
    for table, text_column in TEXT_TABLES.items():
        for row_id, text in connection.execute(f"SELECT row_id, {text_column} FROM {table} ORDER BY row_id").fetchall():
            rows.append((table, row_id, text))
    connection.close()
    shard_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for shard_index, start in enumerate(range(0, len(rows), batch_size)):
            shard_path = shard_dir / f"text-{shard_index:06d}.npz"
            if shard_path.exists():
                continue
            batch = rows[start:start + batch_size]
            conversations = [[
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ] for _, _, text in batch]
            results = await request_embeddings(client, endpoint, model, conversations, [row_id for _, row_id, _ in batch], None)
            np.savez(shard_path, tables=np.array([table for table, _, _ in batch]), row_ids=np.array([row_id for _, row_id, _ in batch]), vectors=np.array([vector for _, vector in results], dtype=np.float32))
            print(f"text_embedding_shard path={shard_path} rows={len(batch)}")


async def embed_video_batch(client: httpx.AsyncClient, endpoint: str, model: str, batch: list[tuple[str, Path, Path]], semaphore: asyncio.Semaphore) -> None:
    try:
        conversations = [[
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": [{"type": "video_url", "video_url": {"url": clip_path.resolve().as_uri()}}, {"type": "text", "text": ""}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ] for _, clip_path, _ in batch]
        try:
            results = await request_embeddings(client, endpoint, model, conversations, [row_id for row_id, _, _ in batch], None)
        except httpx.HTTPStatusError as error:
            payload = error.response.json()
            if error.response.status_code != 500 or payload["exception_type"] != "RuntimeError" or payload["detail"] != "Requested next frame while there are no more frames left to decode.":
                raise
            for row_id, clip_path, _ in batch:
                print(f"video_embedding_skip row_id={row_id} clip_path={clip_path} exception_type={payload['exception_type']} detail={payload['detail']}")
            return
        for (row_id, vector), (_, _, shard_path) in zip(results, batch, strict=True):
            np.savez(shard_path, row_ids=np.array([row_id]), vectors=np.array([vector], dtype=np.float32))
            print(f"video_embedding_shard path={shard_path} rows=1")
    finally:
        semaphore.release()


async def embed_video(database_path: Path, endpoint: str, model: str, max_in_flight: int, shard_dir: Path, clip_dir: Path) -> None:
    if max_in_flight != 64:
        raise ValueError(f"video max in-flight requests must be 64, got {max_in_flight}")
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute("SELECT c.row_id, c.video_id FROM video_clips c JOIN videos v USING(video_id) ORDER BY v.sequence_index, c.start_sec").fetchall()
    connection.close()
    shard_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=max_in_flight, max_keepalive_connections=max_in_flight)
    semaphore = asyncio.Semaphore(max_in_flight)
    pending = []
    for shard_index, (row_id, video_id) in enumerate(rows):
        shard_path = shard_dir / f"video-{shard_index:06d}.npz"
        if shard_path.exists():
            shard = np.load(shard_path)
            if shard["row_ids"].tolist() != [row_id]:
                raise RuntimeError(f"video shard row mismatch path={shard_path} expected={row_id} actual={shard['row_ids'].tolist()}")
            continue
        clip_index = int(row_id.rsplit(":", 1)[1])
        source_video_id = row_id.split(":", 1)[0]
        clip_path = clip_dir / source_video_id / f"{clip_index:06d}.mp4"
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        pending.append((row_id, clip_path, shard_path))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async with asyncio.TaskGroup() as tasks:
            for start in range(0, len(pending), VIDEO_REQUEST_BATCH_SIZE):
                batch = pending[start:start + VIDEO_REQUEST_BATCH_SIZE]
                await semaphore.acquire()
                tasks.create_task(embed_video_batch(client, endpoint, model, batch, semaphore))


def build_numpy_index(shard_dir: Path, index_dir: Path) -> None:
    counts: dict[str, int] = {}
    row_id_lengths: dict[str, int] = {}
    embedding_dimension = 0
    for path in sorted(shard_dir.glob("text-*.npz")):
        with np.load(path) as shard:
            tables = shard["tables"].astype(str)
            row_ids = shard["row_ids"].astype(str)
            vectors = shard["vectors"]
            if embedding_dimension == 0:
                embedding_dimension = vectors.shape[1]
            elif vectors.shape[1] != embedding_dimension:
                raise ValueError(f"embedding dimension mismatch path={path} expected={embedding_dimension} actual={vectors.shape[1]}")
            for table_value in np.unique(tables):
                table = str(table_value)
                selected = row_ids[tables == table]
                if table not in counts:
                    counts[table] = 0
                    row_id_lengths[table] = 0
                counts[table] += len(selected)
                row_id_lengths[table] = max(row_id_lengths[table], max(len(row_id) for row_id in selected))
    for path in sorted(shard_dir.glob("video-*.npz")):
        with np.load(path) as shard:
            row_ids = shard["row_ids"].astype(str)
            vectors = shard["vectors"]
            if embedding_dimension == 0:
                embedding_dimension = vectors.shape[1]
            elif vectors.shape[1] != embedding_dimension:
                raise ValueError(f"embedding dimension mismatch path={path} expected={embedding_dimension} actual={vectors.shape[1]}")
            if "video_clips" not in counts:
                counts["video_clips"] = 0
                row_id_lengths["video_clips"] = 0
            counts["video_clips"] += len(row_ids)
            row_id_lengths["video_clips"] = max(row_id_lengths["video_clips"], max(len(row_id) for row_id in row_ids))
    if embedding_dimension == 0:
        raise RuntimeError(f"no embedding shards found in {shard_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)
    matrices = {table: np.lib.format.open_memmap(index_dir / f"{table}.vectors.npy", mode="w+", dtype=np.float32, shape=(count, embedding_dimension)) for table, count in counts.items()}
    row_id_arrays = {table: np.lib.format.open_memmap(index_dir / f"{table}.row_ids.npy", mode="w+", dtype=f"<U{row_id_lengths[table]}", shape=(count,)) for table, count in counts.items()}
    positions = {table: 0 for table in counts}

    for path in sorted(shard_dir.glob("text-*.npz")):
        with np.load(path) as shard:
            tables = shard["tables"].astype(str)
            row_ids = shard["row_ids"].astype(str)
            vectors = shard["vectors"].astype(np.float32)
            for table_value in np.unique(tables):
                table = str(table_value)
                selected = tables == table
                table_vectors = vectors[selected]
                norms = np.linalg.norm(table_vectors, axis=1)
                if np.any(norms == 0):
                    raise ValueError(f"zero-norm embedding path={path} table={table}")
                start = positions[table]
                end = start + int(selected.sum())
                matrices[table][start:end] = table_vectors / norms[:, None]
                row_id_arrays[table][start:end] = row_ids[selected]
                positions[table] = end
        print(f"embedding_text_shard_indexed path={path}")
    for path in sorted(shard_dir.glob("video-*.npz")):
        with np.load(path) as shard:
            row_ids = shard["row_ids"].astype(str)
            vectors = shard["vectors"].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"zero-norm embedding path={path} table=video_clips")
            start = positions["video_clips"]
            end = start + len(row_ids)
            matrices["video_clips"][start:end] = vectors / norms[:, None]
            row_id_arrays["video_clips"][start:end] = row_ids
            positions["video_clips"] = end
        print(f"embedding_video_shard_indexed path={path}")
    if positions != counts:
        raise RuntimeError(f"embedding index row counts do not match expected counts expected={counts} actual={positions}")
    for table in counts:
        matrices[table].flush()
        row_id_arrays[table].flush()
        print(f"embedding_numpy_index_written table={table} rows={counts[table]} dimensions={embedding_dimension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-batch-size", type=int, required=True)
    parser.add_argument("--video-max-in-flight", type=int, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(embed_text(arguments.database, arguments.endpoint, arguments.model, arguments.text_batch_size, arguments.shard_dir))
    if not arguments.text_only:
        asyncio.run(embed_video(arguments.database, arguments.endpoint, arguments.model, arguments.video_max_in_flight, arguments.shard_dir, arguments.clip_dir))
    build_numpy_index(arguments.shard_dir, arguments.index_dir)
