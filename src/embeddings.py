import argparse
import asyncio
import base64
from pathlib import Path

import cv2
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


async def request_embedding(client: httpx.AsyncClient, endpoint: str, model: str, messages: list[dict], row_id: str, media_io_kwargs: dict | None) -> tuple[str, list[float]]:
    print(f"embedding_request row_id={row_id}")
    body = {"model": model, "messages": messages, "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True}
    if media_io_kwargs is not None:
        body["media_io_kwargs"] = media_io_kwargs
    response = await client.post(endpoint, json=body)
    response.raise_for_status()
    payload = response.json()
    return row_id, payload["data"][0]["embedding"]


async def embed_text(database_path: Path, endpoint: str, model: str, concurrency: int, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = []
    for table, text_column in TEXT_TABLES.items():
        for row_id, text in connection.execute(f"SELECT row_id, {text_column} FROM {table} ORDER BY row_id").fetchall():
            rows.append((table, row_id, text))
    connection.close()
    shard_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for shard_index, start in enumerate(range(0, len(rows), concurrency)):
            shard_path = shard_dir / f"text-{shard_index:06d}.npz"
            if shard_path.exists():
                continue
            batch = rows[start:start + concurrency]
            messages = [[
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ] for _, _, text in batch]
            results = await asyncio.gather(*(request_embedding(client, endpoint, model, prompt, row_id, None) for prompt, (_, row_id, _) in zip(messages, batch, strict=True)))
            np.savez(shard_path, tables=np.array([table for table, _, _ in batch]), row_ids=np.array([row_id for _, row_id, _ in batch]), vectors=np.array([vector for _, vector in results], dtype=np.float32))
            print(f"text_embedding_shard path={shard_path} rows={len(batch)}")


def clip_data_url(video_path: str, start_sec: float, end_sec: float) -> tuple[str, dict]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video fps video={video_path} fps={native_fps}")
    encoded_frames = []
    frame_indices = []
    for timestamp in np.arange(start_sec, end_sec, 0.5):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot read frame video={video_path} timestamp={timestamp}")
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            capture.release()
            raise RuntimeError(f"cannot encode frame video={video_path} timestamp={timestamp}")
        encoded_frames.append(base64.b64encode(buffer).decode())
        frame_indices.append(round((timestamp - start_sec) * native_fps))
    capture.release()
    duration = end_sec - start_sec
    data_url = "data:video/jpeg;base64," + ",".join(encoded_frames)
    media_io_kwargs = {"video": {"num_frames": -1, "fps": native_fps, "frames_indices": frame_indices, "total_num_frames": round(duration * native_fps), "duration": duration, "do_sample_frames": False}}
    return data_url, media_io_kwargs


async def embed_video(database_path: Path, endpoint: str, model: str, concurrency: int, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute("SELECT c.row_id, v.video_path, c.start_sec, c.end_sec FROM video_clips c JOIN videos v USING(video_id) ORDER BY c.row_id").fetchall()
    connection.close()
    shard_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for shard_index, start in enumerate(range(0, len(rows), concurrency)):
            shard_path = shard_dir / f"video-{shard_index:06d}.npz"
            if shard_path.exists():
                continue
            batch = rows[start:start + concurrency]
            prompts = []
            media_kwargs = []
            for _, video_path, start_sec, end_sec in batch:
                data_url, metadata = clip_data_url(video_path, start_sec, end_sec)
                prompts.append([
                    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": data_url}}, {"type": "text", "text": ""}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                ])
                media_kwargs.append(metadata)
            results = await asyncio.gather(*(request_embedding(client, endpoint, model, prompt, row_id, metadata) for prompt, metadata, (row_id, _, _, _) in zip(prompts, media_kwargs, batch, strict=True)))
            np.savez(shard_path, row_ids=np.array([row_id for row_id, _, _, _ in batch]), vectors=np.array([vector for _, vector in results], dtype=np.float32))
            print(f"video_embedding_shard path={shard_path} rows={len(batch)}")


def import_shards(database_path: Path, model: str, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path))
    connection.execute("BEGIN TRANSACTION")
    for path in sorted(shard_dir.glob("text-*.npz")):
        shard = np.load(path)
        for table, row_id, vector in zip(shard["tables"], shard["row_ids"], shard["vectors"], strict=True):
            connection.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)", [str(table), str(row_id), "text", model, vector.tolist()])
    for path in sorted(shard_dir.glob("video-*.npz")):
        shard = np.load(path)
        for row_id, vector in zip(shard["row_ids"], shard["vectors"], strict=True):
            connection.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)", ["video_clips", str(row_id), "video", model, vector.tolist()])
    connection.execute("COMMIT")
    connection.execute("CHECKPOINT")
    connection.close()
    print(f"embedding_shards_imported path={database_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-concurrency", type=int, required=True)
    parser.add_argument("--video-concurrency", type=int, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(embed_text(arguments.database, arguments.endpoint, arguments.model, arguments.text_concurrency, arguments.shard_dir))
    asyncio.run(embed_video(arguments.database, arguments.endpoint, arguments.model, arguments.video_concurrency, arguments.shard_dir))
    import_shards(arguments.database, arguments.model, arguments.shard_dir)
