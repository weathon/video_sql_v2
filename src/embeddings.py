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


def clip_metadata(native_fps: float, start_sec: float, end_sec: float) -> dict:
    duration = end_sec - start_sec
    timestamps = np.arange(start_sec, end_sec, 0.5)
    return {"video": {"num_frames": -1, "fps": native_fps, "frames_indices": [round((timestamp - start_sec) * native_fps) for timestamp in timestamps], "total_num_frames": round(duration * native_fps), "duration": duration, "do_sample_frames": False}}


def clip_data_url(video_path: str, start_sec: float, end_sec: float) -> tuple[str, dict]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video fps video={video_path} fps={native_fps}")
    encoded_frames = []
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
    capture.release()
    data_url = "data:video/jpeg;base64," + ",".join(encoded_frames)
    return data_url, clip_metadata(native_fps, start_sec, end_sec)


async def embed_video(database_path: Path, endpoint: str, model: str, batch_size: int, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute("SELECT c.row_id, v.video_path, c.start_sec, c.end_sec FROM video_clips c JOIN videos v USING(video_id) ORDER BY v.sequence_index, c.start_sec").fetchall()
    connection.close()
    video_fps = {}
    for _, video_path, _, _ in rows:
        if video_path in video_fps:
            continue
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        video_fps[video_path] = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if video_fps[video_path] <= 0:
            raise RuntimeError(f"invalid video fps video={video_path} fps={video_fps[video_path]}")
    batches = []
    for row in rows:
        metadata = clip_metadata(video_fps[row[1]], row[2], row[3])
        if not batches or batches[-1][0] != metadata or len(batches[-1][1]) == batch_size:
            batches.append((metadata, [row]))
        else:
            batches[-1][1].append(row)
    shard_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for shard_index, (metadata, batch) in enumerate(batches):
            shard_path = shard_dir / f"video-{shard_index:06d}.npz"
            if shard_path.exists():
                continue
            conversations = []
            for _, video_path, start_sec, end_sec in batch:
                data_url, generated_metadata = clip_data_url(video_path, start_sec, end_sec)
                if generated_metadata != metadata:
                    raise RuntimeError(f"video metadata changed while encoding {video_path} {start_sec}-{end_sec}")
                conversations.append([
                    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": data_url}}, {"type": "text", "text": ""}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                ])
            results = await request_embeddings(client, endpoint, model, conversations, [row_id for row_id, _, _, _ in batch], metadata)
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
    parser.add_argument("--text-batch-size", type=int, required=True)
    parser.add_argument("--video-batch-size", type=int, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(embed_text(arguments.database, arguments.endpoint, arguments.model, arguments.text_batch_size, arguments.shard_dir))
    asyncio.run(embed_video(arguments.database, arguments.endpoint, arguments.model, arguments.video_batch_size, arguments.shard_dir))
    import_shards(arguments.database, arguments.model, arguments.shard_dir)
