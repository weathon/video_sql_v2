import argparse
import asyncio
import base64
from fractions import Fraction
import json
from pathlib import Path
import subprocess

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


def probe_video_fps(video_path: str) -> float:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "json", video_path]
    print(f"ffprobe video={video_path}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffprobe_failed video={video_path} error={result.stderr.strip()}")
        raise RuntimeError(f"ffprobe failed video={video_path} returncode={result.returncode}")
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream video={video_path} streams={len(streams)}")
    native_fps = float(Fraction(streams[0]["avg_frame_rate"]))
    if native_fps <= 0:
        raise RuntimeError(f"invalid video fps video={video_path} fps={native_fps}")
    return native_fps


def clip_data_urls(video_path: str, clips: list[tuple[float, float]]) -> list[str]:
    for previous, current in zip(clips, clips[1:]):
        if previous[1] != current[0]:
            raise RuntimeError(f"FFmpeg batch is not contiguous video={video_path} previous={previous} current={current}")
    start_sec = clips[0][0]
    end_sec = clips[-1][1]
    expected_frames = [len(np.arange(start, end, 0.5)) for start, end in clips]
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid",
        "-ss", str(start_sec), "-i", video_path,
        "-vf", "tpad=stop_mode=clone:stop_duration=0.5,fps=2,hwdownload,format=nv12",
        "-frames:v", str(sum(expected_frames)), "-c:v", "mjpeg", "-q:v", "2",
        "-f", "image2pipe", "pipe:1",
    ]
    print(f"ffmpeg_cuda video={video_path} start={start_sec} end={end_sec} clips={len(clips)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoded_frames = []
    buffer = bytearray()
    while chunk := process.stdout.read(1024 * 1024):
        buffer.extend(chunk)
        while True:
            frame_start = buffer.find(b"\xff\xd8")
            if frame_start == -1:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if frame_start:
                del buffer[:frame_start]
            frame_end = buffer.find(b"\xff\xd9", 2)
            if frame_end == -1:
                break
            encoded_frames.append(base64.b64encode(buffer[:frame_end + 2]).decode())
            del buffer[:frame_end + 2]
    returncode = process.wait()
    error = process.stderr.read().decode().strip()
    if returncode != 0:
        print(f"ffmpeg_cuda_failed video={video_path} start={start_sec} end={end_sec} error={error}")
        raise RuntimeError(f"ffmpeg failed video={video_path} returncode={returncode}")
    if len(encoded_frames) != sum(expected_frames):
        raise RuntimeError(f"FFmpeg frame count mismatch video={video_path} expected={sum(expected_frames)} actual={len(encoded_frames)}")
    data_urls = []
    offset = 0
    for frame_count in expected_frames:
        data_urls.append("data:video/jpeg;base64," + ",".join(encoded_frames[offset:offset + frame_count]))
        offset += frame_count
    return data_urls


def clip_data_url(video_path: str, start_sec: float, end_sec: float) -> tuple[str, dict]:
    native_fps = probe_video_fps(video_path)
    return clip_data_urls(video_path, [(start_sec, end_sec)])[0], clip_metadata(native_fps, start_sec, end_sec)


async def embed_video(database_path: Path, endpoint: str, model: str, batch_size: int, shard_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute("SELECT c.row_id, v.video_path, c.start_sec, c.end_sec FROM video_clips c JOIN videos v USING(video_id) ORDER BY v.sequence_index, c.start_sec").fetchall()
    connection.close()
    video_fps = {}
    for _, video_path, _, _ in rows:
        if video_path in video_fps:
            continue
        video_fps[video_path] = probe_video_fps(video_path)
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
            data_urls = []
            group_start = 0
            while group_start < len(batch):
                group_end = group_start + 1
                while group_end < len(batch) and batch[group_end][1] == batch[group_start][1]:
                    group_end += 1
                data_urls.extend(clip_data_urls(batch[group_start][1], [(start_sec, end_sec) for _, _, start_sec, end_sec in batch[group_start:group_end]]))
                group_start = group_end
            conversations = []
            for data_url in data_urls:
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
