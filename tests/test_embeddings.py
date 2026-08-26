import duckdb
import httpx
import numpy as np
import subprocess

from src.embeddings import embed_video, request_embeddings
from src.preprocess.cache_video_clips import cache_video_clips


def test_ffmpeg_builds_verified_resumable_discrete_clip_cache(tmp_path):
    source_path = tmp_path / "source.mp4"
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=4:duration=34", "-c:v", "libx264", "-force_key_frames", "expr:gte(t,n_forced*30)", "-pix_fmt", "yuv420p", str(source_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test video creation failed: {result.stderr}")
    database_path = tmp_path / "cache.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE videos(video_id VARCHAR, video_path VARCHAR, sequence_index INTEGER, duration_sec DOUBLE, global_start INTERVAL)")
    connection.execute("CREATE TABLE video_clips(row_id VARCHAR, video_id VARCHAR, start_sec DOUBLE, end_sec DOUBLE, global_start INTERVAL, global_end INTERVAL)")
    connection.execute("INSERT INTO videos VALUES ('1', ?, 0, 34, INTERVAL 0 SECOND)", [str(source_path)])
    connection.execute("INSERT INTO video_clips VALUES ('1:video_clips:0', '1', 0, 30, INTERVAL 0 SECOND, INTERVAL 30 SECOND), ('1:video_clips:1', '1', 30, 34, INTERVAL 30 SECOND, INTERVAL 34 SECOND)")
    connection.close()
    clip_dir = tmp_path / "clips"
    cache_video_clips(database_path, clip_dir)
    clip_path = clip_dir / "1" / "000000.mp4"
    second_clip_path = clip_dir / "1" / "000001.mp4"
    assert clip_path.is_file()
    assert clip_path.stat().st_size > 0
    assert second_clip_path.is_file()
    second_clip_path.unlink()
    cache_video_clips(database_path, clip_dir)
    assert second_clip_path.is_file()
    assert second_clip_path.stat().st_size > 0


def test_cache_uses_actual_keyframe_aligned_boundaries(tmp_path):
    source_path = tmp_path / "variable.mp4"
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=4:duration=64", "-c:v", "libx264", "-x264-params", "scenecut=0:keyint=10000", "-force_key_frames", "35", "-pix_fmt", "yuv420p", str(source_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test video creation failed: {result.stderr}")
    database_path = tmp_path / "variable.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE videos(video_id VARCHAR, video_path VARCHAR, sequence_index INTEGER, duration_sec DOUBLE, global_start INTERVAL)")
    connection.execute("CREATE TABLE video_clips(row_id VARCHAR, video_id VARCHAR, start_sec DOUBLE, end_sec DOUBLE, global_start INTERVAL, global_end INTERVAL)")
    connection.execute("INSERT INTO videos VALUES ('1', ?, 0, 64, INTERVAL 100 SECOND)", [str(source_path)])
    connection.execute("INSERT INTO video_clips VALUES ('1:video_clips:0', '1', 0, 30, INTERVAL 100 SECOND, INTERVAL 130 SECOND), ('1:video_clips:1', '1', 30, 60, INTERVAL 130 SECOND, INTERVAL 160 SECOND), ('1:video_clips:2', '1', 60, 64, INTERVAL 160 SECOND, INTERVAL 164 SECOND)")
    connection.close()
    cache_video_clips(database_path, tmp_path / "clips")
    connection = duckdb.connect(str(database_path), read_only=True)
    boundaries = connection.execute("SELECT start_sec, end_sec, epoch(global_start), epoch(global_end) FROM video_clips ORDER BY row_id").fetchall()
    connection.close()
    assert boundaries == [(0.0, 35.0, 100.0, 135.0), (35.0, 64.0, 135.0, 164.0)]


async def test_embedding_batch_request_uses_qwen_chat_protocol():
    captured = {}

    def respond(request):
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.25, -0.25]}, {"index": 0, "embedding": [0.5, -0.5]}]})

    messages = [{"role": "user", "content": [{"type": "text", "text": "query"}]}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        results = await request_embeddings(client, "http://embedding/v1/embeddings", "qwen", [messages, messages], ["row-1", "row-2"], None)
    assert results == [("row-1", [0.5, -0.5]), ("row-2", [0.25, -0.25])]
    assert captured == {"model": "qwen", "messages": [messages, messages], "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True}
    assert "input" not in captured


async def test_video_embeddings_use_one_local_clip_per_request_with_one_second_launch_spacing(tmp_path, monkeypatch):
    database_path = tmp_path / "memory.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE videos(video_id VARCHAR, sequence_index INTEGER)")
    connection.execute("CREATE TABLE video_clips(row_id VARCHAR, video_id VARCHAR, start_sec DOUBLE)")
    connection.execute("INSERT INTO videos VALUES ('1', 0)")
    connection.execute("INSERT INTO video_clips VALUES ('1:video_clips:0', '1', 0), ('1:video_clips:1', '1', 30)")
    connection.close()
    clip_dir = tmp_path / "clips" / "1"
    clip_dir.mkdir(parents=True)
    (clip_dir / "000000.mp4").write_bytes(b"first")
    (clip_dir / "000001.mp4").write_bytes(b"second")
    requests = []
    sleeps = []

    async def fake_request(client, endpoint, model, conversations, row_ids, media_io_kwargs):
        requests.append((conversations, row_ids, media_io_kwargs))
        return [(row_ids[0], [float(len(requests)), 0.0])]

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("src.embeddings.request_embeddings", fake_request)
    monkeypatch.setattr("src.embeddings.asyncio.sleep", fake_sleep)
    shard_dir = tmp_path / "shards"
    await embed_video(database_path, "http://embedding", "qwen", 64, shard_dir, tmp_path / "clips")
    assert len(requests) == 2
    assert [request[1] for request in requests] == [["1:video_clips:0"], ["1:video_clips:1"]]
    assert requests[0][0][0][1]["content"][0]["video_url"]["url"] == (clip_dir / "000000.mp4").resolve().as_uri()
    assert requests[1][0][0][1]["content"][0]["video_url"]["url"] == (clip_dir / "000001.mp4").resolve().as_uri()
    assert len(sleeps) == 1
    assert 0.9 < sleeps[0] <= 1.0
    assert np.load(shard_dir / "video-000000.npz")["row_ids"].tolist() == ["1:video_clips:0"]
    assert np.load(shard_dir / "video-000001.npz")["row_ids"].tolist() == ["1:video_clips:1"]
