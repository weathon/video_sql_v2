import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image
import subprocess

from src.harness import AgentRaised, VideoSQLHarness


class EmbeddingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": [1.0, 0.0]}]}


def create_database(path: Path, video_path: Path):
    connection = duckdb.connect(str(path))
    connection.execute(Path("src/schema.sql").read_text(encoding="utf-8"))
    connection.execute("INSERT INTO videos VALUES ('1', 0, 'test', ?, 4.0, INTERVAL 0 SECOND, INTERVAL 4 SECOND)", [video_path.name])
    connection.execute("INSERT INTO captions VALUES ('c1', '1', 'merged', 30, 0, 4, INTERVAL 0 SECOND, INTERVAL 4 SECOND, 'alpha memory')")
    connection.execute("INSERT INTO video_clips VALUES ('1:video_clips:0', '1', 0, 4, INTERVAL 0 SECOND, INTERVAL 4 SECOND)")
    connection.close()


def create_video(path: Path):
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=4:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-output_ts_offset", "0.5", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test video creation failed: {result.stderr}")


def test_sql_search_notes_and_video(tmp_path, monkeypatch):
    clip_dir = tmp_path / "clips"
    video_path = clip_dir / "1" / "000000.mp4"
    video_path.parent.mkdir(parents=True)
    database_path = tmp_path / "memory.duckdb"
    create_video(video_path)
    create_database(database_path, video_path)
    embedding_index_dir = tmp_path / "embedding_index"
    embedding_index_dir.mkdir()
    np.save(embedding_index_dir / "captions.vectors.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    np.save(embedding_index_dir / "captions.row_ids.npy", np.array(["c1"]))
    harness = VideoSQLHarness(database_path, "http://embedding", "test", embedding_index_dir, clip_dir, tmp_path / "media", tmp_path / "notes.json")

    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE many AS SELECT range AS value, repeat('x', 600) AS text FROM range(101)")
    connection.close()
    result = harness.run_sql("SELECT * FROM many ORDER BY value")
    assert len(result["rows"]) == 100
    assert result["hidden_more"] is True
    assert result["rows"][0][1]["original_characters"] == 600
    interval_result = harness.run_sql("SELECT global_start FROM videos")
    json.dumps(interval_result)
    assert interval_result["rows"] == [["0:00:00"]]
    assert harness.run_sql("SELECT missing FROM nowhere")["status"] == "query_error"

    monkeypatch.setattr("src.harness.httpx.post", lambda *args, **kwargs: EmbeddingResponse())
    search = harness.semantic_search("alpha", 1, "captions")
    assert search["results"][0]["row_id"] == "c1"
    assert search["results"][0]["score"] == 1.0

    harness.add_note("keep this")
    assert harness.read_notes() == {"entries": ["keep this"]}
    try:
        harness.agent_raise("broken database")
    except AgentRaised as error:
        assert str(error) == "broken database"
    else:
        raise AssertionError("agent_raise did not raise")

    sampled = harness.sample_video("1", 0, 4, 4, 0, 1000, 0, 1000, True)
    assert len(sampled["contact_sheets"]) == 2
    assert len(sampled["frame_timestamps"]) == 16
    assert sampled["captions"][0]["text"] == "alpha memory"
    for path in sampled["contact_sheets"]:
        with Image.open(path) as image:
            assert image.size == (1920, 900)
    parallel_args = [
        ("1", 0, 4, 2, 0, 1000, 0, 1000, True),
        ("1", 0, 4, 4, 200, 800, 100, 900, False),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel_samples = list(executor.map(lambda arguments: harness.sample_video(*arguments), parallel_args))
    parallel_paths = [path for result in parallel_samples for path in result["contact_sheets"]]
    assert len(parallel_paths) == len(set(parallel_paths))
    assert all(Path(path).stat().st_size > 0 for path in parallel_paths)
    screenshots = harness.evidence_screenshots("1", [0.5, 2.5])
    assert len(screenshots) == 2
    assert all(Path(path).is_file() for path in screenshots)
