import json
from pathlib import Path

import cv2
import duckdb
import numpy as np
from PIL import Image

from src.harness import AgentRaised, VideoSQLHarness


class EmbeddingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": [1.0, 0.0]}]}


def create_database(path: Path, video_path: Path):
    connection = duckdb.connect(str(path))
    connection.execute(Path("src/schema.sql").read_text(encoding="utf-8"))
    connection.execute("INSERT INTO videos VALUES ('1', 0, 'test', ?, 4.0, INTERVAL 0 SECOND, INTERVAL 4 SECOND)", [str(video_path)])
    connection.execute("INSERT INTO captions VALUES ('c1', '1', 'merged', 30, 0, 4, INTERVAL 0 SECOND, INTERVAL 4 SECOND, 'alpha memory')")
    connection.execute("INSERT INTO embeddings VALUES ('captions', 'c1', 'text', 'test', [1.0, 0.0])")
    connection.close()


def create_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    for index in range(16):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 200
        writer.write(frame)
    writer.release()


def test_sql_search_notes_and_video(tmp_path, monkeypatch):
    video_path = tmp_path / "video.avi"
    database_path = tmp_path / "memory.duckdb"
    create_video(video_path)
    create_database(database_path, video_path)
    harness = VideoSQLHarness(database_path, "http://embedding", "test", tmp_path / "media", tmp_path / "notes.json")

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
    screenshots = harness.evidence_screenshots("1", [0.5, 2.5])
    assert len(screenshots) == 2
    assert all(Path(path).is_file() for path in screenshots)
