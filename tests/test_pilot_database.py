from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from src.harness import VideoSQLHarness


DATABASE = Path("data/pilot.duckdb")


def test_real_pilot_database_counts_joins_and_time_mapping(tmp_path):
    connection = duckdb.connect(str(DATABASE), read_only=True)
    expected = {
        "activities": 785,
        "asr_segments": 103207,
        "captions": 29647,
        "entities": 54082,
        "episodic_triples": 92296,
        "event_chains": 173,
        "event_steps": 775,
        "semantic_triples": 123025,
        "topic_chains": 1680,
        "topic_facts": 10359,
        "video_clips": 9167,
        "videos": 14,
    }
    assert {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in expected} == expected
    assert connection.execute("SELECT count(*) FROM information_schema.tables WHERE table_name IN ('questions', 'question_clues')").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM information_schema.columns WHERE lower(column_name) IN ('answer', 'question', 'question_type', 'temporal_certificate')").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM videos WHERE filename LIKE '/%'").fetchone()[0] == 0
    assert connection.execute("SELECT subject,predicate,object FROM semantic_triples WHERE video_id='21' AND object LIKE '%monkey brother%' ORDER BY chunk_id NULLS LAST").fetchall() == [("ishowspeed", "calls", "man in bright pink t‑shirt \"monkey brother\""), ("ishowspeed", "calls", "man in bright pink t‑shirt \"monkey brother\"")]
    connection.close()

    harness = VideoSQLHarness(DATABASE, "http://embedding", "qwen", Path("data/embedding_index"), Path("data/video_clips"), tmp_path / "media", tmp_path / "notes.json")
    assert harness.run_sql("SELECT current_setting('enable_external_access'), current_setting('lock_configuration')")["rows"] == [[False, True]]
    with pytest.raises(duckdb.PermissionException, match="file system operations are disabled"):
        harness.run_sql("SELECT * FROM read_text('pyproject.toml')")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: harness.run_sql("SELECT count(*) FROM videos"), range(32)))
    assert all(result["rows"] == [[14]] for result in results)
    assert harness.run_sql("CREATE TABLE forbidden(value INTEGER)")["status"] == "query_error"
