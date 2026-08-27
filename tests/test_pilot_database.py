from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

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
        "question_clues": 811,
        "questions": 670,
        "semantic_triples": 123025,
        "topic_chains": 1680,
        "topic_facts": 10359,
        "video_clips": 9167,
        "videos": 14,
    }
    assert {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in expected} == expected
    assert connection.execute("SELECT count(*) FROM question_clues qc LEFT JOIN questions q USING(question_id) LEFT JOIN videos v USING(video_id) WHERE q.question_id IS NULL OR v.video_id IS NULL").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM question_clues qc JOIN videos v USING(video_id) WHERE abs(epoch(qc.global_start) - epoch(v.global_start) - qc.start_sec) > 0.000001 OR abs(epoch(qc.global_end) - epoch(v.global_start) - qc.end_sec) > 0.000001").fetchone()[0] == 0
    assert connection.execute("SELECT subject,predicate,object FROM semantic_triples WHERE video_id='21' AND object LIKE '%monkey brother%' ORDER BY chunk_id NULLS LAST").fetchall() == [("ishowspeed", "calls", "man in bright pink t‑shirt \"monkey brother\""), ("ishowspeed", "calls", "man in bright pink t‑shirt \"monkey brother\"")]
    connection.close()

    harness = VideoSQLHarness(DATABASE, "http://embedding", "qwen", Path("data/embedding_index"), Path("data/video_clips"), tmp_path / "media", tmp_path / "notes.json")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: harness.run_sql("SELECT count(*) FROM questions"), range(32)))
    assert all(result["rows"] == [[670]] for result in results)
    assert harness.run_sql("CREATE TABLE forbidden(value INTEGER)")["status"] == "query_error"
