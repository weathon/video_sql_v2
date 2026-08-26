from pathlib import Path

import duckdb


def test_schema_tables(tmp_path):
    database = tmp_path / "schema.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(Path("src/schema.sql").read_text(encoding="utf-8"))
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    connection.close()
    assert tables == {
        "activities", "asr_segments", "captions", "embeddings", "entities",
        "episodic_triples", "event_chains", "event_steps", "question_clues",
        "questions", "semantic_triples", "topic_chains", "topic_facts",
        "video_clips", "videos",
    }
