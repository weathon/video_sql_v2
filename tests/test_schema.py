from pathlib import Path

import duckdb


def test_schema_contains_magic_nodes_edges_and_chains():
    connection = duckdb.connect(":memory:")
    connection.execute((Path("src/schema.sql")).read_text())
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert tables == {"entities", "episodes", "event_chain_entities", "event_chains", "event_steps", "graph_edges", "graph_nodes", "openie_triples", "semantic_triples", "topic_chains", "topic_facts", "videos", "visual_clips"}
