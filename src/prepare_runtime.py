import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


SEARCH_TABLES = (
    "episodes",
    "entities",
    "semantic_triples",
    "topic_facts",
    "event_steps",
    "object_states",
    "visual_clips",
)


TABLE_SQL = {
    "episodes": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               start_s, end_s, granularity, text
        FROM source_db.episodes WHERE video_id = '{timeline}'
    """,
    "entities": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               name, normalized_name, first_seen_s, last_seen_s, frequency
        FROM source_db.entities WHERE video_id = '{timeline}'
    """,
    "semantic_triples": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               regexp_replace(caption_30sec_row_id, '^{prefix}', '') AS caption_30sec_row_id,
               snapshot_s, subject, predicate, object, text
        FROM source_db.semantic_triples WHERE video_id = '{timeline}'
    """,
    "topic_chains": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               entity, has_lifecycle, level,
               regexp_replace(scope_id, '^{prefix}', '') AS scope_id,
               start_s, end_s, summary
        FROM source_db.topic_chains WHERE video_id = '{timeline}'
    """,
    "topic_facts": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               regexp_replace(chain_row_id, '^{prefix}', '') AS chain_row_id,
               regexp_replace(caption_30sec_row_id, '^{prefix}', '') AS caption_30sec_row_id,
               time_s, fact
        FROM source_db.topic_facts WHERE video_id = '{timeline}'
    """,
    "event_chains": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               name, chain_type, level,
               regexp_replace(scope_id, '^{prefix}', '') AS scope_id,
               start_s, end_s, summary
        FROM source_db.event_chains WHERE video_id = '{timeline}'
    """,
    "event_steps": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               regexp_replace(chain_row_id, '^{prefix}', '') AS chain_row_id,
               regexp_replace(caption_start_row_id, '^{prefix}', '') AS caption_start_row_id,
               regexp_replace(caption_end_row_id, '^{prefix}', '') AS caption_end_row_id,
               step_index, start_s, end_s, description
        FROM source_db.event_steps WHERE video_id = '{timeline}'
    """,
    "visual_clips": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               start_s, end_s, text
        FROM source_db.visual_clips WHERE video_id = '{timeline}'
    """,
    "object_states": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               regexp_replace(episode_3min_row_id, '^{prefix}', '') AS episode_3min_row_id,
               regexp_replace(caption_30sec_row_id, '^{prefix}', '') AS caption_30sec_row_id,
               start_s, end_s, object, status, text
        FROM source_db.object_states WHERE video_id = '{timeline}'
    """,
    "graph_edges": """
        SELECT regexp_replace(row_id, '^{prefix}', '') AS row_id,
               regexp_replace(src_row_id, '^{prefix}', '') AS src_row_id,
               regexp_replace(dst_row_id, '^{prefix}', '') AS dst_row_id,
               edge_type, weight
        FROM source_db.graph_edges WHERE video_id = '{timeline}'
    """,
}


def build_database(source_db: Path, output_db: Path, timeline: str) -> None:
    if output_db.exists():
        raise FileExistsError(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(output_db))
    connection.execute(f"ATTACH '{source_db}' AS source_db (READ_ONLY)")
    prefix = timeline + ":"
    for table, query in TABLE_SQL.items():
        connection.execute(
            f"CREATE TABLE {table} AS " + query.format(prefix=prefix, timeline=timeline)
        )
    connection.execute(
        """
        CREATE TABLE topic_fact_evidence AS
        SELECT regexp_replace(e.row_id, ?, '') AS row_id,
               regexp_replace(e.fact_row_id, ?, '') AS fact_row_id,
               regexp_replace(e.caption_30sec_row_id, ?, '') AS caption_30sec_row_id
        FROM source_db.topic_fact_evidence e
        JOIN source_db.topic_facts f ON f.row_id = e.fact_row_id
        WHERE f.video_id = ?
        """,
        [prefix, prefix, prefix, timeline],
    )
    connection.execute(
        """
        CREATE TABLE event_step_evidence AS
        SELECT regexp_replace(e.row_id, ?, '') AS row_id,
               regexp_replace(e.step_row_id, ?, '') AS step_row_id,
               regexp_replace(e.caption_30sec_row_id, ?, '') AS caption_30sec_row_id
        FROM source_db.event_step_evidence e
        JOIN source_db.event_steps s ON s.row_id = e.step_row_id
        WHERE s.video_id = ?
        """,
        [prefix, prefix, prefix, timeline],
    )
    connection.close()


def build_embeddings(
    source_db: Path, source_index: Path, output_root: Path, timeline: str
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(source_db), read_only=True)
    prefix = timeline + ":"
    for table in SEARCH_TABLES:
        row_ids = np.load(source_index / f"{table}.row_ids.npy", mmap_mode="r", allow_pickle=False)
        vectors = np.load(source_index / f"{table}.vectors.npy", mmap_mode="r", allow_pickle=False)
        selected = np.fromiter(
            (str(row_id).startswith(prefix) for row_id in row_ids),
            dtype=np.bool_,
            count=len(row_ids),
        )
        public_ids = np.asarray(
            [str(row_id).removeprefix(prefix) for row_id in row_ids[selected]]
        )
        expected = connection.execute(
            f"SELECT count(*) FROM {table} WHERE video_id = ?", [timeline]
        ).fetchone()[0]
        assert len(public_ids) == expected
        np.save(output_root / f"{table}.row_ids.npy", public_ids, allow_pickle=False)
        np.save(output_root / f"{table}.vectors.npy", vectors[selected], allow_pickle=False)
    connection.close()


def write_media_map(source_db: Path, dataset_root: Path, output: Path, timeline: str) -> None:
    connection = duckdb.connect(str(source_db), read_only=True)
    rows = connection.execute(
        """
        SELECT source_video_id, filename, duration_s, timeline_start_s, timeline_end_s
        FROM source_videos WHERE video_id = ? ORDER BY timeline_start_s
        """,
        [timeline],
    ).fetchall()
    sources = []
    for source_video_id, filename, duration_s, start_s, end_s in rows:
        path = dataset_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(
            {
                "source_video_id": source_video_id,
                "path": str(path),
                "duration_s": duration_s,
                "timeline_start_s": start_s,
                "timeline_end_s": end_s,
            }
        )
    output.write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for timeline in ("day", "week", "month"):
        root = args.output_root / timeline
        build_database(args.source_db, root / "index.duckdb", timeline)
        build_embeddings(args.source_db, args.source_index, root / "embedding_index", timeline)
        write_media_map(args.source_db, args.dataset_root, root / "media-map.json", timeline)
        print(f"prepared {timeline}", flush=True)


if __name__ == "__main__":
    main()
