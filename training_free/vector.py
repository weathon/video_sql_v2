"""SQL-scoped ordinary vector retrieval over permitted memory nodes."""

import json

import duckdb
import numpy as np
import sqlglot
from sqlglot import exp

from training_free.index_schema import timestamp_seconds


def search_vectors(context, query_vector, n, pre_sql, post_sql, trace):
    for name, sql in (("pre_sql", pre_sql), ("post_sql", post_sql)):
        if sql is not None:
            statements = sqlglot.parse(sql, read="duckdb")
            if len(statements) != 1 or not isinstance(statements[0], exp.Query):
                raise ValueError(f"{name} must be exactly one read-only SELECT/WITH query")
    tables = ("entity", "episode", "semantic", "visual_clip")
    with duckdb.connect(str(context.database), read_only=True, config={"enable_external_access": "false", "allow_unsigned_extensions": "false"}) as connection:
        connection.execute("CREATE TEMP VIEW memory_nodes AS SELECT * FROM entity UNION ALL SELECT * FROM episode UNION ALL SELECT * FROM semantic UNION ALL SELECT * FROM visual_clip")
        permitted = {row[0] for row in connection.execute("SELECT id FROM memory_nodes").fetchall()}
        if pre_sql is None:
            selected = permitted
        else:
            cursor = connection.execute(pre_sql)
            if [column[0] for column in cursor.description] != ["id"]:
                raise ValueError("pre_sql must return exactly one column named id, containing memory node IDs")
            selected = {row[0] for row in cursor.fetchall()}
            if not selected <= permitted:
                raise ValueError("pre_sql returned IDs outside this question's permitted memory_nodes")
        candidates = []
        query_norm = np.linalg.norm(query_vector)
        for table in tables:
            positions = context.eligible_positions[table]
            if pre_sql is not None:
                positions = positions[np.isin(context.row_ids[table][positions], list(selected))]
            matrix = context.vectors[table][positions]
            scores = matrix @ query_vector / (np.linalg.norm(matrix, axis=1) * query_norm)
            candidates.extend((float(score), table, int(position)) for score, position in zip(scores, positions))
        candidates.sort(key=lambda item: item[0], reverse=True)
        hits = []
        source_times = [source["timeline_start_s"] for source in context.sources]
        for rank, (score, table, position) in enumerate(candidates[:n], 1):
            row_id = str(context.row_ids[table][position])
            row = connection.execute(f"SELECT id, node_type, text, start_ts, end_ts, metadata FROM {table} WHERE id = ?", [row_id]).fetchone()
            bounds = []
            for timestamp in row[3:5]:
                value = str(int(timestamp)).zfill(9)
                coarse = (int(value[0]) - 1) * 86400 + int(value[1:3]) * 3600 + int(value[3:5]) * 60 + int(value[5:7])
                source_index = int(np.searchsorted(source_times, coarse, side="right")) - 1
                if source_index < 0:
                    raise ValueError(f"Index timestamp precedes media: {timestamp}")
                bounds.append(timestamp_seconds(int(timestamp), context.sources[source_index]["fps"]))
            hits.append({"id": row[0], "node_type": row[1], "text": row[2], "start_ts": int(row[3]), "end_ts": int(row[4]), "start_s": bounds[0], "end_s": bounds[1], "metadata": json.loads(row[5]), "rank": rank, "score": score})
    trace["selected_node_ids"] = sorted(selected)
    trace["permitted_node_count"] = len(permitted)
    trace["vector_hits"] = hits
    if post_sql is None:
        return {"candidate_count": len(selected), "vector_hit_count": len(hits), "rows": hits}
    with duckdb.connect(config={"enable_external_access": "false", "allow_unsigned_extensions": "false"}) as connection:
        connection.execute("CREATE TABLE vector_hits (id VARCHAR, node_type VARCHAR, text VARCHAR, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE, metadata JSON, rank INTEGER, score DOUBLE)")
        if hits:
            connection.executemany("INSERT INTO vector_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [(hit["id"], hit["node_type"], hit["text"], hit["start_ts"], hit["end_ts"], hit["start_s"], hit["end_s"], json.dumps(hit["metadata"], ensure_ascii=False), hit["rank"], hit["score"]) for hit in hits])
        cursor = connection.execute(post_sql)
        columns = [column[0] for column in cursor.description]
        if len(columns) != len(set(columns)):
            raise ValueError("post_sql must use distinct output column names")
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {"candidate_count": len(selected), "vector_hit_count": len(hits), "columns": columns, "rows": rows}
