"""SQL-scoped MAGIC PPR over the question's permitted graph nodes."""

import json
from types import SimpleNamespace

import duckdb
import numpy as np
import sqlglot
import torch
import torch.nn.functional as F
from sqlglot import exp

from training_free.magic import load_magic_artifacts
from training_free.index_schema import timestamp_seconds
from worldmm.memory.unified.graph import MultimodalGraph
from worldmm.memory.unified.nodes import NodeType
from worldmm.memory.unified.retrieval import cross_modal_retrieve


def search_graph(context, query, query_vector, n, pre_sql, post_sql, trace):
    for name, sql in (("pre_sql", pre_sql), ("post_sql", post_sql)):
        if sql is not None:
            statements = sqlglot.parse(sql, read="duckdb")
            if len(statements) != 1 or not isinstance(statements[0], exp.Query):
                raise ValueError(f"{name} must be exactly one read-only SELECT/WITH query")
    with duckdb.connect(str(context.database), read_only=True, config={"enable_external_access": "false", "allow_unsigned_extensions": "false"}) as connection:
        connection.execute("CREATE TEMP VIEW memory_nodes AS SELECT * FROM entity UNION ALL SELECT * FROM episode UNION ALL SELECT * FROM semantic UNION ALL SELECT * FROM visual_clip")
        permitted = {row[0] for row in connection.execute("SELECT id FROM memory_nodes").fetchall()}
        if pre_sql is None:
            selected = permitted
        else:
            cursor = connection.execute(pre_sql)
            if [column[0] for column in cursor.description] != ["id"]:
                raise ValueError("pre_sql must return exactly one column named id, containing graph node IDs")
            selected = {row[0] for row in cursor.fetchall()}
            if not selected <= permitted:
                raise ValueError("pre_sql returned IDs outside this question's permitted memory_nodes")
    trace["selected_node_ids"] = sorted(selected)
    trace["permitted_node_count"] = len(permitted)
    hits = []
    if selected:
        full_graph = load_magic_artifacts(context.embedding_index.parent)["graph"]
        if not selected <= full_graph.nodes.keys():
            raise RuntimeError("Question database contains node IDs missing from the MAGIC graph")
        graph = MultimodalGraph()
        graph.nodes = {node_id: node for node_id, node in full_graph.nodes.items() if node_id in selected}
        graph.edges = [edge for edge in full_graph.edges if edge.src_id in selected and edge.dst_id in selected]
        for node in graph.nodes.values():
            if node.text_embedding is None or (node.node_type == NodeType.VISUAL_CLIP and node.visual_embedding is None):
                raise RuntimeError(f"Missing required offline PPR embedding for {node.id}")
        graph.build_igraph(cache_embeddings=False)
        query_tensor = torch.tensor(query_vector[None, :], dtype=torch.float32)
        text_sims = np.zeros(len(graph._all_node_ids), dtype=np.float32)
        text_positions = [index for index, node_id in enumerate(graph._all_node_ids) if graph.nodes[node_id].node_type in (NodeType.EPISODE, NodeType.SEMANTIC)]
        for start in range(0, len(text_positions), 4096):
            positions = text_positions[start:start + 4096]
            matrix = torch.from_numpy(np.stack([graph.nodes[graph._all_node_ids[index]].text_embedding for index in positions]))
            text_sims[positions] = F.cosine_similarity(query_tensor, matrix, dim=1).numpy()
        vis_sims = np.empty(len(graph._vis_node_ids), dtype=np.float32)
        for start in range(0, len(graph._vis_node_ids), 4096):
            node_ids = graph._vis_node_ids[start:start + 4096]
            matrix = torch.from_numpy(np.stack([graph.nodes[node_id].visual_embedding for node_id in node_ids]))
            vis_sims[start:start + len(node_ids)] = F.cosine_similarity(query_tensor, matrix, dim=1).numpy()
        embeddings = SimpleNamespace(encode_text=lambda text: query_vector, encode_vis_query=lambda text: query_vector)
        results = cross_modal_retrieve(graph=graph, query=query, embedding_model=embeddings, top_k=n, text_sims=text_sims, vis_sims=vis_sims)
        trace["graph_stats"] = graph.stats()
        source_times = [source["timeline_start_s"] for source in context.sources]
        for rank, (node, score) in enumerate(results, 1):
            bounds = []
            for timestamp in (node.start_ts, node.end_ts):
                time = str(int(timestamp)).zfill(9)
                coarse = (int(time[0]) - 1) * 86400 + int(time[1:3]) * 3600 + int(time[3:5]) * 60 + int(time[5:7])
                source_index = int(np.searchsorted(source_times, coarse, side="right")) - 1
                if source_index < 0:
                    raise ValueError(f"Index timestamp precedes media: {timestamp}")
                bounds.append(timestamp_seconds(int(timestamp), context.sources[source_index]["fps"]))
            hits.append({"id": node.id, "node_type": node.node_type.value, "text": node.text, "start_ts": int(node.start_ts), "end_ts": int(node.end_ts), "start_s": bounds[0], "end_s": bounds[1], "metadata": node.metadata, "rank": rank, "score": float(score)})
    trace["ppr_hits"] = hits
    if post_sql is None:
        return {"candidate_count": len(selected), "ppr_hit_count": len(hits), "rows": hits}
    with duckdb.connect(config={"enable_external_access": "false", "allow_unsigned_extensions": "false"}) as connection:
        connection.execute("CREATE TABLE ppr_hits (id VARCHAR, node_type VARCHAR, text VARCHAR, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE, metadata JSON, rank INTEGER, score DOUBLE)")
        if hits:
            connection.executemany("INSERT INTO ppr_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [(hit["id"], hit["node_type"], hit["text"], hit["start_ts"], hit["end_ts"], hit["start_s"], hit["end_s"], json.dumps(hit["metadata"], ensure_ascii=False), hit["rank"], hit["score"]) for hit in hits])
        cursor = connection.execute(post_sql)
        columns = [column[0] for column in cursor.description]
        if len(columns) != len(set(columns)):
            raise ValueError("post_sql must use distinct output column names")
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {"candidate_count": len(selected), "ppr_hit_count": len(hits), "columns": columns, "rows": rows}
