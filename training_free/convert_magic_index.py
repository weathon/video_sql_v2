"""Build MAGIC-video's own graph from magic-video-artifacts and express the same graph as SQL tables + npy for our harness.

Stages (run in .venv_codex_magic with PYTHONPATH=training_free/MAGIC-video/src:training_free/MAGIC-video/src/EgoRAG):
  graph  -> runs/egolife-week-shared/magic_graph/graph.pkl (nodes carry text/visual embeddings)
  sql    -> runs/egolife-week-shared/magic_index/index.duckdb + embedding_index/*.npy
"""

import bisect
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import duckdb
import httpx
import numpy as np

from training_free.normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runs/egolife-week-shared"
MAGIC_INDEX = SHARED / "magic_index"
GRAPH_PATH = SHARED / "magic_graph/graph.pkl"
ARTIFACTS = Path("/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE")
STAGE2 = Path("/raid/wg25r/video_sql_v2/data/pilot/stage2")
WEEK = Path("/raid/wg25r/video_sql_v2/data/runtime/week")
EMBED_URL = "http://127.0.0.1:8014/v1/embeddings"
EMBED_MODEL = "qwen3-vl-embedding-8b"
PERSON = "A1_JAKE"
FPS = 20.0
GRANULARITIES = ["30sec", "3min", "10min", "1h"]
TEXT_CACHE = MAGIC_INDEX / "text_vectors"


# ---------------------------------------------------------------- time mapping
def clock_seconds(hhmmssff: str) -> float:
    hhmmssff = hhmmssff.zfill(8)
    return int(hhmmssff[0:2]) * 3600 + int(hhmmssff[2:4]) * 60 + int(hhmmssff[4:6]) + int(hhmmssff[6:8]) / FPS


class Timeline:
    def __init__(self):
        self.sources = json.loads((WEEK / "media-map.json").read_text())["sources"]
        self.by_day = {}
        self.by_name = {}
        for source in self.sources:
            name = Path(source["path"]).name
            day = source["source_video_id"].split(":")[1].upper()
            source["clock_start_s"] = clock_seconds(name[-12:-4])
            source["clock_end_s"] = source["clock_start_s"] + source["duration_s"]
            self.by_day.setdefault(day, []).append(source)
            self.by_name[name] = source
        self.starts = {day: [clip["clock_start_s"] for clip in clips] for day, clips in self.by_day.items()}

    def seconds(self, date: str, hhmmssff: str) -> float:
        clock = clock_seconds(hhmmssff)
        clips = self.by_day[date]
        index = bisect.bisect_right(self.starts[date], clock) - 1
        if index < 0:
            raise RuntimeError(f"{date} {hhmmssff} precedes the first recorded clip of that day")
        clip = clips[index]
        if clock < clip["clock_end_s"]:
            return clip["timeline_start_s"] + (clock - clip["clock_start_s"])
        return clip["timeline_end_s"]

    def ts_seconds(self, ts: int) -> float:
        text = str(ts)
        return self.seconds(f"DAY{text[0]}", text[1:].zfill(8))


# ---------------------------------------------------------------- embeddings
def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(texts, client, batch_size=32):
    """Embed normalize_text(text) for each text via 8014, cached per text on disk. Returns list of float32 arrays (shared per identical text)."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_text(text) for text in texts]
    missing = []
    seen = set()
    for item in normalized:
        key = text_key(item)
        if key in seen or (TEXT_CACHE / f"{key}.npy").exists():
            continue
        seen.add(key)
        missing.append(item)
    print(f"embedding {len(missing)} new texts of {len(set(normalized))} distinct ({len(texts)} total)", flush=True)
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        response = client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": batch})
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(batch) or any(item["index"] != i for i, item in enumerate(data)):
            raise RuntimeError(f"invalid embedding response indexes for batch starting {start}")
        for item, entry in zip(batch, data):
            vector = np.asarray(entry["embedding"], dtype=np.float32)
            if vector.shape != (4096,) or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid vector shape or values: {vector.shape}")
            np.save(TEXT_CACHE / f"{text_key(item)}.npy", vector)
        print(f"  embedded {min(start + batch_size, len(missing))}/{len(missing)}", flush=True)
    loaded = {}
    out = []
    for item in normalized:
        key = text_key(item)
        if key not in loaded:
            loaded[key] = np.load(TEXT_CACHE / f"{key}.npy")
        out.append(loaded[key])
    return out


def official_30sec_vectors(timeline):
    """Old stage2 shards hold Qwen3-VL-Embedding vectors of normalize_text(official 30sec caption) per week source file."""
    rows = {}
    for line in (STAGE2 / "run/chunks.jsonl").open():
        record = json.loads(line)
        if record["video_id"] == "week":
            rows[record["row_id"]] = record
    vectors = {}
    shards = sorted((STAGE2 / "embedding_shards/text/episodes").glob("*.npz"))
    for index, shard in enumerate(shards):
        loaded = np.load(shard, allow_pickle=False)
        row_id = str(loaded["row_ids"][0])
        if row_id not in rows:
            continue
        record = rows[row_id]
        if record["official_text"] is None:
            continue
        trace = json.loads((STAGE2 / "run/traces/embeddings/text/episodes" / f"{index:08d}" / "attempt-1.json").read_text())
        traced = trace["request"]["input"][0]
        if traced != normalize_text(record["official_text"]):
            raise RuntimeError(f"traced text differs from normalize_text(official caption) for {row_id}")
        vectors[Path(record["source_filename"]).name] = (record["official_text"], loaded["vectors"][0].astype(np.float32))
    print(f"loaded {len(vectors)} official 30sec caption vectors from old shards", flush=True)
    return vectors


def video_vectors(timeline):
    vectors = {}
    for path in sorted((MAGIC_INDEX / "visual_vectors").glob("[0-9]*.npz")):
        loaded = np.load(path, allow_pickle=False)
        vectors[str(loaded["row_ids"][0])] = loaded["vectors"][0].astype(np.float32)
    if len(vectors) != len(timeline.sources):
        raise RuntimeError(f"{len(vectors)} video vectors for {len(timeline.sources)} sources; run embed_week_videos first")
    return vectors


# ---------------------------------------------------------------- stage: graph
def rewritten_captions(timeline):
    out_dir = MAGIC_INDEX / "artifacts/captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for gran in GRANULARITIES:
        data = json.loads((ARTIFACTS / f"captions/A1_JAKE_{gran}.json").read_text())
        if gran == "30sec":
            # Two official 30sec captions are empty strings; per user decision they are dropped from the graph and the SQL tables.
            dropped = [Path(entry["video_path"]).name for entry in data if not entry["text"].strip()]
            print(f"dropping {len(dropped)} empty 30sec captions: {dropped}", flush=True)
            data = [entry for entry in data if entry["text"].strip()]
            for entry in data:
                name = Path(entry["video_path"]).name
                entry["video_path"] = timeline.by_name[name]["path"]
        target = out_dir / f"A1_JAKE_{gran}.json"
        target.write_text(json.dumps(data, ensure_ascii=False))
        files[gran] = str(target)
    return files


def build_graph():
    sys.path.insert(0, str(ROOT / "training_free/MAGIC-video/src"))
    from worldmm.memory.unified.graph import MultimodalGraph
    from worldmm.memory.unified.nodes import NodeType

    timeline = Timeline()
    caption_files = rewritten_captions(timeline)
    caption_30sec = json.loads(Path(caption_files["30sec"]).read_text())
    videos = video_vectors(timeline)
    visual_embeddings = {entry["video_path"]: videos[Path(entry["video_path"]).name] for entry in caption_30sec}

    graph = MultimodalGraph()
    graph.add_episode_nodes(caption_files)
    graph.add_visual_nodes(caption_30sec, visual_embeddings)
    graph.add_entity_nodes(str(ARTIFACTS / "episodic_memory/openie_results_openai_gpt-oss-120b.json"), caption_30sec, person=PERSON)
    graph.add_semantic_nodes(str(ARTIFACTS / "semantic_memory/semantic_consolidation_results_openai_gpt-oss-120b.json"), person=PERSON)
    print(f"graph built: {graph.stats()}", flush=True)

    official = official_30sec_vectors(timeline)
    client = httpx.Client(timeout=3600)
    other_nodes = []
    reused = 0
    for node in graph.nodes.values():
        if node.node_type == NodeType.EPISODE and node.metadata["granularity"] == "30sec":
            name = Path(node.metadata["video_path"]).name
            text, vector = official[name]
            if text != node.text:
                raise RuntimeError(f"official caption text mismatch for {name}")
            node.text_embedding = vector
            reused += 1
        elif node.node_type == NodeType.VISUAL_CLIP:
            # Same caption text as the 30sec episode; build_igraph zeroes it, but UnifiedMemory.index re-embeds every node whose text_embedding is None on each question, so fill it here.
            name = Path(node.metadata["video_path"]).name
            text, vector = official[name]
            if text != node.text:
                raise RuntimeError(f"official caption text mismatch for visual clip {name}")
            node.text_embedding = vector
        else:
            other_nodes.append(node)
    print(f"reused {reused} official 30sec vectors; embedding {len(other_nodes)} other nodes", flush=True)
    vectors = embed_texts([node.text for node in other_nodes], client)
    for node, vector in zip(other_nodes, vectors):
        node.text_embedding = vector
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(GRAPH_PATH))
    print(f"saved {GRAPH_PATH}", flush=True)


# ---------------------------------------------------------------- stage: sql
def build_sql():
    sys.path.insert(0, str(ROOT / "training_free/MAGIC-video/src"))
    from worldmm.memory.unified.nodes import NodeType

    timeline = Timeline()
    graph = pickle.loads(GRAPH_PATH.read_bytes())
    nodes = graph["nodes"]
    edges = graph["edges"]
    MAGIC_INDEX.mkdir(parents=True, exist_ok=True)
    db_path = MAGIC_INDEX / "index.duckdb"
    partial = MAGIC_INDEX / "index.partial.duckdb"
    if db_path.exists():
        raise FileExistsError(db_path)
    if partial.exists():
        partial.unlink()
    connection = duckdb.connect(str(partial))
    index_dir = MAGIC_INDEX / "embedding_index"
    index_dir.mkdir(exist_ok=True)

    def save_index(table, row_ids, vectors):
        matrix = np.stack(vectors).astype(np.float32)
        if matrix.shape != (len(row_ids), 4096):
            raise RuntimeError(f"{table} vectors shape {matrix.shape}")
        np.save(index_dir / f"{table}.row_ids.npy", np.asarray(row_ids), allow_pickle=False)
        np.save(index_dir / f"{table}.vectors.npy", matrix, allow_pickle=False)

    # episode
    rows, ids, vecs = [], [], []
    for node in nodes.values():
        if node.node_type != NodeType.EPISODE:
            continue
        meta = node.metadata
        rows.append((node.id, meta["granularity"], meta["date"], meta["start_time"], meta["end_time"], node.start_ts, node.end_ts,
                     timeline.ts_seconds(node.start_ts), timeline.ts_seconds(node.end_ts), node.text, meta["video_path"]))
        ids.append(node.id)
        vecs.append(node.text_embedding)
    connection.execute("CREATE TABLE episode(id VARCHAR PRIMARY KEY, granularity VARCHAR, date VARCHAR, start_time VARCHAR, end_time VARCHAR, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE, text VARCHAR, video_path VARCHAR)")
    connection.executemany("INSERT INTO episode VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    save_index("episode", ids, vecs)
    print(f"episode {len(rows)}", flush=True)

    # entity
    rows, ids, vecs = [], [], []
    for node in nodes.values():
        if node.node_type != NodeType.ENTITY:
            continue
        rows.append((node.id, node.text, node.metadata["normalized_name"], node.metadata["frequency"], node.start_ts, node.end_ts,
                     timeline.ts_seconds(node.start_ts), timeline.ts_seconds(node.end_ts)))
        ids.append(node.id)
        vecs.append(node.text_embedding)
    connection.execute("CREATE TABLE entity(id VARCHAR PRIMARY KEY, name VARCHAR, normalized_name VARCHAR, frequency INTEGER, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE)")
    connection.executemany("INSERT INTO entity VALUES (?,?,?,?,?,?,?,?)", rows)
    save_index("entity", ids, vecs)
    print(f"entity {len(rows)}", flush=True)

    # semantic + semantic_texts
    texts = {}
    rows = []
    snapshot_seconds = {}
    for node in nodes.values():
        if node.node_type != NodeType.SEMANTIC:
            continue
        ts = node.start_ts
        if ts not in snapshot_seconds:
            snapshot_seconds[ts] = timeline.ts_seconds(ts)
        if node.text not in texts:
            texts[node.text] = {"text_id": f"semtext_{len(texts):06d}", "subject": node.metadata["subject"], "predicate": node.metadata["predicate"],
                                "object": node.metadata["object"], "first": ts, "last": ts, "count": 0, "vector": node.text_embedding}
        entry = texts[node.text]
        entry["first"] = min(entry["first"], ts)
        entry["last"] = max(entry["last"], ts)
        entry["count"] += 1
        index_in_snapshot = int(node.id.rsplit("_", 1)[1])
        rows.append((node.id, ts, snapshot_seconds[ts], index_in_snapshot, node.metadata["subject"], node.metadata["predicate"], node.metadata["object"], node.text, entry["text_id"]))
    connection.execute("CREATE TABLE semantic(id VARCHAR PRIMARY KEY, snapshot_ts BIGINT, snapshot_s DOUBLE, index_in_snapshot INTEGER, subject VARCHAR, predicate VARCHAR, object VARCHAR, text VARCHAR, text_id VARCHAR)")
    connection.executemany("INSERT INTO semantic VALUES (?,?,?,?,?,?,?,?,?)", rows)
    print(f"semantic {len(rows)}", flush=True)
    connection.execute("CREATE TABLE semantic_texts(text_id VARCHAR PRIMARY KEY, text VARCHAR, subject VARCHAR, predicate VARCHAR, object VARCHAR, first_snapshot_ts BIGINT, last_snapshot_ts BIGINT, first_snapshot_s DOUBLE, last_snapshot_s DOUBLE, snapshot_count INTEGER)")
    connection.executemany("INSERT INTO semantic_texts VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (e["text_id"], text, e["subject"], e["predicate"], e["object"], e["first"], e["last"], snapshot_seconds[e["first"]], snapshot_seconds[e["last"]], e["count"])
        for text, e in texts.items()])
    save_index("semantic_texts", [e["text_id"] for e in texts.values()], [e["vector"] for e in texts.values()])
    print(f"semantic_texts {len(texts)}", flush=True)

    # visual_clip
    rows, ids, vecs = [], [], []
    for node in nodes.values():
        if node.node_type != NodeType.VISUAL_CLIP:
            continue
        meta = node.metadata
        rows.append((node.id, meta["date"], meta["start_time"], meta["end_time"], node.start_ts, node.end_ts,
                     timeline.ts_seconds(node.start_ts), timeline.ts_seconds(node.end_ts), meta["video_path"], node.text))
        ids.append(node.id)
        if node.visual_embedding is None:
            raise RuntimeError(f"visual node {node.id} has no visual embedding")
        vecs.append(node.visual_embedding)
    connection.execute("CREATE TABLE visual_clip(id VARCHAR PRIMARY KEY, date VARCHAR, start_time VARCHAR, end_time VARCHAR, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE, video_path VARCHAR, text VARCHAR)")
    connection.executemany("INSERT INTO visual_clip VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    save_index("visual_clip", ids, vecs)
    print(f"visual_clip {len(rows)}", flush=True)

    # graph_edges
    connection.execute("CREATE TABLE graph_edges(src_id VARCHAR, dst_id VARCHAR, edge_type VARCHAR, weight DOUBLE)")
    connection.executemany("INSERT INTO graph_edges VALUES (?,?,?,?)", [(e.src_id, e.dst_id, e.edge_type.value, e.weight) for e in edges])
    print(f"graph_edges {len(edges)}", flush=True)

    # topic chains
    topic = json.loads((ARTIFACTS / "topic_chains/topic_chains.json").read_text())
    chain_rows, fact_rows, fact_ids, fact_texts = [], [], [], []
    for entity, data in topic.items():
        facts = data["facts"]
        times = []
        for k, fact in enumerate(facts):
            # Same parse as MAGIC unified/memory.py _fact_time_to_day_secs: leading "DAYd HH:MM"; anything else has no time.
            match = re.match(r"DAY(\d+)\s+(\d{1,2}):(\d{2})", fact["time"])
            if match:
                ts = int(match.group(1) + f"{int(match.group(2)):02d}{int(match.group(3)):02d}0000")
                seconds = timeline.ts_seconds(ts)
                times.append((ts, seconds, fact["time"]))
            else:
                ts = None
                seconds = None
            fact_id = f"topicfact_{len(fact_rows):06d}"
            fact_rows.append((fact_id, entity, fact["time"], ts, seconds, fact["fact"]))
            fact_ids.append(fact_id)
            fact_texts.append(fact["fact"])
        chain_rows.append((entity, bool(data["has_lifecycle"]), len(facts), min(times)[2] if times else None, max(times)[2] if times else None,
                           min(times)[1] if times else None, max(times)[1] if times else None))
    connection.execute("CREATE TABLE topic_chains(entity VARCHAR PRIMARY KEY, has_lifecycle BOOLEAN, fact_count INTEGER, first_time VARCHAR, last_time VARCHAR, start_s DOUBLE, end_s DOUBLE)")
    connection.executemany("INSERT INTO topic_chains VALUES (?,?,?,?,?,?,?)", chain_rows)
    connection.execute("CREATE TABLE topic_facts(fact_id VARCHAR PRIMARY KEY, entity VARCHAR, time VARCHAR, time_ts BIGINT, time_s DOUBLE, fact VARCHAR)")
    connection.executemany("INSERT INTO topic_facts VALUES (?,?,?,?,?,?)", fact_rows)
    print(f"topic_chains {len(chain_rows)} topic_facts {len(fact_rows)}", flush=True)

    # event chains
    events = json.loads((ARTIFACTS / "event_chains/step3_enriched_chains.json").read_text())
    chain_rows, step_rows, step_ids, step_texts = [], [], [], []
    for c, chain in enumerate(events):
        chain_id = f"eventchain_{c:03d}"
        chain_rows.append((chain_id, chain["name"], json.dumps(chain["key_entities"], ensure_ascii=False), len(chain["steps"])))
        for k, step in enumerate(chain["steps"]):
            day_number = step["day"].replace("DAY", "")
            start_ts = int(day_number + step["start_time"].replace(":", "") + "00")
            end_ts = int(day_number + step["end_time"].replace(":", "") + "00")
            step_id = f"{chain_id}_step_{k:02d}"
            step_rows.append((step_id, chain_id, chain["name"], k, step["day"], step["start_time"], step["end_time"], start_ts, end_ts,
                              timeline.ts_seconds(start_ts), timeline.ts_seconds(end_ts), step["description"]))
            step_ids.append(step_id)
            step_texts.append(step["description"])
    connection.execute("CREATE TABLE event_chains(chain_id VARCHAR PRIMARY KEY, name VARCHAR, key_entities VARCHAR, step_count INTEGER)")
    connection.executemany("INSERT INTO event_chains VALUES (?,?,?,?)", chain_rows)
    connection.execute("CREATE TABLE event_steps(step_id VARCHAR PRIMARY KEY, chain_id VARCHAR, name VARCHAR, step_index INTEGER, day VARCHAR, start_time VARCHAR, end_time VARCHAR, start_ts BIGINT, end_ts BIGINT, start_s DOUBLE, end_s DOUBLE, description VARCHAR)")
    connection.executemany("INSERT INTO event_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", step_rows)
    print(f"event_chains {len(chain_rows)} event_steps {len(step_rows)}", flush=True)

    client = httpx.Client(timeout=3600)
    save_index("topic_facts", fact_ids, embed_texts(fact_texts, client))
    save_index("event_steps", step_ids, embed_texts(step_texts, client))
    connection.close()
    partial.rename(db_path)
    print(f"saved {db_path}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] in ("graph", "sql"):
        {"graph": build_graph, "sql": build_sql}[sys.argv[1]]()


# ---------------------------------------------------------------- stage: pretext (warm the 8014 text-vector cache before the graph stage)
def pretext():
    texts = []
    for gran in ["3min", "10min", "1h"]:
        texts.extend(entry["text"] for entry in json.loads((ARTIFACTS / f"captions/A1_JAKE_{gran}.json").read_text()))
    ner = json.loads((ARTIFACTS / "episodic_memory/openie_results_openai_gpt-oss-120b.json").read_text())["ner_results"]
    texts.extend(name.strip() for names in ner.values() for name in names if name and name.strip())
    consolidation = json.loads((ARTIFACTS / "semantic_memory/semantic_consolidation_results_openai_gpt-oss-120b.json").read_text())
    texts.extend(f"{t[0]} {t[1]} {t[2]}" for content in consolidation.values() for t in content["consolidated_semantic_triples"] if len(t) >= 3)
    topic = json.loads((ARTIFACTS / "topic_chains/topic_chains.json").read_text())
    texts.extend(fact["fact"] for data in topic.values() for fact in data["facts"])
    events = json.loads((ARTIFACTS / "event_chains/step3_enriched_chains.json").read_text())
    texts.extend(step["description"] for chain in events for step in chain["steps"])
    embed_texts(texts, httpx.Client(timeout=3600))
    print("pretext done", flush=True)


if __name__ == "__main__" and sys.argv[1] == "pretext":
    pretext()
