import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import duckdb

from src.preprocess.normalize import normalize_text


def parse_hms(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS, received {value!r}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def magic_entity_key(value: str) -> str:
    stripped = str(value).strip()
    if stripped in ("I", "i", "me", "Me", "my", "My"):
        return "ishowspeed"
    return stripped.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)], check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value == "N/A":
        raise RuntimeError(f"Video stream duration unavailable for {video_path}")
    return float(value)


def build_database(artifact_root: Path, dataset_root: Path, output_path: Path) -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text()
    connection = duckdb.connect(str(output_path))
    connection.execute(schema)
    connection.execute("BEGIN TRANSACTION")
    edge_counter = 0
    excluded_visual_clip_ids = {"23:visual_clips:750"}

    def insert_graph_node(row_id: str, node_type: str, video_id: str, start_s: float, end_s: float, text: str) -> None:
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", [row_id, node_type, video_id, start_s, end_s, text])

    def insert_edge(video_id: str, src_row_id: str, dst_row_id: str, edge_type: str, weight: float) -> None:
        nonlocal edge_counter
        row_id = f"{video_id}:graph_edges:{edge_counter}"
        edge_counter += 1
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)", [row_id, video_id, src_row_id, dst_row_id, edge_type, weight])

    video_ids = sorted((path.name for path in (artifact_root / "mmlifelong").iterdir() if path.is_dir()), key=int)
    for video_id in video_ids:
        source_dir = artifact_root / "mmlifelong" / video_id / "merge"
        video_path = dataset_root / "videos" / "month" / f"{video_id}.mp4"
        connection.execute("INSERT INTO videos VALUES (?, ?, ?)", [video_id, f"videos/month/{video_id}.mp4", probe_duration(video_path)])
        episodes_by_granularity = {}
        episode_by_chunk = {}
        for granularity in ["30sec", "3min", "10min", "1h"]:
            entries = json.loads((source_dir / "captions" / f"{video_id}_{granularity}.json").read_text())
            rows = []
            for index, entry in enumerate(entries):
                row_id = f"{video_id}:episodes:{granularity}:{index}"
                start_s = float(entry["start_sec"])
                end_s = float(entry["end_sec"])
                text = normalize_text(entry["text"])
                insert_graph_node(row_id, "episode", video_id, start_s, end_s, text)
                connection.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)", [row_id, video_id, start_s, end_s, granularity, text])
                rows.append((row_id, start_s, end_s))
                if granularity == "30sec":
                    chunk_id = "chunk-" + hashlib.md5(entry["text"].encode()).hexdigest()
                    episode_by_chunk[chunk_id] = row_id
                    visual_id = f"{video_id}:visual_clips:{index}"
                    if visual_id not in excluded_visual_clip_ids:
                        insert_graph_node(visual_id, "visual_clip", video_id, start_s, end_s, text)
                        connection.execute("INSERT INTO visual_clips VALUES (?, ?, ?, ?, ?)", [visual_id, video_id, start_s, end_s, text])
                        insert_edge(video_id, visual_id, row_id, "co_clip", 1.0)
                        insert_edge(video_id, row_id, visual_id, "co_clip", 1.0)
            episodes_by_granularity[granularity] = rows
            for current, following in zip(rows, rows[1:]):
                gap = following[1] - current[2]
                if 0 <= gap <= 21600:
                    weight = 1.0 / (gap + 1.0)
                    insert_edge(video_id, current[0], following[0], "temporal_next", weight)
                    insert_edge(video_id, following[0], current[0], "temporal_next", weight)
        for coarse_granularity, fine_granularity, weight in [("3min", "30sec", 1.0), ("10min", "3min", 1.0), ("1h", "10min", 1.0), ("1h", "30sec", 0.5)]:
            for coarse in episodes_by_granularity[coarse_granularity]:
                for fine in episodes_by_granularity[fine_granularity]:
                    if coarse[1] <= fine[1] and fine[2] <= coarse[2]:
                        insert_edge(video_id, coarse[0], fine[0], "contains", weight)

        openie = json.loads((source_dir / "episodic_memory" / "openie_results.json").read_text())
        entity_state = {}
        mention_pairs = []
        for chunk_id, names in openie["ner_results"].items():
            if chunk_id not in episode_by_chunk:
                continue
            episode_id = episode_by_chunk[chunk_id]
            start_s, end_s = connection.execute("SELECT start_s, end_s FROM episodes WHERE row_id = ?", [episode_id]).fetchone()
            for name in names:
                if not str(name).strip():
                    continue
                entity_key = magic_entity_key(name)
                if entity_key not in entity_state:
                    entity_state[entity_key] = [name, start_s, end_s, 1]
                else:
                    entity_state[entity_key][2] = max(entity_state[entity_key][2], end_s)
                    entity_state[entity_key][3] += 1
                mention_pairs.append((entity_key, episode_id))
        entity_ids = {}
        for entity_key, state in entity_state.items():
            row_id = f"{video_id}:entities:{entity_key}"
            entity_ids[entity_key] = row_id
            name = normalize_text(state[0])
            insert_graph_node(row_id, "entity", video_id, state[1], state[2], name)
            connection.execute("INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, name, entity_key, state[1], state[2], state[3]])
        appeared_pairs = set()
        for entity_key, episode_id in mention_pairs:
            entity_id = entity_ids[entity_key]
            insert_edge(video_id, entity_id, episode_id, "mentioned_in", 1.0)
            visual_id = episode_id.replace(":episodes:30sec:", ":visual_clips:")
            if visual_id in excluded_visual_clip_ids:
                continue
            pair = (entity_id, visual_id)
            if pair not in appeared_pairs:
                appeared_pairs.add(pair)
                insert_edge(video_id, entity_id, visual_id, "appeared_in", 0.8)
        for chunk_id, triples in openie["triple_results"].items():
            if chunk_id not in episode_by_chunk:
                continue
            for index, triple in enumerate(triples):
                subject, predicate, object_value = [normalize_text(value) for value in triple[:3]]
                row_id = f"{video_id}:openie_triples:{chunk_id}:{index}"
                connection.execute("INSERT INTO openie_triples VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, episode_by_chunk[chunk_id], video_id, subject, predicate, object_value, normalize_text(f"{subject} {predicate} {object_value}")])

        semantic = json.loads((source_dir / "semantic_memory" / "semantic_consolidation_results.json").read_text())
        for snapshot, content in semantic.items():
            snapshot_s = float(snapshot)
            for index, triple in enumerate(content["consolidated_semantic_triples"]):
                raw_subject, raw_predicate, raw_object = triple[:3]
                subject, predicate, object_value = [normalize_text(value) for value in triple[:3]]
                row_id = f"{video_id}:semantic_triples:{snapshot}:{index}"
                text = normalize_text(f"{subject} {predicate} {object_value}")
                insert_graph_node(row_id, "semantic", video_id, snapshot_s, snapshot_s, text)
                connection.execute("INSERT INTO semantic_triples VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, snapshot_s, subject, predicate, object_value, text])
                subject_key = magic_entity_key(raw_subject)
                object_key = magic_entity_key(raw_object)
                if subject_key in entity_ids:
                    insert_edge(video_id, entity_ids[subject_key], row_id, "has_property", 1.0)
                if object_key in entity_ids:
                    insert_edge(video_id, entity_ids[object_key], row_id, "has_property", 0.5)

        topics = json.loads((source_dir / "topic_chains.json").read_text())
        for topic_index, (entity, topic) in enumerate(topics.items()):
            chain_id = f"{video_id}:topic_chains:{topic_index}"
            connection.execute("INSERT INTO topic_chains VALUES (?, ?, ?, ?)", [chain_id, video_id, normalize_text(entity), bool(topic["has_lifecycle"])])
            for fact_index, fact in enumerate(topic["facts"]):
                connection.execute("INSERT INTO topic_facts VALUES (?, ?, ?, ?, ?)", [f"{chain_id}:facts:{fact_index}", chain_id, video_id, parse_hms(fact["time"]), normalize_text(fact["fact"])])

        events = json.loads((source_dir / "storyline_step3_enriched.json").read_text())
        for chain_index, event in enumerate(events):
            chain_id = f"{video_id}:event_chains:{chain_index}"
            connection.execute("INSERT INTO event_chains VALUES (?, ?, ?, ?)", [chain_id, video_id, normalize_text(event["name"]), normalize_text(event["type"])])
            for entity_index, entity in enumerate(event["key_entities"]):
                connection.execute("INSERT INTO event_chain_entities VALUES (?, ?, ?, ?)", [f"{chain_id}:entities:{entity_index}", chain_id, video_id, normalize_text(entity)])
            for step_index, step in enumerate(event["steps"]):
                connection.execute("INSERT INTO event_steps VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{chain_id}:steps:{step_index}", chain_id, video_id, step_index, parse_hms(step["start_time"]), parse_hms(step["end_time"]), normalize_text(step["description"])])
        print(json.dumps({"video_id": video_id, "episodes": sum(len(rows) for rows in episodes_by_granularity.values()), "entities": len(entity_ids)}))
    connection.execute("COMMIT")
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_database(args.artifact_root, args.dataset_root, args.output)
