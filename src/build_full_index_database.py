import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import duckdb

from src.preprocess.normalize import normalize_text, normalize_triple_atom, parse_hms, parse_time_range


def read_json(path: Path):
    print(f"read_json path={path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def interval_at(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)


def build_database(artifact_root: Path, database_path: Path) -> None:
    catalog = read_json(artifact_root / "catalog.json")
    if database_path.exists():
        raise FileExistsError(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    connection.execute((Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8"))

    for timeline in catalog["videos"]:
        video_id = timeline["video_id"]
        duration = float(timeline["duration_sec"])
        connection.execute("INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?)", [video_id, timeline["sequence_index"], timeline["source_split"], timeline["filename"], duration, interval_at(0), interval_at(duration)])
        for clip in timeline["clips"]:
            connection.execute("INSERT INTO video_clips VALUES (?, ?, ?, ?, ?, ?)", [clip["row_id"], video_id, clip["start_sec"], clip["end_sec"], interval_at(clip["start_sec"]), interval_at(clip["end_sec"])])

        video_root = artifact_root / "mmlifelong" / video_id
        for index, segment in enumerate(read_json(video_root / "whisper" / f"{video_id}_whisper.json")):
            start, end = float(segment["start_sec"]), float(segment["end_sec"])
            connection.execute("INSERT INTO asr_segments VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:asr:{index}", video_id, start, end, interval_at(start), interval_at(end), normalize_text(segment["text"])])

        for source in ("asr", "vlm"):
            for caption in read_json(video_root / "captions" / f"{video_id}_{source}_30sec.json"):
                start, end = float(caption["start_sec"]), float(caption["end_sec"])
                connection.execute("INSERT INTO captions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [caption["row_id"], video_id, source, 30, start, end, interval_at(start), interval_at(end), normalize_text(caption["text"])])
        chunk_times = {}
        for name, granularity in (("30sec", 30), ("3min", 180), ("10min", 600), ("1h", 3600)):
            captions = read_json(video_root / "merge" / "captions" / f"{video_id}_{name}.json")
            for caption in captions:
                start, end = float(caption["start_sec"]), float(caption["end_sec"])
                connection.execute("INSERT INTO captions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [caption["row_id"], video_id, "merged", granularity, start, end, interval_at(start), interval_at(end), normalize_text(caption["text"])])
                if granularity == 30:
                    chunk_times["chunk-" + hashlib.md5(caption["text"].encode()).hexdigest()] = (start, end)

        openie = read_json(video_root / "merge" / "episodic_memory" / "openie_results.json")
        entity_index = 0
        for chunk_id, entities in openie["ner_results"].items():
            for entity in entities:
                connection.execute("INSERT INTO entities VALUES (?, ?, ?, ?)", [f"{video_id}:entities:{entity_index}", video_id, chunk_id, normalize_text(entity)])
                entity_index += 1
        for entity in read_json(video_root / "merge" / "entity_whitelist.json"):
            connection.execute("INSERT INTO entities VALUES (?, ?, ?, ?)", [f"{video_id}:entities:{entity_index}", video_id, None, normalize_text(entity)])
            entity_index += 1
        triple_index = 0
        for chunk_id, triples in openie["triple_results"].items():
            start, end = chunk_times[chunk_id]
            for subject, predicate, obj in triples:
                normalized = [normalize_triple_atom(subject), normalize_triple_atom(predicate), normalize_triple_atom(obj)]
                connection.execute("INSERT INTO episodic_triples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:episodic:{triple_index}", video_id, chunk_id, start, end, interval_at(start), interval_at(end), *normalized, normalize_text(" ".join(normalized))])
                triple_index += 1

        semantic_index = 0
        semantic = read_json(video_root / "merge" / "semantic_memory" / "semantic_extraction_results.json")
        for chunk_id, result in semantic.items():
            for subject, predicate, obj in result["semantic_triples"]:
                normalized = [normalize_triple_atom(subject), normalize_triple_atom(predicate), normalize_triple_atom(obj)]
                connection.execute("INSERT INTO semantic_triples VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:semantic:{semantic_index}", video_id, chunk_id, *normalized, normalize_text(" ".join(normalized))])
                semantic_index += 1
        consolidated = read_json(video_root / "merge" / "semantic_memory" / "semantic_consolidation_results.json")
        for group in consolidated.values():
            for subject, predicate, obj in group["consolidated_semantic_triples"]:
                normalized = [normalize_triple_atom(subject), normalize_triple_atom(predicate), normalize_triple_atom(obj)]
                connection.execute("INSERT INTO semantic_triples VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:semantic:{semantic_index}", video_id, None, *normalized, normalize_text(" ".join(normalized))])
                semantic_index += 1

        for index, activity in enumerate(read_json(video_root / "merge" / "storyline_step1_activities.json")):
            start, end = parse_time_range(activity["time_range"])
            name = normalize_text(activity["name"])
            location = normalize_text(activity["location"])
            summary = normalize_text(activity["summary"])
            connection.execute("INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:activities:{index}", video_id, start, end, interval_at(start), interval_at(end), name, location, summary, normalize_text(f"{name} {location} {summary}")])

        for chain_index, chain in enumerate(read_json(video_root / "merge" / "storyline_step3_enriched.json")):
            chain_id = f"{video_id}:event_chains:{chain_index}"
            name = normalize_text(chain["name"])
            chain_type = normalize_text(chain["type"])
            connection.execute("INSERT INTO event_chains VALUES (?, ?, ?, ?, ?)", [chain_id, video_id, name, chain_type, normalize_text(f"{name} {chain_type}")])
            for step_index, step in enumerate(chain["steps"]):
                start, end = parse_hms(step["start_time"]), parse_hms(step["end_time"])
                description = normalize_text(step["description"])
                connection.execute("INSERT INTO event_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [f"{chain_id}:steps:{step_index}", chain_id, video_id, step_index, start, end, interval_at(start), interval_at(end), description, description])

        for chain_index, chain in enumerate(read_json(video_root / "merge" / "topic_chains.json").values()):
            chain_id = f"{video_id}:topic_chains:{chain_index}"
            entity = normalize_text(chain["entity"])
            connection.execute("INSERT INTO topic_chains VALUES (?, ?, ?, ?)", [chain_id, video_id, entity, bool(chain["has_lifecycle"])])
            for fact_index, fact in enumerate(chain["facts"]):
                start = parse_hms(fact["time"])
                text = normalize_text(fact["fact"])
                connection.execute("INSERT INTO topic_facts VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{chain_id}:facts:{fact_index}", chain_id, video_id, fact_index, start, interval_at(start), text])

    connection.execute("CHECKPOINT")
    connection.close()
    print(f"database_written path={database_path} videos={len(catalog['videos'])} clips={len(catalog['clips'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    arguments = parser.parse_args()
    build_database(arguments.artifact_root, arguments.database)
