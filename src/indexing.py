import argparse
import hashlib
import json
import subprocess
from datetime import timedelta
from pathlib import Path

import duckdb

from src.preprocess.normalize import normalize_text, normalize_triple_atom, parse_hms, parse_time_range


def read_json(path: Path):
    print(f"read_json path={path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def video_duration(path: Path) -> float:
    print(f"video_metadata path={path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ffprobe_failed video={path} error={result.stderr.strip()}")
        raise RuntimeError(f"ffprobe failed video={path} returncode={result.returncode}")
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"invalid video duration path={path} duration={duration}")
    return duration


def derive_offsets(question_sets: list[tuple[str, list[dict]]]) -> dict[str, float]:
    offsets: dict[str, set[float]] = {}
    for _, questions in question_sets:
        for question in questions:
            flattened = []
            for clue in question["clue_interval"]:
                for interval in clue["intervals"]:
                    flattened.append((str(clue["video_id"]), interval))
            if len(flattened) != len(question["total_intervals"]):
                raise ValueError(f"question {question['index']} has mismatched local/global intervals")
            for (video_id, local), total in zip(flattened, question["total_intervals"], strict=True):
                local_start, local_end = local
                total_start, total_end = total
                if abs((total_end - total_start) - (local_end - local_start)) > 1e-6:
                    raise ValueError(f"question {question['index']} interval duration mismatch")
                offsets.setdefault(video_id, set()).add(round(total_start - local_start, 6))
    for video_id, candidates in offsets.items():
        if len(candidates) != 1:
            raise ValueError(f"video {video_id} has inconsistent global offsets: {sorted(candidates)}")
    return {video_id: candidates.pop() for video_id, candidates in offsets.items()}


def interval_at(offset: float, seconds: float) -> timedelta:
    return timedelta(seconds=offset + seconds)


def build_database(pilot_root: Path, database_path: Path) -> None:
    train = read_json(pilot_root / "month" / "train.json")
    val = read_json(pilot_root / "month" / "val.json")
    question_sets = [("month_train", train), ("month_val", val)]
    offsets = derive_offsets(question_sets)
    video_paths = sorted((pilot_root / "videos" / "month").glob("*.mp4"), key=lambda path: int(path.stem))
    if {path.stem for path in video_paths} != set(offsets):
        raise ValueError("video files and question-derived offsets do not match")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise FileExistsError(database_path)
    connection = duckdb.connect(str(database_path))
    connection.execute((Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8"))

    for sequence_index, path in enumerate(sorted(video_paths, key=lambda item: offsets[item.stem])):
        duration = video_duration(path)
        offset = offsets[path.stem]
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?)",
            [path.stem, sequence_index, "month", str(path), duration, interval_at(offset, 0), interval_at(offset, duration)],
        )
        clip_index = 0
        start = 0.0
        while start < duration:
            end = min(start + 30.0, duration)
            connection.execute(
                "INSERT INTO video_clips VALUES (?, ?, ?, ?, ?, ?)",
                [f"{path.stem}:video_clips:{clip_index}", path.stem, start, end, interval_at(offset, start), interval_at(offset, end)],
            )
            clip_index += 1
            start = end

    for split, questions in question_sets:
        for question in questions:
            question_id = int(question["index"])
            connection.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?)",
                [question_id, split, normalize_text(question["question"]), normalize_text(question["answer"]), normalize_text(question["question_type"]), normalize_text(question["temporal_certificate"])],
            )
            for clue_order, clue in enumerate(question["clue_interval"]):
                video_id = str(clue["video_id"])
                for interval_order, (start, end) in enumerate(clue["intervals"]):
                    connection.execute(
                        "INSERT INTO question_clues VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [question_id, video_id, clue_order, interval_order, start, end, interval_at(offsets[video_id], start), interval_at(offsets[video_id], end)],
                    )

    artifact_root = pilot_root / "artifacts" / "mmlifelong"
    caption_granularities = {"30sec": 30, "3min": 180, "10min": 600, "1h": 3600}
    for video_id in sorted(offsets, key=lambda item: offsets[item]):
        offset = offsets[video_id]
        video_root = artifact_root / video_id
        captions_30 = read_json(video_root / "merge" / "captions" / f"{video_id}_30sec.json")
        chunk_times = {}
        for caption in captions_30:
            chunk_id = "chunk-" + hashlib.md5(caption["text"].encode()).hexdigest()
            chunk_times[chunk_id] = (float(caption["start_sec"]), float(caption["end_sec"]))

        for index, segment in enumerate(read_json(video_root / "whisper" / f"{video_id}_whisper.json")):
            start, end = float(segment["start_sec"]), float(segment["end_sec"])
            connection.execute(
                "INSERT INTO asr_segments VALUES (?, ?, ?, ?, ?, ?, ?)",
                [f"{video_id}:asr:{index}", video_id, start, end, interval_at(offset, start), interval_at(offset, end), normalize_text(segment["text"])],
            )

        for source in ("asr", "vlm"):
            path = video_root / "captions" / f"{video_id}_{source}_30sec.json"
            for index, caption in enumerate(read_json(path)):
                start, end = float(caption["start_sec"]), float(caption["end_sec"])
                connection.execute(
                    "INSERT INTO captions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [f"{video_id}:captions:{source}:30:{index}", video_id, source, 30, start, end, interval_at(offset, start), interval_at(offset, end), normalize_text(caption["text"])],
                )
        for name, granularity in caption_granularities.items():
            path = video_root / "merge" / "captions" / f"{video_id}_{name}.json"
            for index, caption in enumerate(read_json(path)):
                start, end = float(caption["start_sec"]), float(caption["end_sec"])
                connection.execute(
                    "INSERT INTO captions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [f"{video_id}:captions:merged:{granularity}:{index}", video_id, "merged", granularity, start, end, interval_at(offset, start), interval_at(offset, end), normalize_text(caption["text"])],
                )

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
                connection.execute(
                    "INSERT INTO episodic_triples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [f"{video_id}:episodic:{triple_index}", video_id, chunk_id, start, end, interval_at(offset, start), interval_at(offset, end), *normalized, normalize_text(" ".join(normalized))],
                )
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
            for triple in group["consolidated_semantic_triples"]:
                if triple == ["Man", "checks", "menu when dining", ""]:
                    triple = ["Man", "checks menu", "when dining"]
                elif triple == ["IShowSpeed", "rides", "attractions with", "friend"]:
                    triple = ["IShowSpeed", "rides attractions with", "friend"]
                elif triple == ["IShowSpeed", "calls", "Man in bright pink t‑shirt", "monkey brother"]:
                    triple = ["IShowSpeed", "calls", "Man in bright pink t‑shirt \"monkey brother\""]
                elif triple == ["the streamer", "checks", "phone while gaming", "true"]:
                    triple = ["the streamer", "checks phone while gaming", "true"]
                if len(triple) != 3:
                    raise ValueError(f"invalid consolidated semantic triple: {triple!r}")
                subject, predicate, obj = triple
                normalized = [normalize_triple_atom(subject), normalize_triple_atom(predicate), normalize_triple_atom(obj)]
                connection.execute("INSERT INTO semantic_triples VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:semantic:{semantic_index}", video_id, None, *normalized, normalize_text(" ".join(normalized))])
                semantic_index += 1

        for index, activity in enumerate(read_json(video_root / "merge" / "storyline_step1_activities.json")):
            start, end = parse_time_range(activity["time_range"])
            name = normalize_text(activity["name"])
            location = normalize_text(activity["location"])
            summary = normalize_text(activity["summary"])
            connection.execute("INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [f"{video_id}:activities:{index}", video_id, start, end, interval_at(offset, start), interval_at(offset, end), name, location, summary, normalize_text(f"{name} {location} {summary}")])

        for chain_index, chain in enumerate(read_json(video_root / "merge" / "storyline_step3_enriched.json")):
            chain_id = f"{video_id}:event_chains:{chain_index}"
            name = normalize_text(chain["name"])
            chain_type = normalize_text(chain["type"])
            connection.execute("INSERT INTO event_chains VALUES (?, ?, ?, ?, ?)", [chain_id, video_id, name, chain_type, normalize_text(f"{name} {chain_type}")])
            for step_index, step in enumerate(chain["steps"]):
                start, end = parse_hms(step["start_time"]), parse_hms(step["end_time"])
                description = normalize_text(step["description"])
                connection.execute("INSERT INTO event_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [f"{chain_id}:steps:{step_index}", chain_id, video_id, step_index, start, end, interval_at(offset, start), interval_at(offset, end), description, description])

        for chain_index, chain in enumerate(read_json(video_root / "merge" / "topic_chains.json").values()):
            chain_id = f"{video_id}:topic_chains:{chain_index}"
            entity = normalize_text(chain["entity"])
            connection.execute("INSERT INTO topic_chains VALUES (?, ?, ?, ?)", [chain_id, video_id, entity, bool(chain["has_lifecycle"])])
            for fact_index, fact in enumerate(chain["facts"]):
                start = parse_hms(fact["time"])
                text = normalize_text(fact["fact"])
                connection.execute("INSERT INTO topic_facts VALUES (?, ?, ?, ?, ?, ?, ?)", [f"{chain_id}:facts:{fact_index}", chain_id, video_id, fact_index, start, interval_at(offset, start), text])

    connection.execute("CHECKPOINT")
    connection.close()
    print(f"database_written path={database_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    arguments = parser.parse_args()
    build_database(arguments.pilot_root, arguments.database)
