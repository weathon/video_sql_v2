"""EgoLifeQA over the MM-Lifelong week timeline: verification report, open-ended questions, cutoff media maps and cutoff databases."""

import bisect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runs/egolife-week-shared"
EGOLIFE = Path("/raid/huggingface/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0")
WEEK = Path("/raid/wg25r/video_sql_v2/data/runtime/week")
FPS = 20.0
FRAME_SIZE = "854:480"


def clock_seconds(hhmmssff: str) -> float:
    return int(hhmmssff[0:2]) * 3600 + int(hhmmssff[2:4]) * 60 + int(hhmmssff[4:6]) + int(hhmmssff[6:8]) / FPS


def clock_text(hhmmssff: str) -> str:
    return f"{hhmmssff[0:2]}:{hhmmssff[2:4]}:{hhmmssff[4:6]}"


def cutoff_key(cutoff_s: float) -> str:
    return f"{cutoff_s:.3f}"


def load_sources():
    sources = json.loads((WEEK / "media-map.json").read_text())["sources"]
    by_day = {}
    for source in sources:
        name = Path(source["path"]).name
        day = source["source_video_id"].split(":")[1].upper()
        if name != f"{day}_A1_JAKE_{name[-12:-4]}.mp4":
            raise RuntimeError(f"unexpected source filename {name}")
        source["day"] = day
        source["clock_start_s"] = clock_seconds(name[-12:-4])
        source["clock_end_s"] = source["clock_start_s"] + source["duration_s"]
        by_day.setdefault(day, []).append(source)
    for day, clips in by_day.items():
        if clips != sorted(clips, key=lambda s: s["clock_start_s"]):
            raise RuntimeError(f"{day} sources are not in clock order")
        if clips != sorted(clips, key=lambda s: s["timeline_start_s"]):
            raise RuntimeError(f"{day} sources are not in timeline order")
    return sources, by_day


def map_cutoff(by_day, day, hhmmssff):
    clips = by_day[day]
    clock = clock_seconds(hhmmssff)
    starts = [clip["clock_start_s"] for clip in clips]
    index = bisect.bisect_right(starts, clock) - 1
    if index < 0:
        raise RuntimeError(f"{day} {hhmmssff} precedes the first recorded clip")
    clip = clips[index]
    if clock < clip["clock_end_s"]:
        return clip["timeline_start_s"] + (clock - clip["clock_start_s"]), "inside_clip", clip
    return clip["timeline_end_s"], "recording_gap_after_clip", clip


def verify(sources, by_day):
    out = SHARED / "verification"
    out.mkdir(parents=True, exist_ok=True)
    durations = []
    mismatches = []
    for source in sources:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", source["path"]],
            capture_output=True, text=True, check=True,
        )
        actual = float(json.loads(probe.stdout)["format"]["duration"])
        durations.append({"path": source["path"], "media_map_duration_s": source["duration_s"], "ffprobe_duration_s": actual})
        if abs(actual - source["duration_s"]) > 0.05:
            mismatches.append(durations[-1])
    gaps = []
    for day, clips in by_day.items():
        for previous, current in zip(clips, clips[1:]):
            clock_gap = current["clock_start_s"] - previous["clock_end_s"]
            timeline_gap = current["timeline_start_s"] - previous["timeline_end_s"]
            if abs(clock_gap) > 1.0 or abs(timeline_gap) > 1e-6:
                gaps.append({"day": day, "after": Path(previous["path"]).name, "before": Path(current["path"]).name,
                             "clock_gap_s": clock_gap, "timeline_gap_s": timeline_gap})
    frames = []
    frame_dir = out / "frames"
    frame_dir.mkdir(exist_ok=True)
    for day, clips in by_day.items():
        picks = [(clips[0], 1.0), (clips[len(clips) // 2], clips[len(clips) // 2]["duration_s"] / 2), (clips[-1], max(clips[-1]["duration_s"] - 1.0, 0.0))]
        for clip, offset in picks:
            name = Path(clip["path"]).stem
            target = frame_dir / f"{name}_at_{offset:.1f}s.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-i", clip["path"], "-frames:v", "1", "-vf", f"scale={FRAME_SIZE}", str(target)],
                check=True,
            )
            frames.append({"day": day, "source": clip["path"], "offset_s": offset,
                           "expected_clock": clock_text(name[-8:]), "timeline_s": clip["timeline_start_s"] + offset, "image": str(target)})
    report = {
        "media_map": str(WEEK / "media-map.json"),
        "source_count": len(sources),
        "timeline_end_s": sources[-1]["timeline_end_s"],
        "day_starts": {day: {"timeline_start_s": clips[0]["timeline_start_s"], "first_clip": Path(clips[0]["path"]).name,
                             "last_clip": Path(clips[-1]["path"]).name, "clip_count": len(clips)} for day, clips in by_day.items()},
        "mapping_rule": "cutoff_s = clip.timeline_start_s + (query clock - clip filename clock) for the clip whose clock range covers the query; queries inside a recording gap use the previous clip's timeline_end_s; frame field FF converted at 20 fps",
        "duration_mismatches": mismatches,
        "gaps": gaps,
        "frame_samples": frames,
        "durations": durations,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"verification: {len(sources)} sources, {len(mismatches)} duration mismatches, {len(gaps)} gaps, {len(frames)} frames", flush=True)


def build_questions(by_day):
    raw_dir = SHARED / "egolife_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = EGOLIFE / "EgoLifeQA/EgoLifeQA_A1_JAKE.json"
    shutil.copy(raw_path, raw_dir / raw_path.name)
    questions = []
    for row in json.loads(raw_path.read_text()):
        day = row["query_time"]["date"]
        time = row["query_time"]["time"]
        if len(time) != 8:
            raise RuntimeError(f"question {row['ID']} has query time {time!r}")
        cutoff_s, mapping, clip = map_cutoff(by_day, day, time)
        answer_text = row[f"choice_{row['answer'].lower()}"]
        questions.append({
            "ID": row["ID"], "type": row["type"], "original_question": row["question"],
            "choices": {letter: row[f"choice_{letter.lower()}"] for letter in "ABCD"},
            "answer_letter": row["answer"], "query_time": row["query_time"], "target_time": row["target_time"],
            "question": f"Asked on {day} at {clock_text(time)} (timeline {cutoff_s:.1f} s). {row['question']}",
            "answer": answer_text,
            "cutoff_s": cutoff_s, "cutoff_key": cutoff_key(cutoff_s), "cutoff_mapping": mapping,
            "cutoff_clip": Path(clip["path"]).name,
            "until_time": int(day.replace("DAY", "") + time),
        })
    (SHARED / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
    print(f"questions: {len(questions)}, distinct cutoffs: {len({q['cutoff_key'] for q in questions})}", flush=True)
    return questions


def write_media_maps(sources, questions):
    for key in sorted({q["cutoff_key"] for q in questions}):
        cutoff_s = float(key)
        kept = []
        for source in sources:
            if source["timeline_start_s"] >= cutoff_s:
                break
            trimmed = {k: source[k] for k in ("source_video_id", "path", "duration_s", "timeline_start_s", "timeline_end_s")}
            if trimmed["timeline_end_s"] > cutoff_s:
                trimmed["timeline_end_s"] = cutoff_s
                trimmed["duration_s"] = cutoff_s - trimmed["timeline_start_s"]
            kept.append(trimmed)
        target = SHARED / "cutoffs" / key / "media-map.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"sources": kept}, indent=2) + "\n")


# MAGIC-graph-as-SQL schema (see baselines_plan.md section 5). Interval tables keep rows starting at or before the cutoff.
CUTOFF_SQL = {
    "episode": ("SELECT * FROM src.episode WHERE start_s <= ?", 1),
    "entity": ("SELECT id, name, normalized_name, frequency, start_ts, end_ts, start_s, least(end_s, ?) AS end_s FROM src.entity WHERE start_s <= ?", 2),
    "semantic": ("SELECT * FROM src.semantic WHERE snapshot_s <= ?", 1),
    "semantic_texts": ("SELECT * FROM src.semantic_texts WHERE first_snapshot_s <= ?", 1),
    "visual_clip": ("SELECT * FROM src.visual_clip WHERE start_s <= ?", 1),
    "topic_facts": ("SELECT * FROM src.topic_facts WHERE time_s <= ?", 1),
    "event_steps": ("SELECT * FROM src.event_steps WHERE start_s <= ?", 1),
}


def write_cutoff_databases(source_db: Path, output_root: Path, questions):
    for key in sorted({q["cutoff_key"] for q in questions}):
        cutoff_s = float(key)
        target = output_root / key / "index.duckdb"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.duckdb")
        if partial.exists():
            partial.unlink()
        connection = duckdb.connect(str(partial))
        connection.execute(f"ATTACH '{source_db}' AS src (READ_ONLY)")
        for table, (sql, arity) in CUTOFF_SQL.items():
            connection.execute(f"CREATE TABLE {table} AS {sql}", [cutoff_s] * arity)
        connection.execute("CREATE TABLE topic_chains AS SELECT * FROM src.topic_chains WHERE entity IN (SELECT entity FROM topic_facts)")
        connection.execute("CREATE TABLE event_chains AS SELECT * FROM src.event_chains WHERE chain_id IN (SELECT chain_id FROM event_steps)")
        connection.execute(
            "CREATE TABLE graph_edges AS SELECT e.* FROM src.graph_edges e WHERE "
            "e.src_id IN (SELECT id FROM episode UNION ALL SELECT id FROM entity UNION ALL SELECT id FROM semantic UNION ALL SELECT id FROM visual_clip) AND "
            "e.dst_id IN (SELECT id FROM episode UNION ALL SELECT id FROM entity UNION ALL SELECT id FROM semantic UNION ALL SELECT id FROM visual_clip)"
        )
        connection.close()
        partial.rename(target)
        print(f"cutoff db {key}", flush=True)


def main():
    stage = sys.argv[1]
    sources, by_day = load_sources()
    if stage == "verify":
        verify(sources, by_day)
    elif stage == "questions":
        questions = build_questions(by_day)
        write_media_maps(sources, questions)
    elif stage == "cutoff-dbs":
        questions = json.loads((SHARED / "questions.json").read_text())
        write_cutoff_databases(Path(sys.argv[2]), Path(sys.argv[3]), questions)
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
