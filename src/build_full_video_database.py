import argparse
import os
from pathlib import Path

import duckdb
from torchcodec.decoders import VideoDecoder

from src.indexing import derive_offsets, interval_at, read_json, video_duration


def build_full_video_database(dataset_root: Path, database_path: Path, clip_dir: Path) -> None:
    month_train = read_json(dataset_root / "month" / "train.json")
    month_val = read_json(dataset_root / "month" / "val.json")
    month_offsets = derive_offsets([("month_train", month_train), ("month_val", month_val)])
    month_offsets["2"] = 42900.0

    week_questions = read_json(dataset_root / "week" / "test.json")
    week_offset_candidates: dict[str, set[float]] = {}
    for question in week_questions:
        flattened = []
        for clue in question["clue_intervals"]:
            for interval in clue["intervals"]:
                flattened.append((str(clue["video_id"]), interval))
        if len(flattened) != len(question["total_intervals"]):
            raise ValueError(f"question {question['index']} has mismatched local/global intervals")
        for (video_id, local), total in zip(flattened, question["total_intervals"], strict=True):
            local_start, local_end = local
            total_start, total_end = total
            if abs((total_end - total_start) - (local_end - local_start)) > 1e-6:
                raise ValueError(f"question {question['index']} interval duration mismatch")
            week_offset_candidates.setdefault(video_id, set()).add(round(total_start - local_start, 6))
    week_offsets = {}
    for video_id, candidates in week_offset_candidates.items():
        if len(candidates) != 1:
            raise ValueError(f"video {video_id} has inconsistent global offsets: {sorted(candidates)}")
        week_offsets[video_id] = candidates.pop()

    month_paths = sorted((dataset_root / "videos" / "month").glob("*.mp4"), key=lambda path: int(path.stem))
    if {path.stem for path in month_paths} != set(month_offsets):
        raise ValueError("month video files and question-derived offsets do not match")
    day_paths = sorted((dataset_root / "videos" / "day").glob("*.mp4"), key=lambda path: int(path.stem))
    if len(day_paths) != 1:
        raise ValueError(f"expected one day video, found {len(day_paths)}")
    week_dirs = sorted((dataset_root / "videos" / "week").glob("day*"), key=lambda path: int(path.name.removeprefix("day")))
    if {path.name for path in week_dirs} != set(week_offsets):
        raise ValueError("week video directories and question-derived offsets do not match")

    video_rows = []
    clip_rows = []
    for path in month_paths:
        duration = video_duration(path)
        offset = month_offsets[path.stem]
        video_rows.append([path.stem, len(video_rows), "month", str(path), duration, interval_at(offset, 0), interval_at(offset, duration)])
        start = 0.0
        clip_index = 0
        while start < duration:
            end = min(start + 30.0, duration)
            clip_rows.append([f"{path.stem}:video_clips:{clip_index}", path.stem, start, end, interval_at(offset, start), interval_at(offset, end)])
            start = end
            clip_index += 1

    for path in day_paths:
        video_duration(path)
        duration = 84910.8140625
        video_rows.append([path.stem, len(video_rows), "day", str(path), duration, interval_at(0, 0), interval_at(0, duration)])
        start = 0.0
        clip_index = 0
        while start < duration:
            end = min(start + 30.0, duration)
            clip_rows.append([f"{path.stem}:video_clips:{clip_index}", path.stem, start, end, interval_at(0, start), interval_at(0, end)])
            start = end
            clip_index += 1

    week_link_count = 0
    for day_dir in week_dirs:
        video_id = day_dir.name
        offset = week_offsets[video_id]
        source_clips = sorted(day_dir.glob("*.mp4"))
        if not source_clips:
            raise RuntimeError(f"no week clips found in {day_dir}")
        cached_day_dir = clip_dir / video_id
        cached_day_dir.mkdir(parents=True, exist_ok=True)
        start = 0.0
        for clip_index, source_path in enumerate(source_clips):
            print(f"week_clip_metadata video={video_id} index={clip_index} path={source_path}")
            duration = VideoDecoder(source_path, seek_mode="approximate").metadata.duration_seconds
            if duration is None or duration <= 0:
                raise RuntimeError(f"week clip has invalid duration path={source_path} duration={duration}")
            target_path = cached_day_dir / f"{clip_index:06d}.mp4"
            if target_path.exists():
                if not target_path.samefile(source_path):
                    raise RuntimeError(f"week cache target does not match source target={target_path} source={source_path}")
            else:
                os.link(source_path.resolve(), target_path)
            end = start + duration
            clip_rows.append([f"{video_id}:video_clips:{clip_index}", video_id, start, end, interval_at(offset, start), interval_at(offset, end)])
            start = end
            week_link_count += 1
        video_rows.append([video_id, len(video_rows), "week", str(day_dir), start, interval_at(offset, 0), interval_at(offset, start)])

    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise FileExistsError(database_path)
    connection = duckdb.connect(str(database_path))
    connection.execute((Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8"))
    connection.executemany("INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?)", video_rows)
    connection.executemany("INSERT INTO video_clips VALUES (?, ?, ?, ?, ?, ?)", clip_rows)
    connection.execute("CHECKPOINT")
    connection.close()
    print(f"full_video_database_written path={database_path} videos={len(video_rows)} clips={len(clip_rows)} week_links={week_link_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    arguments = parser.parse_args()
    build_full_video_database(arguments.dataset_root, arguments.database, arguments.clip_dir)
