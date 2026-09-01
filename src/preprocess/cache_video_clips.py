import argparse
import json
import subprocess
from pathlib import Path

import duckdb


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(json.dumps({"command": "ffprobe", "video_path": str(video_path), "stderr": result.stderr}))
        result.check_returncode()
    value = result.stdout.strip()
    if not value or value == "N/A":
        raise RuntimeError(f"Video stream duration unavailable for {video_path}")
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    read_connection = duckdb.connect(str(args.database), read_only=True)
    videos = read_connection.execute("SELECT video_id, filename, duration_s FROM videos ORDER BY CAST(video_id AS INTEGER)").fetchall()
    visual_rows = {video_id: read_connection.execute("SELECT row_id FROM visual_clips WHERE video_id = ? ORDER BY start_s, row_id", [video_id]).fetchall() for video_id, _, _ in videos}
    read_connection.close()

    boundaries = {}
    for video_id, filename, database_duration in videos:
        source_path = args.dataset_root / filename
        source_duration = probe_duration(source_path)
        if abs(source_duration - database_duration) > 0.05:
            raise RuntimeError(f"Source duration differs from database for video {video_id}: source={source_duration}, database={database_duration}")
        output_dir = args.output_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_paths = sorted(output_dir.glob("*.mp4"))
        if not clip_paths:
            print(json.dumps({"video_id": video_id, "action": "segment", "source": str(source_path)}))
            result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source_path), "-map", "0:v:0", "-c", "copy", "-f", "segment", "-segment_time", "30", "-reset_timestamps", "1", str(output_dir / "%06d.mp4")])
            if result.returncode != 0:
                print(json.dumps({"command": "ffmpeg", "video_id": video_id, "source": str(source_path), "output_dir": str(output_dir), "returncode": result.returncode}))
                result.check_returncode()
            clip_paths = sorted(output_dir.glob("*.mp4"))
        if len(clip_paths) != len(visual_rows[video_id]):
            raise RuntimeError(f"Clip count differs from visual rows for video {video_id}: clips={len(clip_paths)}, visual_rows={len(visual_rows[video_id])}")
        durations = [probe_duration(path) for path in clip_paths]
        cache_duration = sum(durations)
        if abs(cache_duration - source_duration) > 0.05:
            raise RuntimeError(f"Cached duration differs from source for video {video_id}: cached={cache_duration}, source={source_duration}")
        current_s = 0.0
        boundaries[video_id] = []
        for duration in durations:
            boundaries[video_id].append((current_s, current_s + duration))
            current_s += duration
        print(json.dumps({"video_id": video_id, "clips": len(clip_paths), "cached_duration_s": cache_duration}))

    write_connection = duckdb.connect(str(args.database))
    write_connection.execute("BEGIN TRANSACTION")
    for video_id, _, _ in videos:
        for (row_id,), (start_s, end_s) in zip(visual_rows[video_id], boundaries[video_id]):
            write_connection.execute("UPDATE visual_clips SET start_s = ?, end_s = ? WHERE row_id = ?", [start_s, end_s, row_id])
            write_connection.execute("UPDATE graph_nodes SET start_s = ?, end_s = ? WHERE row_id = ?", [start_s, end_s, row_id])
    write_connection.execute("COMMIT")
    write_connection.close()
