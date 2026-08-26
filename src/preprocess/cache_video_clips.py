import argparse
import csv
from pathlib import Path
import subprocess

import duckdb
from torchcodec.decoders import VideoDecoder


def cache_video_clips(database_path: Path, clip_dir: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    videos = connection.execute("SELECT video_id, video_path FROM videos ORDER BY sequence_index").fetchall()
    for video_id, video_path in videos:
        clips = connection.execute("SELECT row_id, start_sec, end_sec FROM video_clips WHERE video_id = ? ORDER BY start_sec", [video_id]).fetchall()
        video_clip_dir = clip_dir / video_id
        video_clip_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = video_clip_dir / "segments.csv"
        if not manifest_path.exists():
            existing = list(video_clip_dir.iterdir())
            if existing:
                raise RuntimeError(f"partial clip cache video={video_id} files={len(existing)}")
            output_pattern = video_clip_dir / "%06d.mp4"
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-map", "0:v:0", "-an", "-c:v", "copy",
                "-f", "segment", "-segment_time", "30", "-reset_timestamps", "1",
                "-segment_list", str(manifest_path), "-segment_list_type", "csv",
                str(output_pattern),
            ]
            print(f"cache_video_clips video={video_id} clips={len(clips)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"cache_video_clips_failed video={video_id} error={result.stderr.strip()}")
                raise RuntimeError(f"FFmpeg clip cache failed video={video_id} returncode={result.returncode}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            generated = list(csv.reader(handle))
        if len(generated) != len(clips):
            raise RuntimeError(f"clip count mismatch video={video_id} expected={len(clips)} actual={len(generated)}")
        for clip_index, ((row_id, start_sec, end_sec), segment) in enumerate(zip(clips, generated, strict=True)):
            clip_path = video_clip_dir / f"{clip_index:06d}.mp4"
            if Path(segment[0]).name != clip_path.name:
                raise RuntimeError(f"clip filename mismatch row_id={row_id} expected={clip_path.name} actual={segment[0]}")
            if not clip_path.is_file() or clip_path.stat().st_size == 0:
                raise RuntimeError(f"missing cached clip row_id={row_id} path={clip_path}")
            generated_duration = VideoDecoder(clip_path, seek_mode="approximate").metadata.duration_seconds
            if generated_duration is None:
                raise RuntimeError(f"cached clip has no duration row_id={row_id} path={clip_path}")
            expected_duration = end_sec - start_sec
            if abs(generated_duration - expected_duration) > 0.05:
                raise RuntimeError(f"clip duration mismatch row_id={row_id} expected={expected_duration} actual={generated_duration}")
        print(f"clip_cache_verified video={video_id} clips={len(clips)}")
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    arguments = parser.parse_args()
    cache_video_clips(arguments.database, arguments.clip_dir)
