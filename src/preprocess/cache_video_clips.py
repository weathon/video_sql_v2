import argparse
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
        existing = sorted(video_clip_dir.glob("*.mp4"))
        if len(existing) > len(clips):
            raise RuntimeError(f"too many cached clips video={video_id} expected={len(clips)} actual={len(existing)}")
        for clip_index, clip_path in enumerate(existing):
            row_id, start_sec, end_sec = clips[clip_index]
            if clip_path.name != f"{clip_index:06d}.mp4":
                raise RuntimeError(f"clip filename mismatch row_id={row_id} expected={clip_index:06d}.mp4 actual={clip_path.name}")
            generated_duration = VideoDecoder(clip_path, seek_mode="approximate").metadata.duration_seconds
            if generated_duration is None:
                raise RuntimeError(f"cached clip has no duration row_id={row_id} path={clip_path}")
            expected_duration = end_sec - start_sec
            if abs(generated_duration - expected_duration) > 0.05:
                raise RuntimeError(f"clip duration mismatch row_id={row_id} expected={expected_duration} actual={generated_duration}")
        start_index = len(existing)
        if start_index < len(clips):
            output_pattern = video_clip_dir / "%06d.mp4"
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
            if start_index:
                command.extend(["-ss", str(clips[start_index][1])])
            command.extend([
                "-i", video_path, "-map", "0:v:0", "-an", "-c:v", "copy",
                "-f", "segment", "-segment_time", "30", "-reset_timestamps", "1",
                "-segment_start_number", str(start_index),
                str(output_pattern),
            ])
            print(f"cache_video_clips video={video_id} start_index={start_index} remaining={len(clips) - start_index}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"cache_video_clips_failed video={video_id} error={result.stderr.strip()}")
                raise RuntimeError(f"FFmpeg clip cache failed video={video_id} returncode={result.returncode}")
        generated = sorted(video_clip_dir.glob("*.mp4"))
        if len(generated) != len(clips):
            raise RuntimeError(f"clip count mismatch video={video_id} expected={len(clips)} actual={len(generated)}")
        for clip_index in range(start_index, len(generated)):
            row_id, start_sec, end_sec = clips[clip_index]
            clip_path = generated[clip_index]
            if clip_path.name != f"{clip_index:06d}.mp4":
                raise RuntimeError(f"clip filename mismatch row_id={row_id} expected={clip_index:06d}.mp4 actual={clip_path.name}")
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
