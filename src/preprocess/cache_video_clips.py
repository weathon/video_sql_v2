import argparse
from datetime import timedelta
from pathlib import Path
import subprocess

import duckdb
from torchcodec.decoders import VideoDecoder


def cache_video_clips(database_path: Path, clip_dir: Path) -> None:
    connection = duckdb.connect(str(database_path))
    videos = connection.execute("SELECT video_id, video_path, duration_sec, global_start FROM videos ORDER BY sequence_index").fetchall()
    for video_id, video_path, video_duration, video_global_start in videos:
        clips = connection.execute("SELECT row_id, start_sec, end_sec FROM video_clips WHERE video_id = ? ORDER BY start_sec", [video_id]).fetchall()
        video_clip_dir = clip_dir / video_id
        video_clip_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(video_clip_dir.glob("*.mp4"))
        if len(existing) > len(clips):
            raise RuntimeError(f"too many cached clips video={video_id} expected={len(clips)} actual={len(existing)}")
        clip_durations = []
        for clip_index, clip_path in enumerate(existing):
            row_id = clips[clip_index][0]
            if clip_path.name != f"{clip_index:06d}.mp4":
                raise RuntimeError(f"clip filename mismatch row_id={row_id} expected={clip_index:06d}.mp4 actual={clip_path.name}")
            if clip_path.stat().st_size == 0:
                raise RuntimeError(f"empty cached clip row_id={row_id} path={clip_path}")
            generated_duration = VideoDecoder(clip_path, seek_mode="approximate").metadata.duration_seconds
            if generated_duration is None or generated_duration <= 0:
                raise RuntimeError(f"cached clip has invalid duration row_id={row_id} path={clip_path} duration={generated_duration}")
            clip_durations.append(generated_duration)
        start_index = len(existing)
        if start_index < len(clips) and abs(sum(clip_durations) - video_duration) > 0.05:
            output_pattern = video_clip_dir / "%06d.mp4"
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
            if start_index:
                command.extend(["-ss", str(sum(clip_durations))])
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
        if len(generated) > len(clips):
            raise RuntimeError(f"too many generated clips video={video_id} expected={len(clips)} actual={len(generated)}")
        for clip_index in range(start_index, len(generated)):
            row_id = clips[clip_index][0]
            clip_path = generated[clip_index]
            if clip_path.name != f"{clip_index:06d}.mp4":
                raise RuntimeError(f"clip filename mismatch row_id={row_id} expected={clip_index:06d}.mp4 actual={clip_path.name}")
            if not clip_path.is_file() or clip_path.stat().st_size == 0:
                raise RuntimeError(f"missing cached clip row_id={row_id} path={clip_path}")
            generated_duration = VideoDecoder(clip_path, seek_mode="approximate").metadata.duration_seconds
            if generated_duration is None or generated_duration <= 0:
                raise RuntimeError(f"cached clip has invalid duration row_id={row_id} path={clip_path} duration={generated_duration}")
            clip_durations.append(generated_duration)
        if abs(sum(clip_durations) - video_duration) > 0.05:
            raise RuntimeError(f"cached video duration mismatch video={video_id} expected={video_duration} actual={sum(clip_durations)}")
        trailing_rows = clips[len(generated):]
        active_clips = clips[:len(generated)]
        connection.execute("BEGIN TRANSACTION")
        for row_id, _, _ in trailing_rows:
            connection.execute("DELETE FROM video_clips WHERE row_id = ?", [row_id])
        actual_start = 0.0
        for (row_id, _, _), generated_duration in zip(active_clips, clip_durations, strict=True):
            actual_end = actual_start + generated_duration
            connection.execute(
                "UPDATE video_clips SET start_sec = ?, end_sec = ?, global_start = ?, global_end = ? WHERE row_id = ?",
                [actual_start, actual_end, video_global_start + timedelta(seconds=actual_start), video_global_start + timedelta(seconds=actual_end), row_id],
            )
            actual_start = actual_end
        connection.execute("COMMIT")
        if trailing_rows:
            print(f"merge_trailing_video_clip_rows video={video_id} removed_row_ids={[row[0] for row in trailing_rows]}")
        print(f"clip_cache_verified video={video_id} clips={len(active_clips)} min_duration={min(clip_durations)} max_duration={max(clip_durations)}")
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    arguments = parser.parse_args()
    cache_video_clips(arguments.database, arguments.clip_dir)
