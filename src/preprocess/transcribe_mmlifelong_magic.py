import json
import logging
import re
import time
from pathlib import Path

import wandb
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO)
video_dir = Path("/raid/wg25r/mmlifelong_pilot/videos/month")
output_root = Path("/raid/wg25r/video_sql_v2/runs/magic_whisper_all")
video_ids = ["4", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

for video_id in video_ids:
    video_path = video_dir / f"{video_id}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

run = wandb.init(
    project="video-sql-v2",
    name="magic-whisper-mmlifelong-all",
    config={
        "source": "MAGIC-video@0f2273f7f07360cc58588399553b5a6ea758b084",
        "model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_filter": True,
        "vad_min_silence_duration_ms": 500,
        "video_ids": video_ids,
    },
)

job_started = time.monotonic()
completed_videos = 0
completed_segments = 0
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

for video_id in video_ids:
    video_path = video_dir / f"{video_id}.mp4"
    whisper_dir = output_root / video_id / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    json_path = whisper_dir / f"{video_id}_whisper.json"
    srt_path = whisper_dir / f"{video_id}_whisper.srt"

    if json_path.is_file() and srt_path.is_file():
        with json_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, list):
            raise TypeError(f"expected list in {json_path}")
        with srt_path.open(encoding="utf-8") as handle:
            srt_blocks = [block.splitlines() for block in handle.read().strip().split("\n\n") if block.strip()]
        if len(srt_blocks) != len(existing):
            raise ValueError(
                f"JSON/SRT segment count mismatch video={video_id} "
                f"json={len(existing)} srt={len(srt_blocks)}"
            )
        for segment_index, (entry, block) in enumerate(zip(existing, srt_blocks), 1):
            if set(entry) != {"start_sec", "end_sec", "text"}:
                raise ValueError(f"invalid JSON segment video={video_id} index={segment_index}")
            if not isinstance(entry["start_sec"], (int, float)) or not isinstance(entry["end_sec"], (int, float)):
                raise TypeError(f"invalid timestamps video={video_id} index={segment_index}")
            if entry["start_sec"] < 0 or entry["end_sec"] < entry["start_sec"]:
                raise ValueError(f"invalid timestamp range video={video_id} index={segment_index}")
            if not isinstance(entry["text"], str):
                raise TypeError(f"invalid text video={video_id} index={segment_index}")
            if len(block) < 3 or block[0] != str(segment_index):
                raise ValueError(f"invalid SRT block video={video_id} index={segment_index}")
            timestamp_match = re.fullmatch(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})",
                block[1],
            )
            if timestamp_match is None:
                raise ValueError(f"invalid SRT timestamp video={video_id} index={segment_index}")
            srt_start = (
                int(timestamp_match[1]) * 3600
                + int(timestamp_match[2]) * 60
                + int(timestamp_match[3])
                + int(timestamp_match[4]) / 1000
            )
            srt_end = (
                int(timestamp_match[5]) * 3600
                + int(timestamp_match[6]) * 60
                + int(timestamp_match[7])
                + int(timestamp_match[8]) / 1000
            )
            if abs(srt_start - entry["start_sec"]) > 0.0015 or abs(srt_end - entry["end_sec"]) > 0.0015:
                raise ValueError(f"JSON/SRT timestamp mismatch video={video_id} index={segment_index}")
            if "\n".join(block[2:]) != entry["text"]:
                raise ValueError(f"JSON/SRT text mismatch video={video_id} index={segment_index}")
        completed_videos += 1
        completed_segments += len(existing)
        print(f"skip_complete video={video_id} segments={len(existing)}", flush=True)
        run.log({"completed_videos": completed_videos, "completed_segments": completed_segments})
        continue

    video_started = time.monotonic()
    print(f"transcribe_start video={video_id} path={video_path}", flush=True)
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(
        f"vad_complete video={video_id} duration={info.duration:.3f} "
        f"duration_after_vad={info.duration_after_vad:.3f} options={info.vad_options}",
        flush=True,
    )

    results = []
    srt_lines = []
    for segment_index, segment in enumerate(segments, 1):
        results.append({
            "start_sec": round(segment.start, 3),
            "end_sec": round(segment.end, 3),
            "text": segment.text.strip(),
        })

        start_h = int(segment.start // 3600)
        start_m = int((segment.start % 3600) // 60)
        start_s = int(segment.start % 60)
        start_ms = int((segment.start % 1) * 1000)
        end_h = int(segment.end // 3600)
        end_m = int((segment.end % 3600) // 60)
        end_s = int(segment.end % 60)
        end_ms = int((segment.end % 1) * 1000)
        srt_lines.append(str(segment_index))
        srt_lines.append(
            f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> "
            f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
        )
        srt_lines.append(segment.text.strip())
        srt_lines.append("")

        if segment_index % 250 == 0:
            print(
                f"transcribe_progress video={video_id} segments={segment_index} "
                f"media_end={segment.end:.3f}",
                flush=True,
            )
            run.log({
                "current_video": int(video_id),
                "current_video_segments": segment_index,
                "current_media_end_sec": segment.end,
            })

    with srt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(srt_lines))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    video_elapsed = time.monotonic() - video_started
    completed_videos += 1
    completed_segments += len(results)
    realtime_speed = info.duration / video_elapsed
    print(
        f"transcribe_complete video={video_id} segments={len(results)} "
        f"elapsed={video_elapsed:.3f} realtime_speed={realtime_speed:.3f} "
        f"json={json_path} srt={srt_path}",
        flush=True,
    )
    run.log({
        "completed_videos": completed_videos,
        "completed_segments": completed_segments,
        "last_video": int(video_id),
        "last_video_audio_hours": info.duration / 3600,
        "last_video_speech_hours": info.duration_after_vad / 3600,
        "last_video_elapsed_seconds": video_elapsed,
        "last_video_realtime_speed": realtime_speed,
    })

job_elapsed = time.monotonic() - job_started
print(
    f"all_complete videos={completed_videos} segments={completed_segments} "
    f"elapsed_hours={job_elapsed / 3600:.3f}",
    flush=True,
)
run.finish()
