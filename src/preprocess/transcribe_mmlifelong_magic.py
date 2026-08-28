import concurrent.futures
import json
import logging
import multiprocessing
import os
import re
import time
from pathlib import Path

gpu_ids = (1, 4, 5, 6)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

import wandb
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO)
video_root = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/videos")
output_root = Path("/raid/wg25r/video_sql_v2/runs/magic_whisper_full")
workers_per_gpu = 512


def validate_complete(video_key, json_path, srt_path):
    with json_path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    if not isinstance(existing, list):
        raise TypeError(f"expected list in {json_path}")
    with srt_path.open(encoding="utf-8") as handle:
        srt_blocks = [block.splitlines() for block in re.split(r"\n{2,}", handle.read().strip()) if block.strip()]
    if len(srt_blocks) != len(existing):
        raise ValueError(
            f"JSON/SRT segment count mismatch video={video_key} "
            f"json={len(existing)} srt={len(srt_blocks)}"
        )
    for segment_index, (entry, block) in enumerate(zip(existing, srt_blocks), 1):
        if set(entry) != {"start_sec", "end_sec", "text"}:
            raise ValueError(f"invalid JSON segment video={video_key} index={segment_index}")
        if not isinstance(entry["start_sec"], (int, float)) or not isinstance(entry["end_sec"], (int, float)):
            raise TypeError(f"invalid timestamps video={video_key} index={segment_index}")
        if entry["start_sec"] < 0 or entry["end_sec"] < entry["start_sec"]:
            raise ValueError(f"invalid timestamp range video={video_key} index={segment_index}")
        if not isinstance(entry["text"], str):
            raise TypeError(f"invalid text video={video_key} index={segment_index}")
        if len(block) < 2 or block[0] != str(segment_index):
            raise ValueError(f"invalid SRT block video={video_key} index={segment_index}")
        timestamp_match = re.fullmatch(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})",
            block[1],
        )
        if timestamp_match is None:
            raise ValueError(f"invalid SRT timestamp video={video_key} index={segment_index}")
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
            raise ValueError(f"JSON/SRT timestamp mismatch video={video_key} index={segment_index}")
        if "\n".join(block[2:]) != entry["text"]:
            raise ValueError(f"JSON/SRT text mismatch video={video_key} index={segment_index}")
    return len(existing)


def transcribe_video(model, gpu_id, video_path_string):
    video_path = Path(video_path_string)
    relative_path = video_path.relative_to(video_root)
    video_key = relative_path.as_posix()
    whisper_dir = output_root / relative_path.parent / relative_path.stem / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    json_path = whisper_dir / f"{relative_path.stem}_whisper.json"
    srt_path = whisper_dir / f"{relative_path.stem}_whisper.srt"

    if json_path.is_file() and srt_path.is_file():
        segment_count = validate_complete(video_key, json_path, srt_path)
        print(f"skip_complete gpu={gpu_id} video={video_key} segments={segment_count}", flush=True)
        return {"video": video_key, "segments": segment_count, "skipped": True}

    video_started = time.monotonic()
    print(f"transcribe_start gpu={gpu_id} video={video_key} path={video_path}", flush=True)
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(
        f"vad_complete gpu={gpu_id} video={video_key} duration={info.duration:.3f} "
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
                f"transcribe_progress gpu={gpu_id} video={video_key} segments={segment_index} "
                f"media_end={segment.end:.3f}",
                flush=True,
            )

    with srt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(srt_lines))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    segment_count = validate_complete(video_key, json_path, srt_path)

    video_elapsed = time.monotonic() - video_started
    realtime_speed = info.duration / video_elapsed
    print(
        f"transcribe_complete gpu={gpu_id} video={video_key} segments={segment_count} "
        f"elapsed={video_elapsed:.3f} realtime_speed={realtime_speed:.3f} "
        f"json={json_path} srt={srt_path}",
        flush=True,
    )
    return {
        "video": video_key,
        "segments": segment_count,
        "skipped": False,
        "audio_hours": info.duration / 3600,
        "speech_hours": info.duration_after_vad / 3600,
        "elapsed_seconds": video_elapsed,
        "realtime_speed": realtime_speed,
    }


def run_gpu(gpu_id, device_index, video_path_strings):
    print(f"gpu_worker_start gpu={gpu_id} files={len(video_path_strings)} concurrency={workers_per_gpu}", flush=True)
    model = WhisperModel(
        "large-v3-turbo",
        device="cuda",
        device_index=device_index,
        compute_type="float16",
        num_workers=workers_per_gpu,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_per_gpu) as executor:
        futures = [executor.submit(transcribe_video, model, gpu_id, path) for path in video_path_strings]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    print(f"gpu_worker_complete gpu={gpu_id} files={len(results)}", flush=True)
    return results


def main():
    video_paths = (
        sorted((video_root / "month").glob("*.mp4"), key=lambda path: int(path.stem))
        + sorted((video_root / "day").glob("*.mp4"), key=lambda path: int(path.stem))
        + sorted((video_root / "week").glob("*/*.mp4"))
    )
    for video_path in video_paths:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

    run = wandb.init(
        project="video-sql-v2",
        name="magic-whisper-mmlifelong-full",
        config={
            "source": "MAGIC-video@0f2273f7f07360cc58588399553b5a6ea758b084",
            "dataset": "MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443",
            "model": "large-v3-turbo",
            "device": "cuda",
            "gpu_ids": list(gpu_ids),
            "workers_per_gpu": workers_per_gpu,
            "compute_type": "float16",
            "beam_size": 5,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 500,
            "video_files": len(video_paths),
        },
    )

    job_started = time.monotonic()
    shards = [[str(path) for path in video_paths[index::len(gpu_ids)]] for index in range(len(gpu_ids))]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=context) as executor:
        futures = [executor.submit(run_gpu, gpu_id, device_index, shard) for device_index, (gpu_id, shard) in enumerate(zip(gpu_ids, shards))]
        nested_results = [future.result() for future in concurrent.futures.as_completed(futures)]

    results = [result for gpu_results in nested_results for result in gpu_results]
    completed_segments = sum(result["segments"] for result in results)
    generated_results = [result for result in results if not result["skipped"]]
    run.log({
        "completed_videos": len(results),
        "completed_segments": completed_segments,
        "generated_videos": len(generated_results),
        "generated_audio_hours": sum(result["audio_hours"] for result in generated_results),
        "generated_speech_hours": sum(result["speech_hours"] for result in generated_results),
    })
    job_elapsed = time.monotonic() - job_started
    print(
        f"all_complete videos={len(results)} segments={completed_segments} "
        f"elapsed_hours={job_elapsed / 3600:.3f}",
        flush=True,
    )
    run.finish()


if __name__ == "__main__":
    main()
