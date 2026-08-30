import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from src.preprocess.build_full_magic_index import ContextWindowExhausted, LocalModel, read_json, safe_key, write_json


class TimeRange(BaseModel):
    start_sec: float
    end_sec: float


class SummarySegment(TimeRange):
    summary: str


class IndexedEntity(BaseModel):
    name: str
    mentions: list[TimeRange]


class TimedTriple(TimeRange):
    subject: str
    predicate: str
    object: str


class HourIndex(BaseModel):
    three_minute_summaries: list[SummarySegment]
    ten_minute_summaries: list[SummarySegment]
    hour_summary: str
    entities: list[IndexedEntity]
    triples: list[TimedTriple]


async def run(args) -> None:
    caption_files = sorted(args.caption_root.glob("*.json"))
    captions = [read_json(path) for path in caption_files]
    captions = [caption for caption in captions if caption["video_id"] == args.video_id]
    captions.sort(key=lambda caption: caption["start_sec"])

    grouped = defaultdict(list)
    for caption in captions:
        grouped[int(caption["start_sec"] // 3600)].append(caption)
    all_jobs = [
        {
            "key": f"{args.video_id}:{hour_index:04d}",
            "hour_index": hour_index,
            "start_sec": rows[0]["start_sec"],
            "end_sec": rows[-1]["end_sec"],
            "captions": rows,
        }
        for hour_index, rows in sorted(grouped.items())
    ]
    jobs = all_jobs[:args.max_windows] if args.max_windows is not None else all_jobs
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    model = LocalModel(args.base_url, args.model, args.max_retries, args.artifact_root)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate_interval(stage, key, rows, start_sec, end_sec):
        caption_text = "\n".join(
            f"[{caption['start_sec']:.3f}-{caption['end_sec']:.3f}] {caption['text']}"
            for caption in rows
        )
        prompt = f"""Build one compact factual index from the following time-ordered captions covering at most one continuous hour.

In this single response:
1. Produce coherent approximately 3-minute summary segments covering the supplied interval.
2. Produce coherent approximately 10-minute summary segments covering the supplied interval.
3. Produce one summary for the complete supplied interval.
4. Extract each explicitly named entity once, with every evidence time range where it appears. Do not infer an unstated real name from a visual description.
5. Extract factual triples directly from the captions once. Give each triple the smallest evidence time range supported by the captions. Do not run a separate semantic-memory pass and do not repeat equivalent triples.

All start_sec and end_sec values must use the absolute seconds shown below and remain inside {start_sec:.3f}-{end_sec:.3f}.

Captions:
{caption_text}"""
        messages = [{"role": "user", "content": prompt}]

        def validate_coverage(result):
            if not result.three_minute_summaries or result.three_minute_summaries[0].start_sec > start_sec + 1 or result.three_minute_summaries[-1].end_sec < end_sec - 1:
                raise ValueError(f"three-minute summaries do not cover {start_sec:.3f}-{end_sec:.3f}")
            if not result.ten_minute_summaries or result.ten_minute_summaries[0].start_sec > start_sec + 1 or result.ten_minute_summaries[-1].end_sec < end_sec - 1:
                raise ValueError(f"ten-minute summaries do not cover {start_sec:.3f}-{end_sec:.3f}")

        async with semaphore:
            started = time.monotonic()
            result = await model.generate(stage, key, messages, messages, HourIndex, validate_coverage)
            latency = time.monotonic() - started
        return result, latency

    async def process(job):
        output_path = args.artifact_root / "shards" / "hour_index" / f"{safe_key(job['key'])}.json"
        if output_path.is_file():
            return read_json(output_path)["latency_sec"]
        half_boundary = job["hour_index"] * 3600 + 1800
        half_rows = [
            [caption for caption in job["captions"] if caption["start_sec"] < half_boundary],
            [caption for caption in job["captions"] if caption["start_sec"] >= half_boundary],
        ]
        half_paths = [args.artifact_root / "shards" / "half_hour_index" / f"{safe_key(job['key'])}_half_{half_index}.json" for half_index in range(2)]
        window_mode = "hour"
        hour_latency = 0.0
        if any(path.is_file() for path in half_paths):
            result = None
        else:
            try:
                result, hour_latency = await generate_interval("hour_index", job["key"], job["captions"], job["start_sec"], job["end_sec"])
            except ContextWindowExhausted:
                result = None
                window_mode = "two_half_hours"
        if result is None and window_mode == "two_half_hours" or any(path.is_file() for path in half_paths):
            window_mode = "two_half_hours"

            async def generate_half(half_index):
                half_path = half_paths[half_index]
                if half_path.is_file():
                    saved = read_json(half_path)
                    return HourIndex.model_validate(saved["index"]), saved["latency_sec"]
                rows = half_rows[half_index]
                half_result, half_latency = await generate_interval("half_hour_index", f"{job['key']}:half:{half_index}", rows, rows[0]["start_sec"], rows[-1]["end_sec"])
                if half_result is not None:
                    write_json(half_path, {"video_id": args.video_id, "hour_index": job["hour_index"], "half_index": half_index, "start_sec": rows[0]["start_sec"], "end_sec": rows[-1]["end_sec"], "caption_count": len(rows), "latency_sec": half_latency, "index": half_result.model_dump(mode="json")})
                return half_result, half_latency

            half_started = time.monotonic()
            halves = await asyncio.gather(generate_half(0), generate_half(1))
            if halves[0][0] is None or halves[1][0] is None:
                print(f"window_failed key={job['key']} attempts={args.max_retries}")
                return None
            first, second = halves[0][0], halves[1][0]
            entities = {}
            for half in (first, second):
                for entity in half.entities:
                    entities.setdefault(entity.name, []).extend(entity.mentions)
            result = HourIndex(
                three_minute_summaries=first.three_minute_summaries + second.three_minute_summaries,
                ten_minute_summaries=first.ten_minute_summaries + second.ten_minute_summaries,
                hour_summary=first.hour_summary + "\n\n" + second.hour_summary,
                entities=[IndexedEntity(name=name, mentions=mentions) for name, mentions in entities.items()],
                triples=first.triples + second.triples,
            )
            latency = hour_latency + time.monotonic() - half_started
        else:
            latency = hour_latency
        if result is None:
            print(f"window_failed key={job['key']} attempts={args.max_retries}")
            return None
        write_json(output_path, {
            "video_id": args.video_id,
            "hour_index": job["hour_index"],
            "start_sec": job["start_sec"],
            "end_sec": job["end_sec"],
            "caption_count": len(job["captions"]),
            "latency_sec": latency,
            "window_mode": window_mode,
            "index": result.model_dump(mode="json"),
        })
        print(f"window_complete key={job['key']} mode={window_mode} captions={len(job['captions'])} latency_sec={latency:.3f}")
        return latency

    started = time.monotonic()
    try:
        latencies = await asyncio.gather(*(process(job) for job in jobs))
    finally:
        await model.client.close()
    wall_sec = time.monotonic() - started
    completed = sum(latency is not None for latency in latencies)
    failed = len(latencies) - completed
    successful_latencies = [latency for latency in latencies if latency is not None]
    estimated_total_sec = sum(successful_latencies) / completed * math.ceil(len(all_jobs) / args.concurrency) if completed else None
    summary = {
        "video_id": args.video_id,
        "caption_count": len(captions),
        "total_windows": len(all_jobs),
        "tested_windows": len(jobs),
        "completed_windows": completed,
        "failed_windows": failed,
        "concurrency": args.concurrency,
        "benchmark_wall_sec": wall_sec,
        "estimated_total_sec": estimated_total_sec,
        "estimated_total_hours": estimated_total_sec / 3600 if estimated_total_sec is not None else None,
    }
    write_json(args.artifact_root / "eta.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--max-windows", type=int)
    arguments = parser.parse_args()
    if arguments.max_retries != 10:
        raise ValueError("max retries must be 10")
    asyncio.run(run(arguments))
