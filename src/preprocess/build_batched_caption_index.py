import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from src.preprocess.build_full_magic_index import LocalModel, read_json, safe_key, write_json


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

    async def process(job):
        output_path = args.artifact_root / "shards" / "hour_index" / f"{safe_key(job['key'])}.json"
        if output_path.is_file():
            return read_json(output_path)["latency_sec"]
        caption_text = "\n".join(
            f"[{caption['start_sec']:.3f}-{caption['end_sec']:.3f}] {caption['text']}"
            for caption in job["captions"]
        )
        prompt = f"""Build one compact factual index from the following time-ordered captions covering at most one continuous hour.

In this single response:
1. Produce coherent approximately 3-minute summary segments covering the supplied interval.
2. Produce coherent approximately 10-minute summary segments covering the supplied interval.
3. Produce one summary for the complete supplied interval.
4. Extract each explicitly named entity once, with every evidence time range where it appears. Do not infer an unstated real name from a visual description.
5. Extract factual triples directly from the captions once. Give each triple the smallest evidence time range supported by the captions. Do not run a separate semantic-memory pass and do not repeat equivalent triples.

All start_sec and end_sec values must use the absolute seconds shown below and remain inside {job['start_sec']:.3f}-{job['end_sec']:.3f}.

Captions:
{caption_text}"""
        messages = [{"role": "user", "content": prompt}]
        async with semaphore:
            started = time.monotonic()
            result = await model.generate("hour_index", job["key"], messages, messages, HourIndex)
            latency = time.monotonic() - started
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
            "index": result.model_dump(mode="json"),
        })
        print(f"window_complete key={job['key']} captions={len(job['captions'])} latency_sec={latency:.3f}")
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
