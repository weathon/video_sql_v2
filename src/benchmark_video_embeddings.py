import argparse
import asyncio
import json
from pathlib import Path
import time

import duckdb
import httpx
import numpy as np

from src.embeddings import request_embeddings

BENCHMARK_CLIPS = 256


async def benchmark(arguments) -> None:
    connection = duckdb.connect(str(arguments.database), read_only=True)
    rows = connection.execute("SELECT c.row_id, c.video_id FROM video_clips c JOIN videos v USING(video_id) ORDER BY v.sequence_index, c.start_sec").fetchall()
    connection.close()
    selected = [rows[index] for index in np.linspace(0, len(rows) - 1, BENCHMARK_CLIPS, dtype=int)]
    arguments.output_dir.mkdir(parents=True)
    semaphore = asyncio.Semaphore(arguments.max_in_flight)
    timeout = httpx.Timeout(1800)
    limits = httpx.Limits(max_connections=arguments.max_in_flight, max_keepalive_connections=arguments.max_in_flight)
    latencies = []
    skipped = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def run_one(position, row_id, video_id):
            async with semaphore:
                clip_index = int(row_id.rsplit(":", 1)[1])
                clip_path = arguments.clip_dir / video_id / f"{clip_index:06d}.mp4"
                if not clip_path.is_file():
                    raise FileNotFoundError(clip_path)
                conversations = [[
                    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": clip_path.resolve().as_uri()}}, {"type": "text", "text": ""}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                ]]
                started = time.perf_counter()
                try:
                    result = await request_embeddings(client, arguments.endpoint, arguments.model, conversations, [row_id], None)
                except httpx.HTTPStatusError as error:
                    if error.response.status_code != 500:
                        raise
                    skipped.append(row_id)
                    print(f"benchmark_skip row_id={row_id} clip_path={clip_path} status=500")
                    return
                latency = time.perf_counter() - started
                np.savez(arguments.output_dir / f"video-{position:06d}.npz", row_ids=np.array([row_id]), vectors=np.array([result[0][1]], dtype=np.float32))
                latencies.append(latency)
                print(f"benchmark_result position={position} row_id={row_id} latency_sec={latency:.6f}")

        started = time.perf_counter()
        async with asyncio.TaskGroup() as tasks:
            for position, (row_id, video_id) in enumerate(selected):
                tasks.create_task(run_one(position, row_id, video_id))
        elapsed = time.perf_counter() - started

    summary = {
        "selected": len(selected),
        "succeeded": len(latencies),
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "clips_per_second": len(latencies) / elapsed,
        "latency_p50_sec": float(np.percentile(latencies, 50)),
        "latency_p95_sec": float(np.percentile(latencies, 95)),
        "max_in_flight": arguments.max_in_flight,
        "row_ids": [row_id for row_id, _ in selected],
    }
    (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"benchmark_complete output_dir={arguments.output_dir} succeeded={summary['succeeded']} skipped={len(skipped)} elapsed_sec={elapsed:.6f} clips_per_second={summary['clips_per_second']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-in-flight", type=int, required=True)
    asyncio.run(benchmark(parser.parse_args()))
