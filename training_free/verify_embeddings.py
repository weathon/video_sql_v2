"""Re-embed sampled rows of the official-caption week shards on the verification embedder and compare cosines."""

import json
import random
import sys
from pathlib import Path

import httpx
import numpy as np

STAGE2 = Path("/raid/wg25r/video_sql_v2/data/pilot/stage2")
OUT = Path(__file__).resolve().parents[1] / "runs/egolife-week-shared/verification"
EMBED_URL = "http://127.0.0.1:8015/v1/embeddings"
TEXT_TABLES = ["episodes", "entities", "semantic_triples", "topic_facts", "event_steps", "object_states"]
TEXT_SAMPLE = 50
VIDEO_SAMPLE = 20


def week_rows(kind, table):
    rows = []
    for index, shard in enumerate(sorted((STAGE2 / "embedding_shards" / kind / table).glob("*.npz"))):
        loaded = np.load(shard, allow_pickle=False)
        row_ids = [str(x) for x in loaded["row_ids"]]
        if len(row_ids) != 1:
            raise RuntimeError(f"shard {shard} holds {len(row_ids)} rows; expected one row per shard")
        if row_ids[0].startswith("week:"):
            rows.append((index, row_ids[0], loaded["vectors"][0].astype(np.float32), shard))
    return rows


def embed(payload, client):
    response = client.post(EMBED_URL, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    if len(data) != 1 or data[0]["index"] != 0:
        raise RuntimeError(f"invalid embedding response indexes: {data}")
    vector = np.asarray(data[0]["embedding"], dtype=np.float32)
    if vector.shape != (4096,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid vector shape or values: {vector.shape}")
    return vector


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(0)
    client = httpx.Client(timeout=600)
    report = {"embed_url": EMBED_URL, "text": {}, "video": {}}
    for table in TEXT_TABLES:
        rows = week_rows("text", table)
        sample = randomizer.sample(rows, TEXT_SAMPLE)
        cosines = []
        details = []
        for index, row_id, stored, shard in sample:
            trace = json.loads((STAGE2 / "run/traces/embeddings/text" / table / f"{index:08d}" / "attempt-1.json").read_text())
            if trace["response"]["shard"] != str(shard):
                raise RuntimeError(f"trace shard {trace['response']['shard']} != {shard}")
            text = trace["request"]["input"]
            if len(text) != 1:
                raise RuntimeError(f"trace {index} for {table} has {len(text)} inputs")
            fresh = embed({"model": "qwen3-vl-embedding-8b", "input": text}, client)
            cosine = float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
            cosines.append(cosine)
            details.append({"row_id": row_id, "shard_index": index, "cosine": cosine, "text_prefix": text[0][:120]})
            print(f"{table} {row_id} cos={cosine:.6f}", flush=True)
        report["text"][table] = {
            "week_rows_in_shards": len(rows), "sampled": len(sample),
            "cosine_min": min(cosines), "cosine_mean": sum(cosines) / len(cosines), "cosine_max": max(cosines),
            "details": details,
        }
        (OUT / "embedding_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if "--skip-video" not in sys.argv:
        rows = week_rows("video", "visual_clips")
        sample = randomizer.sample(rows, VIDEO_SAMPLE)
        cosines = []
        details = []
        for index, row_id, stored, shard in sample:
            trace = json.loads((STAGE2 / "run/traces/embeddings/video/visual_clips" / f"{index:08d}" / "attempt-1.json").read_text())
            if trace["response"]["shard"] != str(shard):
                raise RuntimeError(f"trace shard {trace['response']['shard']} != {shard}")
            request = trace["request"]
            fresh = embed(request, client)
            cosine = float(stored @ fresh / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
            cosines.append(cosine)
            details.append({"row_id": row_id, "shard_index": index, "cosine": cosine, "request_input": request["input"]})
            print(f"visual_clips {row_id} cos={cosine:.6f}", flush=True)
        report["video"]["visual_clips"] = {
            "week_rows_in_shards": len(rows), "sampled": len(sample),
            "cosine_min": min(cosines), "cosine_mean": sum(cosines) / len(cosines), "cosine_max": max(cosines),
            "details": details,
        }
        (OUT / "embedding_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT / 'embedding_check.json'}", flush=True)


if __name__ == "__main__":
    main()
