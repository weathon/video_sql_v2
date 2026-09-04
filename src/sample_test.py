import json
import random
from pathlib import Path

from src.harness import DATASET_SNAPSHOT


rng = random.Random(0)
sampled = []
for timeline, split in (("month", "month/val.json"), ("day", "day/test.json"), ("week", "week/test.json")):
    rows = json.loads((DATASET_SNAPSHOT / split).read_text())
    for row in rng.sample(rows, 100):
        sampled.append(
            {
                "timeline": timeline,
                "source_split": split,
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
            }
        )

Path("data/test-seed-0.json").write_text(json.dumps(sampled, ensure_ascii=False, indent=2) + "\n")
