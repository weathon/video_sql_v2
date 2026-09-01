import argparse
import json
from pathlib import Path


def partition_questions(rows: list[dict], available_video_ids: set[str]) -> tuple[list[dict], list[dict]]:
    eligible = [row for row in rows if {str(clue["video_id"]) for clue in row["clue_interval"]} <= available_video_ids]
    videos_by_question = {row["index"]: {str(clue["video_id"]) for clue in row["clue_interval"]} for row in eligible}
    rows_by_id = {row["index"]: row for row in eligible}
    unseen = set(videos_by_question)
    components = []
    while unseen:
        seed = unseen.pop()
        question_ids = {seed}
        video_ids = set(videos_by_question[seed])
        changed = True
        while changed:
            added = {question_id for question_id in unseen if videos_by_question[question_id] & video_ids}
            changed = bool(added)
            unseen -= added
            question_ids |= added
            for question_id in added:
                video_ids |= videos_by_question[question_id]
        components.append(question_ids)
    components.sort(key=lambda component: (-len(component), min(component)))
    train_ids = components[0]
    test_ids = set().union(*components[1:])
    return ([rows_by_id[question_id] for question_id in sorted(train_ids)], [rows_by_id[question_id] for question_id in sorted(test_ids)])


parser = argparse.ArgumentParser()
parser.add_argument("--train-json", type=Path, required=True)
parser.add_argument("--val-json", type=Path, required=True)
parser.add_argument("--artifact-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
questions = json.loads(args.train_json.read_text()) + json.loads(args.val_json.read_text())
available = {path.name for path in (args.artifact_root / "mmlifelong").iterdir() if path.is_dir()}
train, test = partition_questions(questions, available)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2) + "\n")
(args.output_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"train": len(train), "test": len(test), "train_ids": [row["index"] for row in train], "test_ids": [row["index"] for row in test]}))
