import json
from pathlib import Path

from src.splits import graph_partition


def test_pilot_graph_partition():
    root = Path("/raid/wg25r/mmlifelong_pilot/month")
    train, val = graph_partition(root / "train.json", root / "val.json")
    assert len(train) == 646
    assert len(val) == 24
    original = json.loads((root / "train.json").read_text(encoding="utf-8")) + json.loads((root / "val.json").read_text(encoding="utf-8"))
    train_ids = {question["index"] for question in train}
    val_ids = {question["index"] for question in val}
    assert [question["index"] for question in train] == [question["index"] for question in original if question["index"] in train_ids]
    assert [question["index"] for question in val] == [question["index"] for question in original if question["index"] in val_ids]
    train_videos = {str(clue["video_id"]) for question in train for clue in question["clue_interval"]}
    val_videos = {str(clue["video_id"]) for question in val for clue in question["clue_interval"]}
    assert train_videos.isdisjoint(val_videos)
