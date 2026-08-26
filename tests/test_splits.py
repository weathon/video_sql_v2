from pathlib import Path

from src.splits import graph_partition


def test_pilot_graph_partition():
    root = Path("/raid/wg25r/mmlifelong_pilot/month")
    train, val = graph_partition(root / "train.json", root / "val.json")
    assert len(train) == 646
    assert len(val) == 24
    train_videos = {str(clue["video_id"]) for question in train for clue in question["clue_interval"]}
    val_videos = {str(clue["video_id"]) for question in val for clue in question["clue_interval"]}
    assert train_videos.isdisjoint(val_videos)
