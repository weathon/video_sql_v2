from src.splits import partition_questions


def test_partition_questions_uses_disconnected_video_components():
    rows = [
        {"index": 1, "clue_interval": [{"video_id": "a"}]},
        {"index": 2, "clue_interval": [{"video_id": "a"}, {"video_id": "b"}]},
        {"index": 3, "clue_interval": [{"video_id": "b"}]},
        {"index": 4, "clue_interval": [{"video_id": "c"}]},
        {"index": 5, "clue_interval": [{"video_id": "missing"}]},
    ]
    train, test = partition_questions(rows, {"a", "b", "c"})
    assert [row["index"] for row in train] == [1, 2, 3]
    assert [row["index"] for row in test] == [4]
