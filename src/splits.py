import argparse
import json
from pathlib import Path


def graph_partition(train_path: Path, val_path: Path) -> tuple[list[dict], list[dict]]:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    questions = train + val
    video_to_questions: dict[str, set[int]] = {}
    for position, question in enumerate(questions):
        for clue in question["clue_interval"]:
            video_to_questions.setdefault(str(clue["video_id"]), set()).add(position)
    adjacency = [set() for _ in questions]
    for positions in video_to_questions.values():
        for position in positions:
            adjacency[position].update(positions)
    components = []
    unseen = set(range(len(questions)))
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            position = stack.pop()
            component.append(position)
            neighbors = adjacency[position] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        components.append(component)
    components.sort(key=len, reverse=True)
    if [len(component) for component in components] != [646, 24]:
        raise ValueError(f"expected graph components [646, 24], received {[len(component) for component in components]}")
    train_positions = set(components[0])
    val_positions = set(components[1])
    return [question for position, question in enumerate(questions) if position in train_positions], [question for position, question in enumerate(questions) if position in val_positions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    partition_train, partition_val = graph_partition(arguments.train, arguments.val)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "train.json").write_text(json.dumps(partition_train, ensure_ascii=False, indent=2), encoding="utf-8")
    (arguments.output_dir / "val.json").write_text(json.dumps(partition_val, ensure_ascii=False, indent=2), encoding="utf-8")
