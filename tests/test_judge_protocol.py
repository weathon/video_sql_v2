import json

from src.training import serialize_judge_input


def test_judge_input_contains_only_reference_and_candidate_answers():
    payload = json.loads(serialize_judge_input("男生穿着红衣服", "穿红色衣服的男生"))
    assert payload == {
        "reference_answer": "男生穿着红衣服",
        "candidate_answer": "穿红色衣服的男生",
    }
    assert set(payload) == {"reference_answer", "candidate_answer"}
