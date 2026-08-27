import json

from src.training import JUDGE_INSTRUCTIONS, serialize_judge_input


def test_judge_input_contains_only_reference_and_candidate_answers():
    payload = json.loads(serialize_judge_input("男生穿着红衣服", "穿红色衣服的男生"))
    assert payload == {
        "reference_answer": "男生穿着红衣服",
        "candidate_answer": "穿红色衣服的男生",
    }
    assert set(payload) == {"reference_answer", "candidate_answer"}


def test_numeric_reference_is_answer_content_not_identifier_metadata():
    assert "Treat every reference_answer value as answer content, never as identifier metadata." in JUDGE_INSTRUCTIONS
    assert "compare it only with the candidate's explicitly stated answer value" in JUDGE_INSTRUCTIONS
    assert "ignore incidental occurrences in video IDs, timestamps, row IDs, evidence citations, and other supporting details" in JUDGE_INSTRUCTIONS
