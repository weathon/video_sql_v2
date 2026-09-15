"""Build an evaluate.py-shaped MAGIC open reference directory from the completed official gpt-5 judge results."""

import json
import sys
from pathlib import Path

from training_free.evaluate_mm_official import SYSTEM_PROMPT


judge_dir = Path(sys.argv[1]).resolve()
artifact_root = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3]).resolve()
questions = json.loads((artifact_root / "questions.json").read_text())
config = {"method": "magic", "dataset": "open", "judge_rubric": SYSTEM_PROMPT, "judge_version": "mm-lifelong-official-openai-gpt-5", "questions": questions,
          "source_judge_dir": str(judge_dir), "source_metrics": json.loads((judge_dir / "metrics.json").read_text())}
for question in questions:
    qid = str(question["id"])
    judged = json.loads((judge_dir / "questions" / qid / "result.json").read_text())
    if judged["question"] != question["question"] or judged["answer"] != question["reference_answer"]:
        raise ValueError(f"Judge result does not match the artifact question: {qid}")
    directory = out / "magic/questions" / qid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"id": question["id"], "category": "answered", "score": judged["gpt_score"], "candidate_answer": judged["pred"]["answer"], "source": judged["trace_file"]}, ensure_ascii=False, indent=2) + "\n")
(out / "config.json").write_text(json.dumps({"evaluation": config}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {out} questions={len(questions)}", flush=True)
