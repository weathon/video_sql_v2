"""Compare the two EgoLife runs (our training-free harness on the MAGIC index vs MAGIC-video) and write RESULTS.md plus a stats json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runs/egolife-week-shared"
RUNS = {
    "training-free (MAGIC index)": ROOT / "runs/training-free-egolife-week-magic-index",
    "MAGIC-video": ROOT / "runs/magic-video-egolife-week",
}


def load_run(run_dir, questions):
    results = {}
    skipped = []
    for question in questions:
        directory = run_dir / "qa" / f"question-{question['ID']}"
        result_path = directory / "rollout-00" / "result.json"
        if not result_path.exists():
            result_path = directory / "result.json"
        if result_path.exists():
            results[question["ID"]] = json.loads(result_path.read_text())
        elif (directory / "rollout-00" / "skipped.json").exists() or (directory / "skipped.json").exists():
            skipped.append(question["ID"])
        else:
            raise RuntimeError(f"{run_dir.name}: question {question['ID']} has neither result.json nor skipped.json")
    return results, skipped


def stats(rows):
    scores = [r["score"] for r in rows]
    return {"questions": len(scores), "total_score": sum(scores), "mean_score": sum(scores) / len(scores) if scores else None,
            "accuracy_score_5": sum(s == 5 for s in scores) / len(scores) if scores else None}


def main():
    questions = json.loads((SHARED / "questions.json").read_text())
    types = sorted({q["type"] for q in questions})
    loaded = {name: load_run(path, questions) for name, path in RUNS.items()}
    table = {}
    for name, (results, skipped) in loaded.items():
        rows = list(results.values())
        table[name] = {"overall": stats(rows), "by_type": {t: stats([r for r in rows if r["type"] == t]) for t in types}, "skipped": skipped}
    names = list(RUNS)
    lines = ["# EgoLifeQA (A1_JAKE, 500 open-ended questions) on the MAGIC-video artifacts index", "",
             "Scores are the 0–5 judge scores; accuracy is the fraction of questions scored exactly 5. Skipped questions are excluded from every denominator.", "",
             "| group | metric | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for group in ["overall", *types]:
        for metric in ["questions", "total_score", "mean_score", "accuracy_score_5"]:
            values = []
            for name in names:
                value = table[name]["overall"][metric] if group == "overall" else table[name]["by_type"][group][metric]
                values.append("-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)))
            lines.append(f"| {group} | {metric} | " + " | ".join(values) + " |")
    lines.append("")
    for name in names:
        lines.append(f"Skipped question IDs, {name}: {', '.join(table[name]['skipped']) if table[name]['skipped'] else 'none'}")
    lines.append("")
    (ROOT / "training_free/RESULTS.md").write_text("\n".join(lines))
    (SHARED / "results_stats.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
