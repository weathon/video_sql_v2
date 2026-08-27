import asyncio
import json
import os
from pathlib import Path

from agents import Agent, ModelSettings, OpenAIResponsesModel
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.shared import Reasoning
import wandb

from src.agent_runtime import AgentRuntime, ModelEndpoint
from src.harness import VideoSQLHarness
from src.training import Judgment, MODEL, MODEL_CONTEXT_LIMIT, PILOT_COST_LIMIT_USD, ROOT, RUN_ROOT, serialize_responses, run_structured_agent, usage_and_cost


RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, ModelBehaviorError, MaxTurnsExceeded)


class PilotEvaluator:
    def __init__(self):
        api_key = os.environ["OPENAI_API_KEY"]
        self.endpoint = ModelEndpoint(MODEL, api_key, "https://api.openai.com/v1", "high", "24h")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = OpenAIResponsesModel(model=MODEL, openai_client=self.client)
        self.settings = ModelSettings(reasoning=Reasoning(effort="high"), truncation="disabled", prompt_cache_retention="24h", preserve_raw_usage=True)
        self.judge = Agent(
            name="pilot validation judge",
            instructions="Score the candidate answer against the reference answer and the complete rollout trace. Use 5 for fully correct and complete, 4 for correct with a minor imprecision, 3 for partially correct with a material omission, 2 for some relevant progress with a major error, 1 for minimal relevant content, and 0 for wrong, unsupported, missing, or failed. Judge the actual answer and evidence, not effort.",
            model=self.model,
            model_settings=self.settings,
            output_type=Judgment,
        )
        self.cumulative_cost = 0.0
        for path in list(RUN_ROOT.glob("train/question-*/rollout-*/result.json")) + list(RUN_ROOT.glob("train/question-*/summary.json")) + list(RUN_ROOT.glob("batches/batch-*/updater.json")) + list(RUN_ROOT.glob("val/*/question-*/result.json")):
            self.cumulative_cost += json.loads(path.read_text(encoding="utf-8"))["cost_usd"]
        wandb_id = (RUN_ROOT / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
        self.wandb_run = wandb.init(project="video-sql-v2", id=wandb_id, resume="must", name="pilot-luna-seed2026")

    async def run_question(self, question: dict, condition: str, skill: str) -> dict:
        question_id = int(question["index"])
        question_dir = RUN_ROOT / "val" / condition / f"question-{question_id}"
        result_path = question_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["new_result"] = False
            return payload
        question_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            attempt_dir = question_dir / f"attempt-{attempt}"
            harness = VideoSQLHarness(
                ROOT / "data" / "pilot.duckdb",
                "http://127.0.0.1:8012/v1/embeddings",
                "qwen3-vl-embedding-8b",
                ROOT / "data" / "embedding_index",
                ROOT / "data" / "video_clips",
                attempt_dir / "media",
                attempt_dir / "notes.json",
            )
            runtime = AgentRuntime(harness, self.endpoint, self.client, MODEL_CONTEXT_LIMIT, ROOT / "prompts" / "COMPACTION.md", ROOT / "prompts" / "REFLECTION_FINAL.md")
            try:
                rollout = await runtime.run_rollout(question["question"], skill, attempt_dir / "trace.json", False)
                break
            except RETRYABLE_ERRORS as error:
                attempt_dir.mkdir(parents=True, exist_ok=True)
                (attempt_dir / "error.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"validation_rollout_retry condition={condition} question_id={question_id} attempt={attempt} error={type(error).__name__}: {error}")
                if attempt == 3:
                    raise
        trace = json.loads(Path(rollout.trace_path).read_text(encoding="utf-8"))
        judge_input = json.dumps({"question": question["question"], "reference_answer": question["answer"], "candidate_answer": rollout.answer, "rollout_trace": trace}, ensure_ascii=False)
        judge_result = await run_structured_agent(self.judge, judge_input, "pilot-validation-judge", f"{condition}-question-{question_id}")
        responses = list(trace["model_responses"])
        for nested_trace in trace["nested_tool_traces"]:
            responses.extend(nested_trace["responses"])
        judge_responses = serialize_responses(judge_result)
        responses.extend(judge_responses)
        usage, cost = usage_and_cost(responses)
        payload = {
            "condition": condition,
            "question_id": question_id,
            "question_type": question["question_type"],
            "answer": rollout.answer,
            "reference_answer": question["answer"],
            "score": judge_result.final_output.score,
            "judgment": judge_result.final_output.model_dump(mode="json"),
            "trace_path": rollout.trace_path,
            "judge_responses": judge_responses,
            "usage": usage,
            "cost_usd": cost,
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["new_result"] = True
        return payload

    async def run(self) -> None:
        questions = json.loads((RUN_ROOT / "split" / "val.json").read_text(encoding="utf-8"))
        conditions = {
            "trained_skill": (RUN_ROOT / "skills" / "SKILL.md").read_text(encoding="utf-8"),
            "initial_skill": (ROOT / "skills" / "initial" / "SKILL.md").read_text(encoding="utf-8"),
        }
        results_by_condition = {}
        for condition, skill in conditions.items():
            async with asyncio.TaskGroup() as tasks:
                tasks_by_id = {int(question["index"]): tasks.create_task(self.run_question(question, condition, skill)) for question in questions}
            results = {question_id: task.result() for question_id, task in tasks_by_id.items()}
            results_by_condition[condition] = results
            for question_id, result in results.items():
                if result["new_result"]:
                    self.cumulative_cost += result["cost_usd"]
                self.wandb_run.log({f"pilot_val/{condition}_score": result["score"], "pilot_val/question_id": question_id, "pilot/cumulative_cost_usd": self.cumulative_cost, **{f"pilot_val/{condition}_usage_{key}": value for key, value in result["usage"].items()}})
            if self.cumulative_cost > PILOT_COST_LIMIT_USD:
                print(f"pilot_cost_limit_reached cumulative_cost_usd={self.cumulative_cost:.6f} completed_validation_condition={condition}")
                self.wandb_run.finish()
                await self.client.close()
                return

        paired = []
        by_type = {}
        for question in questions:
            question_id = int(question["index"])
            trained = results_by_condition["trained_skill"][question_id]["score"]
            initial = results_by_condition["initial_skill"][question_id]["score"]
            difference = trained - initial
            paired.append({"question_id": question_id, "question_type": question["question_type"], "trained_score": trained, "initial_score": initial, "difference": difference})
            question_type = question["question_type"]
            if question_type not in by_type:
                by_type[question_type] = []
            by_type[question_type].append(difference)
        report = {
            "questions": len(questions),
            "trained_mean": sum(item["trained_score"] for item in paired) / len(paired),
            "initial_mean": sum(item["initial_score"] for item in paired) / len(paired),
            "paired_mean_difference": sum(item["difference"] for item in paired) / len(paired),
            "paired_by_question_type": {question_type: sum(differences) / len(differences) for question_type, differences in by_type.items()},
            "paired_results": paired,
            "cumulative_cost_usd": self.cumulative_cost,
        }
        (RUN_ROOT / "val" / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.wandb_run.log({"pilot_val/trained_mean": report["trained_mean"], "pilot_val/initial_mean": report["initial_mean"], "pilot_val/paired_mean_difference": report["paired_mean_difference"], "pilot/cumulative_cost_usd": self.cumulative_cost})
        self.wandb_run.finish()
        await self.client.close()
        print(f"pilot_validation_complete questions={len(questions)} trained_mean={report['trained_mean']:.6f} initial_mean={report['initial_mean']:.6f} paired_mean_difference={report['paired_mean_difference']:.6f} cumulative_cost_usd={self.cumulative_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(PilotEvaluator().run())
