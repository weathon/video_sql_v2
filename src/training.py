import asyncio
import json
import os
from pathlib import Path
import random
import shutil

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field
import wandb
from wandb.sdk.lib.runid import generate_id

from src.agent_runtime import AgentRuntime, ModelEndpoint, RolloutResult, serialize_usage
from src.harness import VideoSQLHarness
from src.splits import graph_partition


MODEL = "gpt-5.6-luna"
MODEL_CONTEXT_LIMIT = 1_050_000
SEED = 2026
BATCH_QUESTIONS = 5
ROLLOUTS_PER_QUESTION = 5
PILOT_COST_LIMIT_USD = 100.0
INPUT_USD_PER_MILLION = 0.20
CACHED_INPUT_USD_PER_MILLION = 0.02
CACHE_WRITE_USD_PER_MILLION = 0.25
OUTPUT_USD_PER_MILLION = 1.20
LONG_CONTEXT_THRESHOLD = 272_000
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, ModelBehaviorError, MaxTurnsExceeded)

ROOT = Path(__file__).resolve().parents[1]
PILOT_SOURCE = Path("/raid/wg25r/mmlifelong_pilot/month")
RUN_ROOT = ROOT / "runs" / "pilot_luna"


class Judgment(BaseModel):
    score: int = Field(ge=0, le=5)
    rationale: str


class Reflection(BaseModel):
    reflection: str
    meta_reflection: str


class QuestionSummary(BaseModel):
    summary: str


class SkillUpdate(BaseModel):
    skill_md: str
    meta_skill_md: str


def serialize_responses(result) -> list[dict]:
    return [
        {
            "output": [item.model_dump(mode="json") for item in response.output],
            "usage": serialize_usage(response.usage),
            "response_id": response.response_id,
            "request_id": response.request_id,
            "raw_usage": response.raw_usage,
        }
        for response in result.raw_responses
    ]


def usage_and_cost(responses: list[dict]) -> tuple[dict, float]:
    totals = {"requests": 0, "ordinary_input_tokens": 0, "cache_write_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    cost = 0.0
    for response in responses:
        usage = response["usage"]
        input_tokens = usage["input_tokens"]
        cached_tokens = usage["input_tokens_details"]["cached_tokens"]
        cache_write_tokens = usage["input_tokens_details"]["cache_write_tokens"]
        if not isinstance(cached_tokens, int) or not isinstance(cache_write_tokens, int):
            raise TypeError(f"normalized cache usage must be integers: {usage['input_tokens_details']}")
        ordinary_tokens = input_tokens - cached_tokens - cache_write_tokens
        if ordinary_tokens < 0:
            raise ValueError(f"cache token details exceed input tokens: {usage}")
        long_input_multiplier = 2.0 if input_tokens > LONG_CONTEXT_THRESHOLD else 1.0
        long_output_multiplier = 1.5 if input_tokens > LONG_CONTEXT_THRESHOLD else 1.0
        cost += ordinary_tokens * INPUT_USD_PER_MILLION * long_input_multiplier / 1_000_000
        cost += cached_tokens * CACHED_INPUT_USD_PER_MILLION * long_input_multiplier / 1_000_000
        cost += cache_write_tokens * CACHE_WRITE_USD_PER_MILLION * long_input_multiplier / 1_000_000
        cost += usage["output_tokens"] * OUTPUT_USD_PER_MILLION * long_output_multiplier / 1_000_000
        totals["requests"] += usage["requests"]
        totals["ordinary_input_tokens"] += ordinary_tokens
        totals["cache_write_tokens"] += cache_write_tokens
        totals["cached_input_tokens"] += cached_tokens
        totals["output_tokens"] += usage["output_tokens"]
    return totals, cost


def trace_for_reflection(trace: dict) -> dict:
    return {
        "question": trace["question"],
        "status": trace["status"],
        "answer": trace["answer"],
        "turns": trace["turns"],
        "compactions": [{"index": item["index"], "handoff": item["handoff"]} for item in trace["compactions"]],
    }


async def run_structured_agent(agent: Agent, agent_input: str, workflow_name: str, group_id: str, responses: list[dict]):
    corrected_input = agent_input
    for attempt in range(1, 4):
        try:
            result = await Runner.run(agent, corrected_input, max_turns=1, run_config=RunConfig(reasoning_item_id_policy="preserve", workflow_name=workflow_name, group_id=group_id))
            responses.extend(serialize_responses(result))
            return result
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as error:
            print(f"llm_boundary_retry workflow={workflow_name} group_id={group_id} attempt={attempt} error={type(error).__name__}: {error}")
            if attempt == 3:
                raise
        except ModelBehaviorError as error:
            if error.run_data is not None:
                responses.extend({"output": [item.model_dump(mode="json") for item in response.output], "usage": serialize_usage(response.usage), "response_id": response.response_id, "request_id": response.request_id, "raw_usage": response.raw_usage} for response in error.run_data.raw_responses)
            print(f"llm_structure_retry workflow={workflow_name} group_id={group_id} attempt={attempt} error={error}")
            if attempt == 3:
                raise
            corrected_input = agent_input + f"\n\nThe previous attempt failed structured-output validation with this exact error: {error}. Return a valid response matching the required schema."


class PilotTrainer:
    def __init__(self):
        api_key = os.environ["OPENAI_API_KEY"]
        self.endpoint = ModelEndpoint(MODEL, api_key, "https://api.openai.com/v1", "high", "24h")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = OpenAIResponsesModel(model=MODEL, openai_client=self.client)
        self.settings = ModelSettings(reasoning=Reasoning(effort="high"), truncation="disabled", prompt_cache_retention="24h", preserve_raw_usage=True)
        self.judge = Agent(
            name="pilot rollout judge",
            instructions="Compare only the candidate answer with the reference answer. Score semantic correctness and completeness, treating meaning-preserving paraphrases as correct. Use 5 for fully correct and complete, 4 for correct with a minor imprecision, 3 for partially correct with a material omission, 2 for some correct content with a major error, 1 for minimally correct content, and 0 for wrong or missing. Do not evaluate, request, or infer anything about the rollout trace, retrieval process, evidence, or effort.",
            model=self.model,
            model_settings=self.settings,
            output_type=Judgment,
        )
        self.summarizer = Agent(
            name="pilot question summarizer",
            instructions="Summarize the five rollout outcomes for skill learning. Identify retrieval decisions shared by successes and failures, preserve reusable evidence-handling lessons, and do not include the specific reference answer as a reusable rule.",
            model=self.model,
            model_settings=self.settings,
            output_type=QuestionSummary,
        )
        self.updater = Agent(
            name="pilot skill updater",
            instructions="Update SKILL.md and META-SKILL.md from the five question summaries. Return the complete new content of both files in the structured fields skill_md and meta_skill_md. Change only what is needed: preserve all unchanged text verbatim and do not rewrite, reorganize, or broadly rephrase either file. Keep necessary changes short and reusable; never encode question-specific answers.",
            model=self.model,
            model_settings=self.settings,
            output_type=SkillUpdate,
        )
        self.cumulative_cost = 0.0
        for path in list(RUN_ROOT.glob("train/question-*/rollout-*/result.json")) + list(RUN_ROOT.glob("train/question-*/summary.json")) + list(RUN_ROOT.glob("batches/batch-*/updater.json")):
            self.cumulative_cost += json.loads(path.read_text(encoding="utf-8"))["cost_usd"]

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        wandb_id_path = RUN_ROOT / "wandb_run_id.txt"
        if wandb_id_path.exists():
            wandb_id = wandb_id_path.read_text(encoding="utf-8").strip()
            self.wandb_run = wandb.init(project="video-sql-v2", id=wandb_id, resume="must", name="pilot-luna-seed2026")
        else:
            wandb_id = generate_id()
            self.wandb_run = wandb.init(project="video-sql-v2", id=wandb_id, resume="never", name="pilot-luna-seed2026")
            wandb_id_path.write_text(wandb_id + "\n", encoding="utf-8")

    async def run_rollout(self, question: dict, rollout_index: int, skill: str, meta_skill: str) -> dict:
        question_id = int(question["index"])
        rollout_dir = RUN_ROOT / "train" / f"question-{question_id}" / f"rollout-{rollout_index}"
        result_path = rollout_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["new_result"] = False
            return payload
        rollout_dir.mkdir(parents=True, exist_ok=True)

        rollout = None
        for existing_trace_path in sorted(rollout_dir.glob("attempt-*/trace.json"), reverse=True):
            existing_trace = json.loads(existing_trace_path.read_text(encoding="utf-8"))
            if "status" not in existing_trace:
                continue
            if existing_trace["status"] == "answered":
                rollout = RolloutResult(status="answered", answer=existing_trace["answer"], score=None, reflection=None, compactions=len(existing_trace["compactions"]), trace_path=str(existing_trace_path))
            elif existing_trace["status"] == "compaction_limit":
                rollout = RolloutResult(status="compaction_limit", answer=None, score=0, reflection=existing_trace["reflection"], compactions=len(existing_trace["compactions"]), trace_path=str(existing_trace_path))
            else:
                raise ValueError(f"unknown saved rollout status: {existing_trace['status']}")
            break
        if rollout is None:
            attempt_dirs = [path for path in rollout_dir.glob("attempt-*") if (path / "trace.json").exists() or (path / "error.json").exists()]
            next_attempt = max(int(path.name.removeprefix("attempt-")) for path in attempt_dirs) + 1 if attempt_dirs else 1
            if next_attempt > 3:
                raise RuntimeError(f"rollout retries exhausted before resume question_id={question_id} rollout={rollout_index}")
            for attempt in range(next_attempt, 4):
                attempt_dir = rollout_dir / f"attempt-{attempt}"
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
                    rollout = await runtime.run_rollout(question["question"], skill, attempt_dir / "trace.json", True)
                    break
                except RETRYABLE_ERRORS as error:
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    (attempt_dir / "error.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"rollout_retry question_id={question_id} rollout={rollout_index} attempt={attempt} error={type(error).__name__}: {error}")
                    if attempt == 3:
                        raise

        trace = json.loads(Path(rollout.trace_path).read_text(encoding="utf-8"))
        judge_responses = []
        reflection_responses = []
        if rollout.status == "compaction_limit":
            score = 0
            judgment = None
            reflection = rollout.reflection
        else:
            judge_input = json.dumps({"reference_answer": question["answer"], "candidate_answer": rollout.answer}, ensure_ascii=False)
            judge_result = await run_structured_agent(self.judge, judge_input, "pilot-judge", f"question-{question_id}-rollout-{rollout_index}", judge_responses)
            judgment = judge_result.final_output.model_dump(mode="json")
            score = judgment["score"]
            reflection_agent = Agent(
                name="pilot rollout reflection",
                instructions="Produce a reflection and a meta-reflection from the complete rollout trace, reference answer, and judge result. Follow the supplied META-SKILL. Identify the retrieval decision that caused success or failure and a reusable correction without leaking the specific answer.\n\nMETA-SKILL:\n" + meta_skill,
                model=self.model,
                model_settings=self.settings,
                output_type=Reflection,
            )
            reflection_input = json.dumps({"question": question["question"], "reference_answer": question["answer"], "candidate_answer": rollout.answer, "judgment": judgment, "rollout_trace": trace_for_reflection(trace)}, ensure_ascii=False)
            reflection_result = await run_structured_agent(reflection_agent, reflection_input, "pilot-reflection", f"question-{question_id}-rollout-{rollout_index}", reflection_responses)
            reflection = reflection_result.final_output.model_dump(mode="json")
        responses = []
        for attempt_trace_path in sorted(rollout_dir.glob("attempt-*/trace.json")):
            attempt_trace = json.loads(attempt_trace_path.read_text(encoding="utf-8"))
            responses.extend(attempt_trace["model_responses"])
            for nested_trace in attempt_trace["nested_tool_traces"]:
                responses.extend(nested_trace["responses"])
            for compaction in attempt_trace["compactions"]:
                responses.extend(compaction["responses"])
            if "reflection_responses" in attempt_trace:
                responses.extend(attempt_trace["reflection_responses"])
        responses.extend(judge_responses)
        responses.extend(reflection_responses)
        usage, cost = usage_and_cost(responses)
        result = {
            "question_id": question_id,
            "rollout_index": rollout_index,
            "status": rollout.status,
            "compactions": rollout.compactions,
            "answer": rollout.answer,
            "score": score,
            "judgment": judgment,
            "reflection": reflection,
            "trace_path": rollout.trace_path,
            "judge_responses": judge_responses,
            "reflection_responses": reflection_responses,
            "usage": usage,
            "cost_usd": cost,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["new_result"] = True
        return result

    async def run(self) -> None:
        split_dir = RUN_ROOT / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        train, val = graph_partition(PILOT_SOURCE / "train.json", PILOT_SOURCE / "val.json")
        train_path = split_dir / "train.json"
        val_path = split_dir / "val.json"
        serialized_train = json.dumps(train, ensure_ascii=False, indent=2)
        serialized_val = json.dumps(val, ensure_ascii=False, indent=2)
        if train_path.exists() and train_path.read_text(encoding="utf-8") != serialized_train:
            raise ValueError("existing Pilot train split differs from fixed graph partition")
        if val_path.exists() and val_path.read_text(encoding="utf-8") != serialized_val:
            raise ValueError("existing Pilot val split differs from fixed graph partition")
        train_path.write_text(serialized_train, encoding="utf-8")
        val_path.write_text(serialized_val, encoding="utf-8")

        question_by_id = {int(question["index"]): question for question in train}
        order = list(question_by_id)
        random.Random(SEED).shuffle(order)
        order_path = RUN_ROOT / "question_order.json"
        serialized_order = json.dumps(order, indent=2)
        if order_path.exists() and order_path.read_text(encoding="utf-8") != serialized_order:
            raise ValueError("existing Pilot question order differs from seed 2026 order")
        order_path.write_text(serialized_order, encoding="utf-8")

        skill_dir = RUN_ROOT / "skills"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        meta_skill_path = skill_dir / "META-SKILL.md"
        if not skill_path.exists():
            shutil.copyfile(ROOT / "skills" / "initial" / "SKILL.md", skill_path)
        if not meta_skill_path.exists():
            shutil.copyfile(ROOT / "skills" / "initial" / "META-SKILL.md", meta_skill_path)

        for batch_start in range(0, len(order), BATCH_QUESTIONS):
            batch_index = batch_start // BATCH_QUESTIONS
            batch_ids = order[batch_start:batch_start + BATCH_QUESTIONS]
            batch_dir = RUN_ROOT / "batches" / f"batch-{batch_index:03d}"
            snapshot_skill = batch_dir / "SKILL.md"
            snapshot_meta_skill = batch_dir / "META-SKILL.md"
            updater_path = batch_dir / "updater.json"
            if snapshot_skill.exists() and snapshot_meta_skill.exists():
                if skill_path.read_text(encoding="utf-8") != snapshot_skill.read_text(encoding="utf-8") or meta_skill_path.read_text(encoding="utf-8") != snapshot_meta_skill.read_text(encoding="utf-8"):
                    shutil.copyfile(snapshot_skill, skill_path)
                    shutil.copyfile(snapshot_meta_skill, meta_skill_path)
                continue
            if updater_path.exists():
                updater_payload = json.loads(updater_path.read_text(encoding="utf-8"))
                skill_path.write_text(updater_payload["skill_md"], encoding="utf-8")
                meta_skill_path.write_text(updater_payload["meta_skill_md"], encoding="utf-8")
                batch_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(skill_path, snapshot_skill)
                shutil.copyfile(meta_skill_path, snapshot_meta_skill)
                recovered_batch_scores = [json.loads((RUN_ROOT / "train" / f"question-{question_id}" / f"rollout-{rollout_index}" / "result.json").read_text(encoding="utf-8"))["score"] for question_id in batch_ids for rollout_index in range(ROLLOUTS_PER_QUESTION)]
                self.wandb_run.log({"pilot/batch_mean_score": sum(recovered_batch_scores) / len(recovered_batch_scores), "pilot/batch_index": batch_index, "pilot/cumulative_cost_usd": self.cumulative_cost})
                if self.cumulative_cost > PILOT_COST_LIMIT_USD:
                    print(f"pilot_cost_limit_reached cumulative_cost_usd={self.cumulative_cost:.6f} recovered_batch={batch_index}")
                    self.wandb_run.finish()
                    await self.client.close()
                    return
                continue
            if snapshot_skill.exists() != snapshot_meta_skill.exists():
                raise RuntimeError(f"incomplete batch skill snapshot without updater artifact batch={batch_index}")
            skill = skill_path.read_text(encoding="utf-8")
            meta_skill = meta_skill_path.read_text(encoding="utf-8")
            batch_summaries = []
            batch_scores = []
            for question_position, question_id in enumerate(batch_ids):
                question = question_by_id[question_id]
                async with asyncio.TaskGroup() as tasks:
                    rollout_tasks = [tasks.create_task(self.run_rollout(question, rollout_index, skill, meta_skill)) for rollout_index in range(ROLLOUTS_PER_QUESTION)]
                rollout_results = [task.result() for task in rollout_tasks]
                for rollout_result in rollout_results:
                    if rollout_result["new_result"]:
                        self.cumulative_cost += rollout_result["cost_usd"]
                    batch_scores.append(rollout_result["score"])
                    self.wandb_run.log({"pilot/rollout_score": rollout_result["score"], "pilot/question_id": question_id, "pilot/batch_index": batch_index, "pilot/final_status": rollout_result["status"], "pilot/compactions": rollout_result["compactions"], "pilot/cumulative_cost_usd": self.cumulative_cost, **{f"pilot/usage_{key}": value for key, value in rollout_result["usage"].items()}})
                if self.cumulative_cost > PILOT_COST_LIMIT_USD:
                    print(f"pilot_cost_limit_reached cumulative_cost_usd={self.cumulative_cost:.6f} completed_question_id={question_id}")
                    self.wandb_run.finish()
                    await self.client.close()
                    return
                question_dir = RUN_ROOT / "train" / f"question-{question_id}"
                summary_path = question_dir / "summary.json"
                if summary_path.exists():
                    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    summary_input = json.dumps({"question": question["question"], "reference_answer": question["answer"], "rollouts": [{"answer": result["answer"], "score": result["score"], "judgment": result["judgment"], "reflection": result["reflection"]} for result in rollout_results]}, ensure_ascii=False)
                    summary_responses = []
                    summary_result = await run_structured_agent(self.summarizer, summary_input, "pilot-question-summary", f"question-{question_id}", summary_responses)
                    summary_usage, summary_cost = usage_and_cost(summary_responses)
                    summary_payload = {"question_id": question_id, "summary": summary_result.final_output.summary, "responses": summary_responses, "usage": summary_usage, "cost_usd": summary_cost}
                    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.cumulative_cost += summary_cost
                batch_summaries.append(summary_payload)
                question_mean = sum(result["score"] for result in rollout_results) / len(rollout_results)
                self.wandb_run.log({"pilot/question_mean_score": question_mean, "pilot/question_id": question_id, "pilot/cumulative_cost_usd": self.cumulative_cost})
                if self.cumulative_cost > PILOT_COST_LIMIT_USD:
                    print(f"pilot_cost_limit_reached cumulative_cost_usd={self.cumulative_cost:.6f} completed_question_id={question_id}")
                    self.wandb_run.finish()
                    await self.client.close()
                    return

            batch_dir.mkdir(parents=True, exist_ok=True)
            updater_input = json.dumps({"current_skill": skill, "current_meta_skill": meta_skill, "question_summaries": [{"question_id": item["question_id"], "summary": item["summary"]} for item in batch_summaries]}, ensure_ascii=False)
            updater_responses = []
            updater_result = await run_structured_agent(self.updater, updater_input, "pilot-skill-updater", f"batch-{batch_index}", updater_responses)
            updater_usage, updater_cost = usage_and_cost(updater_responses)
            updater_payload = {"batch_index": batch_index, "question_ids": batch_ids, "skill_md": updater_result.final_output.skill_md, "meta_skill_md": updater_result.final_output.meta_skill_md, "responses": updater_responses, "usage": updater_usage, "cost_usd": updater_cost}
            updater_path.write_text(json.dumps(updater_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.cumulative_cost += updater_cost
            skill_path.write_text(updater_payload["skill_md"], encoding="utf-8")
            meta_skill_path.write_text(updater_payload["meta_skill_md"], encoding="utf-8")
            shutil.copyfile(skill_path, snapshot_skill)
            shutil.copyfile(meta_skill_path, snapshot_meta_skill)
            self.wandb_run.log({"pilot/batch_mean_score": sum(batch_scores) / len(batch_scores), "pilot/batch_index": batch_index, "pilot/cumulative_cost_usd": self.cumulative_cost})
            print(f"pilot_batch_complete batch={batch_index} questions={batch_ids} mean_score={sum(batch_scores) / len(batch_scores):.6f} cumulative_cost_usd={self.cumulative_cost:.6f}")
            if self.cumulative_cost > PILOT_COST_LIMIT_USD:
                print(f"pilot_cost_limit_reached cumulative_cost_usd={self.cumulative_cost:.6f} completed_batch={batch_index}")
                self.wandb_run.finish()
                await self.client.close()
                return

        self.wandb_run.finish()
        await self.client.close()
        print(f"pilot_training_complete questions={len(order)} cumulative_cost_usd={self.cumulative_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(PilotTrainer().run())
