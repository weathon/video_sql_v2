"""One Trace2Skill Combined evolution over fresh day/week subset QA rollouts."""

import asyncio
import base64
import dataclasses
import json
import random
import shutil
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import weave
import wandb
from agents import Agent, ModelSettings, OpenAIResponsesModel, StopAtTools, ToolOutputImage, function_tool, trace
from pydantic import BaseModel, Field, ValidationError
from third_party.trace2skill.tracing_bridge import init_weave

from src.harness import InvalidTrace, VideoSQLHarness, run_stream, tool_error_json
from src.val import JudgeAttemptsExhausted, judge_answer, retryable, write_error


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/trace2skill-baseline-nvfp4-day-week-seed0"
UPSTREAM = ROOT / "third_party/trace2skill"
RUNTIME = Path("/raid/wg25r/video_sql_v2/data/runtime")
DATASET = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443")
MODEL = "Inferact/Qwen3.8-Flash-Next-NVFP4"
BASE_URL = "http://127.0.0.1:8003/v1"
API_KEY = "EMPTY"
SKILL_HEADER = "---\nname: video-memory-qa\ndescription: Retrieve and verify evidence to answer questions about long videos.\n---\n\n"


class MemoryItem(BaseModel):
    title: str
    description: str
    content: str


class SuccessReport(BaseModel):
    lean_solution_path: str
    success_memory: list[MemoryItem] = Field(max_length=3)


class FailureReport(BaseModel):
    failure_cause: list[MemoryItem] = Field(min_length=1)
    failure_memory: list[MemoryItem] = Field(max_length=3)


class RunMetrics:
    def __init__(self, run, phase):
        self.run = run
        self.phase = phase
        self.started = time.monotonic()
        self.responses = deque()
        self.trace_lengths = []
        self.rollout_times = []
        self.replacement_metrics = []

    def record_response(self, usage):
        now = time.monotonic()
        self.responses.append((now, usage["input_tokens"], usage["output_tokens"]))
        while self.responses[0][0] < now - 60:
            self.responses.popleft()
        elapsed = min(now - self.started, 60)
        # Completion-accounted throughput over the last 60 seconds, including nested
        # agents, retries and judge; input counts include tokens served from cache.
        self.run.log({
            f"{self.phase}/throughput/input_tokens_per_s": sum(r[1] for r in self.responses) / elapsed,
            f"{self.phase}/throughput/output_tokens_per_s": sum(r[2] for r in self.responses) / elapsed,
            f"{self.phase}/throughput/responses_per_s": len(self.responses) / elapsed,
            f"{self.phase}/request_context_tokens": usage["input_tokens"] + usage["output_tokens"],
        })

    def record_rollout(self, result):
        self.trace_lengths.append(result["trace_length"])
        self.rollout_times.append(result["rollout_time_seconds"])
        self.replacement_metrics.append(result["replacement_metrics"])
        count = len(self.trace_lengths)
        values = {f"{self.phase}/completed_rollouts": count,
                  f"{self.phase}/average_rollout_time_seconds": sum(self.rollout_times) / count,
                  f"{self.phase}/mean_total_context_saved_tokens": sum(r["total_context_saved_tokens"] for r in self.replacement_metrics) / count,
                  f"{self.phase}/replace_tool_calls": sum(r["replace_tool_calls"] for r in self.replacement_metrics)}
        for key in result["trace_length"]:
            values[f"{self.phase}/trace/mean_{key}"] = sum(r[key] for r in self.trace_lengths) / count
        self.run.log(values)
        (RUN / f"{self.phase}-trace-metrics.json").write_text(json.dumps(values, indent=2) + "\n")


class LimitedResponsesModel(OpenAIResponsesModel):
    def __init__(self, client, semaphore, metrics):
        super().__init__(model=MODEL, openai_client=client)
        self.semaphore = semaphore
        self.metrics = metrics

    async def stream_response(self, *args, **kwargs):
        # Acquire per model request, so nested agents and judges share the same 256 slots.
        async with self.semaphore:
            async for event in super().stream_response(*args, **kwargs):
                if event.type == "response.completed":
                    self.metrics.record_response(event.response.usage.model_dump())
                yield event


class LimitedEmbeddingClient:
    def __init__(self, client, semaphore):
        self.client = client
        self.semaphore = semaphore

    async def post(self, *args, **kwargs):
        async with self.semaphore:
            return await self.client.post(*args, **kwargs)

    async def aclose(self):
        await self.client.aclose()


def make_harness(directory, skill_dir, semaphore, metrics, timeline):
    harness = VideoSQLHarness(
        database=RUNTIME / timeline / "index.duckdb", embedding_index=RUNTIME / timeline / "embedding_index",
        media_map=RUNTIME / timeline / "media-map.json", skill_dir=skill_dir, run_dir=directory,
        model_name=MODEL, api_key=API_KEY, base_url=BASE_URL,
        embedding_url="http://127.0.0.1:8014/v1", embedding_model="qwen3-vl-embedding-8b",
        context_limit=314573, compaction_threshold=178000,
    )
    harness.context.model_settings = ModelSettings(
        reasoning={"effort": "medium"}, store=False, preserve_raw_usage=True,
    )
    harness.context.model = LimitedResponsesModel(harness.context.client, semaphore, metrics)
    harness.context.embedding_client = LimitedEmbeddingClient(harness.context.embedding_client, semaphore)
    return harness


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k not in ("semaphore", "metrics")})
async def rollout(question, rollout_index, skill_version, semaphore, metrics):
    directory = RUN / skill_version / question["timeline"] / f"question-{question['index']}" / f"rollout-{rollout_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / "skipped.json").exists():
        print(f"SKIP saved exhausted rollout: {directory}", flush=True)
        return None
    skill_dir = RUN / "skills" / skill_version
    metadata = {"run": RUN.name, "timeline": question["timeline"], "question": str(question["index"]),
                "rollout": rollout_index, "skill_version": skill_version}
    with weave.attributes(metadata), trace("Trace2Skill QA rollout", metadata=metadata):
        for attempt in range(1, 4):
            attempt_dir = directory / f"attempt-{attempt}"
            if (attempt_dir / "error.json").exists():
                continue
            if attempt_dir.exists() and not (attempt_dir / "answer.json").exists():
                attempt_dir.rename(directory / f"interrupted-{attempt}-{uuid.uuid4().hex}")
            harness = make_harness(attempt_dir, skill_dir, semaphore, metrics, question["timeline"])
            try:
                with weave.attributes({"attempt": attempt}):
                    answer_path = attempt_dir / "answer.json"
                    if answer_path.exists():
                        answer = json.loads(answer_path.read_text())
                    else:
                        started = time.monotonic()
                        try:
                            session = await harness.answer(question["question"])
                        finally:
                            (attempt_dir / "rollout-time.json").write_text(json.dumps({
                                "seconds": time.monotonic() - started,
                            }) + "\n")
                        if session.status != "answered" or session.answer is None:
                            raise InvalidTrace(f"QA ended without an answer: {session.status}")
                        answer = dataclasses.asdict(session)
                        answer_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                    score = await judge_answer(harness, question["question"], question["answer"], answer["answer"], 3)
                result = {**metadata, "attempt": attempt, "question_text": question["question"],
                          "reference_answer": question["answer"], "answer": answer["answer"],
                          "score": score, "success": score >= 4, "status": "answered",
                          "rollout_time_seconds": sum(json.loads(p.read_text())["seconds"]
                                                      for p in directory.glob("*/rollout-time.json")),
                          "trace_files": sorted(str(p) for p in directory.rglob("traces/*.json")),
                          "media_files": sorted(str(p) for p in directory.rglob("media/*.jpg"))}
                responses = []
                tool_count = 0
                replace_count = 0
                replacement_call_ids = set()
                for filename in result["trace_files"]:
                    artifact = json.loads(Path(filename).read_text())
                    if artifact["agent"]["name"] != "judge":
                        responses.extend(artifact["raw_responses"])
                        tool_count += len(artifact["tool_calls"])
                        replace_count += sum(call["name"] == "replace_tool" for call in artifact["tool_calls"])
                        replacement_call_ids.update(call["call_id"] for call in artifact["tool_calls"] if call["name"] == "replace_tool")
                total_context_saved = 0
                for path in sorted(directory.glob("*/replace-*.json")):
                    measurement = json.loads(path.read_text())
                    if measurement["replace_call_id"] not in replacement_call_ids:
                        continue
                    for side in ("before", "after"):
                        if f"{side}_tokens" in measurement:
                            continue
                        print(f"count replacement tokens side={side} artifact={path}", flush=True)
                        async with semaphore:
                            token_count = await harness.context.client.responses.input_tokens.count(
                                model=MODEL,
                                input=[{"role": "user", "content": question["question"]}, *measurement[side]],
                                reasoning=measurement["reasoning"],
                            )
                        measurement[f"{side}_tokens"] = token_count.input_tokens
                        path.write_text(json.dumps(measurement, ensure_ascii=False) + "\n")
                    total_context_saved += measurement["before_tokens"] - measurement["after_tokens"]
                result["replacement_metrics"] = {
                    "total_context_saved_tokens": total_context_saved,
                    "replace_tool_calls": replace_count,
                }
                result["trace_length"] = {
                    "generated_tokens": sum(r["usage"]["output_tokens"] for r in responses),
                    "model_calls": len(responses), "tool_calls": tool_count,
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in responses) / len(responses),
                    "peak_context_tokens": max(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in responses),
                }
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(f"{skill_version} timeline={question['timeline']} question={question['index']} rollout={rollout_index} score={score}", flush=True)
                return result
            except asyncio.CancelledError:
                raise
            except JudgeAttemptsExhausted as error:
                write_error(attempt_dir / "error.json", error)
                break
            except BaseException as error:
                write_error(attempt_dir / "error.json", error)
                if not retryable(error):
                    raise
            finally:
                await harness.close()
    (directory / "skipped.json").write_text(json.dumps({
        **metadata, "status": "skipped", "reason": "qa_or_judge_attempts_exhausted",
        "trace_files": sorted(str(p) for p in directory.rglob("traces/*.json")),
        "media_files": sorted(str(p) for p in directory.rglob("media/*.jpg")),
    }, indent=2) + "\n")
    print(f"SKIP QA/judge attempts exhausted: {directory}", flush=True)
    return None


@weave.op(postprocess_inputs=lambda x: {"timeline": x["result"]["timeline"], "question": x["result"]["question"], "rollout": x["result"]["rollout"], "success": x["result"]["success"]})
async def analyze(result, semaphore, metrics):
    directory = RUN / "analysis" / result["timeline"] / f"question-{result['question']}" / f"rollout-{result['rollout']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "record.json"
    if output_path.exists():
        return json.loads(output_path.read_text())
    if (directory / "skipped.json").exists():
        print(f"SKIP saved exhausted analysis: {directory}", flush=True)
        return None
    # Canonical histories contain every model/tool input and returned value. Raw events,
    # usage, duplicate API representations and images remain in the original artifacts.
    histories = []
    for filename in result["trace_files"]:
        artifact = json.loads(Path(filename).read_text())
        if artifact["agent"]["name"] == "judge":
            continue
        # Analysis reads summaries; encrypted state remains in the replay artifacts.
        histories.append({"file": filename, "agent": artifact["agent"],
                          "history": [
                              {key: value for key, value in item.items() if key != "encrypted_content"}
                              if item.get("type") == "reasoning" else item
                              for item in artifact["history"]
                          ],
                          "tool_calls": [{k: v for k, v in call.items() if k != "original_output"}
                                         for call in artifact["tool_calls"]],
                          "error": artifact["error"]})
    bundle = {"question": result["question_text"], "answer": result["answer"],
              "score": result["score"], "histories": histories}
    (directory / "analysis_input.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    harness = make_harness(directory, RUN / "skills/initial", semaphore, metrics, result["timeline"])
    metadata = {"run": RUN.name, "timeline": result["timeline"], "question": result["question"], "rollout": result["rollout"], "skill_version": "initial"}
    try:
        with weave.attributes(metadata), trace("Trace2Skill trajectory analysis", metadata=metadata):
            if result["success"]:
                instructions = (UPSTREAM / "analysis/success_analysis_system_llm.txt").read_text()
                instructions = instructions.replace("spreadsheet manipulation tasks", "video question answering tasks").replace("`input.xlsx`", "video evidence, memory database")
                instructions += "\nReturn the Lean Solution Path and at most three Success Memory Items using the supplied JSON schema."
                content = [{"type": "input_text", "text": json.dumps(bundle, ensure_ascii=False)}]
                for filename in result["media_files"]:
                    content.extend([{"type": "input_text", "text": filename},
                                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(Path(filename).read_bytes()).decode(), "detail": "high"}])
                messages = [{"role": "user", "content": content}]
                for attempt in range(1, 4):
                    try:
                        (directory / f"success-request-{attempt}.json").write_text(json.dumps(messages, ensure_ascii=False) + "\n")
                        async with semaphore:
                            raw = await harness.context.client.responses.with_raw_response.parse(
                                model=MODEL, instructions=instructions, input=messages,
                                reasoning={"effort": "medium"}, store=False,
                                text_format=SuccessReport,
                            )
                        body = raw.content
                        (directory / f"success-response-{attempt}.json").write_bytes(body)
                        payload = json.loads(body)
                        metrics.record_response(payload["usage"])
                        messages.extend(payload["output"])
                        response = raw.parse()
                        if response.status != "completed" or response.output_parsed is None:
                            raise RuntimeError(f"Incomplete success analysis: {response.model_dump_json()}")
                        report = response.output_parsed
                        break
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        write_error(directory / f"success-attempt-{attempt}-error.json", error)
                        if not retryable(error):
                            raise
                        if attempt == 3:
                            (directory / "skipped.json").write_text(json.dumps({
                                **metadata, "status": "skipped", "reason": "success_analysis_attempts_exhausted",
                                "source_file": str(directory / "analysis_input.json"),
                            }, indent=2) + "\n")
                            print(f"SKIP success analysis attempts exhausted: {directory}", flush=True)
                            return None
                        if isinstance(error, ValidationError):
                            messages.append({"role": "user", "content": f"Correct the structured output. Validation error: {error}"})
                (directory / "success_report.json").write_text(report.model_dump_json(indent=2) + "\n")
                groups = {"success_memory": report.success_memory}
            else:
                validated_fix = None
                submitted_report = None

                @function_tool(failure_error_function=tool_error_json)
                async def read_trace(file: str, start_line: int, end_line: int) -> str:
                    """Read an inclusive line range from a listed execution artifact; inspect all relevant tool inputs and results."""
                    if file not in result["trace_files"]:
                        raise ValueError("Choose a file from trace_files in the task input")
                    lines = Path(file).read_text().splitlines()
                    if not 1 <= start_line <= end_line <= len(lines):
                        raise ValueError(f"Valid line range is 1..{len(lines)}")
                    return "\n".join(lines[start_line - 1:end_line])

                @function_tool(failure_error_function=tool_error_json)
                async def evaluate_output(answer: str, minimal_fix: str, evidence: str) -> str:
                    """Validate the corrected answer and minimal reasoning/tool change using the unchanged 0–5 judge."""
                    nonlocal validated_fix
                    fix_id = uuid.uuid4().hex
                    judge_harness = make_harness(directory / f"fix-{fix_id}", RUN / "skills/initial", semaphore, metrics, result["timeline"])
                    try:
                        score = await judge_answer(judge_harness, result["question_text"], result["reference_answer"], answer, 3)
                    finally:
                        await judge_harness.close()
                    fix = {"answer": answer, "minimal_fix": minimal_fix, "evidence": evidence, "score": score, "pass": score >= 4}
                    path = directory / f"validated-fix-{fix_id}.json"
                    path.write_text(json.dumps(fix, ensure_ascii=False, indent=2) + "\n")
                    validated_fix = fix if score >= 4 else None
                    return json.dumps(fix, ensure_ascii=False)

                @function_tool(failure_error_function=tool_error_json)
                async def submit_failure_analysis(report: FailureReport) -> str:
                    """Submit causal failure items and at most three reusable memory items after validating the minimal fix."""
                    nonlocal submitted_report
                    if validated_fix is None:
                        raise ValueError("Validate a minimal fix with evaluate_output before submitting analysis")
                    submitted_report = report
                    return report.model_dump_json()

                @function_tool(failure_error_function=tool_error_json)
                async def read_frame(file: str) -> list[ToolOutputImage]:
                    """Inspect an original rollout frame from media_files, including its actual pixels."""
                    if file not in result["media_files"]:
                        raise ValueError("Choose a file from media_files in the task input")
                    return [ToolOutputImage(image_url="data:image/jpeg;base64," + base64.b64encode(Path(file).read_bytes()).decode(), detail="high")]

                instructions = (ROOT / "baseline/error_analysis_qa.txt").read_text()
                agent = Agent(
                    name="trace2skill-error-analysis", instructions=instructions,
                    model=harness.context.model, model_settings=harness.context.model_settings,
                    tools=[tool for tool in harness.answer_tools if tool.name != "submit_answer"]
                          + [read_trace, read_frame, evaluate_output, submit_failure_analysis],
                    tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_failure_analysis"]),
                )
                task = {**result, "trace_line_counts": {p: len(Path(p).read_text().splitlines()) for p in result["trace_files"]}}
                for attempt in range(1, 4):
                    try:
                        validated_fix = submitted_report = None
                        run, _, _ = await run_stream(harness.context, agent, json.dumps(task, ensure_ascii=False), f"error-analysis-{attempt}", False)
                        report = FailureReport.model_validate_json(str(run.final_output))
                        if submitted_report is None or report != submitted_report or validated_fix is None:
                            raise InvalidTrace("Failure analysis ended without a validated submit_failure_analysis call")
                        break
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        write_error(directory / f"analysis-attempt-{attempt}-error.json", error)
                        if not retryable(error):
                            raise
                        if attempt == 3:
                            (directory / "skipped.json").write_text(json.dumps({
                                **metadata, "status": "skipped", "reason": "failure_analysis_attempts_exhausted",
                                "source_file": str(directory / "analysis_input.json"),
                            }, indent=2) + "\n")
                            print(f"SKIP failure analysis attempts exhausted: {directory}", flush=True)
                            return None
                (directory / "failure_report.json").write_text(report.model_dump_json(indent=2) + "\n")
                groups = {"failure_cause": report.failure_cause, "failure_memory": report.failure_memory}
    finally:
        await harness.close()
    items = [{"type": kind, "number": i, **item.model_dump()} for kind, group in groups.items() for i, item in enumerate(group, 1)]
    record = {"record_source": "success" if result["success"] else "error",
              "instance_id": f"{result['timeline']}-{result['question']}-rollout-{result['rollout']}",
              "source_file": str(directory / "analysis_input.json"), "items": items}
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"analysis {record['instance_id']} {record['record_source']}", flush=True)
    return record


async def main():
    RUN.mkdir(parents=True, exist_ok=True)
    init_weave()
    wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name,
                           name=RUN.name, resume="allow", dir=str(RUN),
                           config={"model": MODEL, "reasoning": "medium", "base_url": BASE_URL,
                                   "api": "responses", "reasoning_replay": True,
                                   "client_concurrency": 256, "replace_tool": True,
                                   "split": "test", "timelines": ["day", "week"], "questions_per_timeline": 100, "seed": 0,
                                   "rollouts_per_question": 5})
    (RUN / "wandb-run.json").write_text(json.dumps({"id": wandb_run.id, "url": wandb_run.url}, indent=2) + "\n")
    initial = RUN / "skills/initial"
    if not initial.exists():
        shutil.copytree(ROOT / "skills/initial", initial)
    # Refuse a changed seed on resume; each source file, including the root guide, is byte-identical.
    for source in (ROOT / "skills/initial").glob("*.md"):
        if (initial / source.name).read_bytes() != source.read_bytes():
            raise RuntimeError(f"Initial skill changed: {source.name}")
    selection = RUN / "questions.json"
    if selection.exists():
        questions = json.loads(selection.read_text())
    else:
        questions = []
        for timeline in ("day", "week"):
            subset = random.Random(0).sample(json.loads((DATASET / timeline / "test.json").read_text()), 100)
            questions.extend({**question, "timeline": timeline} for question in subset)
        selection.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
    semaphore = asyncio.Semaphore(256)
    # A separate scheduling bound avoids loading thousands of simultaneous harness contexts.
    for version in ("initial", "evolved"):
        metrics = RunMetrics(wandb_run, version)
        jobs = iter((question, index) for question in questions for index in range(5))
        results = []
        skipped = 0

        async def collect_worker():
            nonlocal skipped
            for question, index in jobs:
                saved = RUN / version / question["timeline"] / f"question-{question['index']}" / f"rollout-{index:02d}" / "result.json"
                if saved.exists():
                    results.append(json.loads(saved.read_text()))
                    metrics.record_rollout(results[-1])
                    continue
                with weave.attributes({"run": RUN.name, "timeline": question["timeline"], "question": str(question["index"]), "rollout": index, "skill_version": version}):
                    result = await rollout(question, index, version, semaphore, metrics)
                    if result is None:
                        skipped += 1
                        continue
                    results.append(result)
                    metrics.record_rollout(result)

        async with asyncio.TaskGroup() as group:
            for _ in range(256):
                group.create_task(collect_worker())
        results.sort(key=lambda r: (r["timeline"], int(r["question"]), r["rollout"]))
        summary = {"questions": len(questions), "rollouts": len(results), "skipped_rollouts": skipped,
                   "mean_score_over_5": sum(r["score"] / 5 for r in results) / len(results),
                   "successes": sum(r["success"] for r in results)}
        (RUN / f"{version}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if version == "evolved":
            break
        analysis_jobs = iter(results)
        metrics = RunMetrics(wandb_run, "analysis")
        records = []
        analysis_skipped = 0

        async def analysis_worker():
            nonlocal analysis_skipped
            for result in analysis_jobs:
                record = await analyze(result, semaphore, metrics)
                if record is None:
                    analysis_skipped += 1
                    continue
                records.append(record)

        async with asyncio.TaskGroup() as group:
            for _ in range(256):
                group.create_task(analysis_worker())
        records.sort(key=lambda r: r["instance_id"])
        summary["analysis_skipped_rollouts"] = analysis_skipped
        (RUN / f"{version}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (RUN / "combined-records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        evolved = RUN / "skills/evolved"
        if not evolved.exists():
            weave.finish()
            with (RUN / "evolution.log").open("a") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(UPSTREAM / "baseline_evolve.py"), str(RUN),
                    cwd=UPSTREAM, stdout=log, stderr=asyncio.subprocess.STDOUT,
                )
                status = await process.wait()
            if status != 0:
                raise RuntimeError(f"Trace2Skill evolution exited {status}; see {RUN / 'evolution.log'}")
            init_weave()
    comparison = {version: json.loads((RUN / f"{version}-summary.json").read_text()) for version in ("initial", "evolved")}
    (RUN / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    weave.finish()
    wandb_run.finish()
    print("TRACE2SKILL BASELINE COMPLETE " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
