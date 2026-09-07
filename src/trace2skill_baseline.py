"""One Trace2Skill Combined evolution over five independent month QA rollouts."""

import asyncio
import base64
import dataclasses
import json
import shutil
import sys
import uuid
from pathlib import Path

import weave
from agents import Agent, OpenAIResponsesModel, StopAtTools, ToolOutputImage, function_tool, trace
from pydantic import BaseModel, Field, ValidationError
from third_party.trace2skill.tracing_bridge import init_weave

from src.harness import InvalidTrace, VideoSQLHarness, run_stream, tool_error_json
from src.val import judge_answer, retryable, write_error


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/trace2skill-baseline"
UPSTREAM = ROOT / "third_party/trace2skill"
RUNTIME = Path("/raid/wg25r/video_sql_v2/data/runtime/month")
TRAIN = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443/month/train.json")
MODEL = "Qwen/Qwen3.8-27B-FP8"
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


class LimitedResponsesModel(OpenAIResponsesModel):
    def __init__(self, client, semaphore):
        super().__init__(model=MODEL, openai_client=client)
        self.semaphore = semaphore

    async def stream_response(self, *args, **kwargs):
        # Acquire per model request, so nested agents and judges share the same 128 slots.
        async with self.semaphore:
            async for event in super().stream_response(*args, **kwargs):
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


def make_harness(directory, skill_dir, semaphore):
    harness = VideoSQLHarness(
        database=RUNTIME / "index.duckdb", embedding_index=RUNTIME / "embedding_index",
        media_map=RUNTIME / "media-map.json", skill_dir=skill_dir, run_dir=directory,
        model_name=MODEL, api_key="local", base_url="http://127.0.0.1:8003/v1",
        embedding_url="http://127.0.0.1:8014/v1", embedding_model="qwen3-vl-embedding-8b",
        context_limit=314573, compaction_threshold=178000, reasoning_context=None,
    )
    harness.context.model = LimitedResponsesModel(harness.context.client, semaphore)
    harness.context.embedding_client = LimitedEmbeddingClient(harness.context.embedding_client, semaphore)
    return harness


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k != "semaphore"})
async def rollout(question, rollout_index, skill_version, semaphore):
    directory = RUN / skill_version / f"question-{question['index']}" / f"rollout-{rollout_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    skill_dir = RUN / "skills" / skill_version
    metadata = {"run": RUN.name, "question": str(question["index"]),
                "rollout": rollout_index, "skill_version": skill_version}
    with weave.attributes(metadata), trace("Trace2Skill QA rollout", metadata=metadata):
        for attempt in range(1, 4):
            attempt_dir = directory / f"attempt-{attempt}"
            if (attempt_dir / "error.json").exists():
                continue
            if attempt_dir.exists() and not (attempt_dir / "answer.json").exists():
                attempt_dir.rename(directory / f"interrupted-{attempt}-{uuid.uuid4().hex}")
            harness = make_harness(attempt_dir, skill_dir, semaphore)
            try:
                with weave.attributes({"attempt": attempt}):
                    answer_path = attempt_dir / "answer.json"
                    if answer_path.exists():
                        answer = json.loads(answer_path.read_text())
                    else:
                        session = await harness.answer(question["question"])
                        if session.status != "answered" or session.answer is None:
                            raise InvalidTrace(f"QA ended without an answer: {session.status}")
                        answer = dataclasses.asdict(session)
                        answer_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
                    score = await judge_answer(harness, question["question"], question["answer"], answer["answer"], 3)
                result = {**metadata, "attempt": attempt, "question_text": question["question"],
                          "reference_answer": question["answer"], "answer": answer["answer"],
                          "score": score, "success": score >= 4, "status": "answered",
                          "trace_files": sorted(str(p) for p in directory.rglob("traces/*.json")),
                          "media_files": sorted(str(p) for p in directory.rglob("media/*.jpg"))}
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(f"{skill_version} question={question['index']} rollout={rollout_index} score={score}", flush=True)
                return result
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                write_error(attempt_dir / "error.json", error)
                if not retryable(error):
                    raise
            finally:
                await harness.close()
    raise RuntimeError(f"All three QA attempts exhausted: {directory}")


@weave.op(postprocess_inputs=lambda x: {"question": x["result"]["question"], "rollout": x["result"]["rollout"], "success": x["result"]["success"]})
async def analyze(result, semaphore):
    directory = RUN / "analysis" / f"question-{result['question']}" / f"rollout-{result['rollout']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "record.json"
    if output_path.exists():
        return json.loads(output_path.read_text())
    # Canonical histories contain every model/tool input and returned value. Raw events,
    # usage, duplicate API representations and images remain in the original artifacts.
    histories = []
    for filename in result["trace_files"]:
        artifact = json.loads(Path(filename).read_text())
        if artifact["agent"]["name"] == "judge":
            continue
        histories.append({"file": filename, "agent": artifact["agent"],
                          "history": artifact["history"], "error": artifact["error"]})
    bundle = {"question": result["question_text"], "answer": result["answer"],
              "score": result["score"], "histories": histories}
    (directory / "analysis_input.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    harness = make_harness(directory, RUN / "skills/initial", semaphore)
    metadata = {"run": RUN.name, "question": result["question"], "rollout": result["rollout"], "skill_version": "initial"}
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
                        async with semaphore:
                            raw = await harness.context.client.responses.with_raw_response.parse(
                                model=MODEL, instructions=instructions, input=messages,
                                reasoning={"effort": "medium"}, store=False,
                                text_format=SuccessReport,
                            )
                        body = raw.content
                        (directory / f"success-response-{attempt}.json").write_bytes(body)
                        messages.extend(json.loads(body)["output"])
                        response = raw.parse()
                        if response.status != "completed" or response.output_parsed is None:
                            raise RuntimeError(f"Incomplete success analysis: {response.model_dump_json()}")
                        report = response.output_parsed
                        break
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        write_error(directory / f"success-attempt-{attempt}-error.json", error)
                        if attempt == 3 or not retryable(error):
                            raise
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
                    judge_harness = make_harness(directory / f"fix-{fix_id}", RUN / "skills/initial", semaphore)
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
                        if attempt == 3 or not retryable(error):
                            raise
                (directory / "failure_report.json").write_text(report.model_dump_json(indent=2) + "\n")
                groups = {"failure_cause": report.failure_cause, "failure_memory": report.failure_memory}
    finally:
        await harness.close()
    items = [{"type": kind, "number": i, **item.model_dump()} for kind, group in groups.items() for i, item in enumerate(group, 1)]
    record = {"record_source": "success" if result["success"] else "error",
              "instance_id": f"month-{result['question']}-rollout-{result['rollout']}",
              "source_file": str(directory / "analysis_input.json"), "items": items}
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"analysis {record['instance_id']} {record['record_source']}", flush=True)
    return record


async def main():
    RUN.mkdir(parents=True, exist_ok=True)
    init_weave()
    initial = RUN / "skills/initial"
    if not initial.exists():
        shutil.copytree(ROOT / "skills/initial", initial)
    # Refuse a changed seed on resume; each source file, including the root guide, is byte-identical.
    for source in (ROOT / "skills/initial").glob("*.md"):
        if (initial / source.name).read_bytes() != source.read_bytes():
            raise RuntimeError(f"Initial skill changed: {source.name}")
    questions = json.loads(TRAIN.read_text())
    semaphore = asyncio.Semaphore(128)
    # A separate scheduling bound avoids loading thousands of simultaneous harness contexts.
    for version in ("initial", "evolved"):
        jobs = iter((question, index) for question in questions for index in range(5))
        results = []

        async def collect_worker():
            for question, index in jobs:
                with weave.attributes({"run": RUN.name, "question": str(question["index"]), "rollout": index, "skill_version": version}):
                    results.append(await rollout(question, index, version, semaphore))

        async with asyncio.TaskGroup() as group:
            for _ in range(128):
                group.create_task(collect_worker())
        results.sort(key=lambda r: (int(r["question"]), r["rollout"]))
        summary = {"questions": len(questions), "rollouts": len(results),
                   "mean_score_over_5": sum(r["score"] / 5 for r in results) / len(results),
                   "successes": sum(r["success"] for r in results)}
        (RUN / f"{version}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if version == "evolved":
            break
        analysis_jobs = iter(results)
        records = []

        async def analysis_worker():
            for result in analysis_jobs:
                records.append(await analyze(result, semaphore))

        async with asyncio.TaskGroup() as group:
            for _ in range(128):
                group.create_task(analysis_worker())
        records.sort(key=lambda r: r["instance_id"])
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
    print("TRACE2SKILL BASELINE COMPLETE " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
