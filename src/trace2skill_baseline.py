"""One Trace2Skill Combined evolution over fresh day/week subset QA rollouts."""

import asyncio
import base64
import copy
import dataclasses
import heapq
import itertools
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from contextvars import ContextVar

import weave
import wandb
from agents import Agent, OpenAIResponsesModel, StopAtTools, ToolOutputImage, ToolOutputText, function_tool, trace
from dotenv import dotenv_values
from pydantic import BaseModel, Field, ValidationError
from third_party.trace2skill.tracing_bridge import RunMetrics, init_weave, response_cost

from src.harness import OPENAI_BASE_URL, REQUEST_TRACE, InvalidTrace, VideoSQLHarness, frame_jpeg_bytes, run_stream, stripped_json, tool_error_json
from src.val import JudgeAttemptsExhausted, judge_answer, retryable, write_error


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/luna-day-week-tree-month-val-seed0"
UPSTREAM = ROOT / "third_party/trace2skill"
RUNTIME = Path("/raid/wg25r/video_sql_v2/data/runtime")
DATASET = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443")
MODEL = "gpt-5.6-luna"
BASE_URL = OPENAI_BASE_URL
API_KEY = dotenv_values(ROOT / ".env")["PERSONAL_OPENAI_KEY"]
EMBEDDING_URL = dotenv_values(ROOT / ".env")["EMBEDDING_URL"]
SKILL_HEADER = "---\nname: video-memory-qa\ndescription: Retrieve and verify evidence to answer questions about long videos.\n---\n\n"
REQUEST_PRIORITY = ContextVar("request_priority", default=2)


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


class RequestSlots:
    def __init__(self, run):
        self.condition = asyncio.Condition()
        self.waiters = []
        self.order = itertools.count()
        self.run = run
        self.active = 0
        self.peak = 0

    async def __aenter__(self):
        ticket = (REQUEST_PRIORITY.get(), next(self.order))
        async with self.condition:
            heapq.heappush(self.waiters, ticket)
            try:
                await self.condition.wait_for(lambda: self.active < 128 and self.waiters[0] == ticket)
            except asyncio.CancelledError:
                self.waiters.remove(ticket)
                heapq.heapify(self.waiters)
                self.condition.notify_all()
                raise
            heapq.heappop(self.waiters)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.condition.notify_all()
        self.run.log({"client/active_requests": self.active, "client/peak_requests": self.peak})

    async def __aexit__(self, *exc):
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()


class LimitedResponsesModel(OpenAIResponsesModel):
    def __init__(self, context, semaphore, metrics):
        super().__init__(model=MODEL, openai_client=context.client)
        self.context = context
        self.semaphore = semaphore
        self.metrics = metrics

    async def stream_response(self, *args, **kwargs):
        request = REQUEST_TRACE.get()
        async with self.semaphore:
            async for event in super().stream_response(*args, **kwargs):
                if event.type in ("response.completed", "response.incomplete", "response.failed"):
                    response = event.response
                    if response.usage is None:
                        raise RuntimeError(f"Cannot account for response without usage: {response.id} {event.type}")
                    recorded = {"response_id": response.id, "agent": request["agent"],
                                "usage": response.usage.model_dump(), "service_tier": response.service_tier,
                                "status": event.type}
                    self.context.response_usage[response.id] = recorded
                    if request["agent"] != "judge" and event.type == "response.completed":
                        self.context.path_responses.append(recorded)
                    if "snapshot" in request:
                        self.context.response_states[response.id] = request["snapshot"]
                    self.metrics.record_response(recorded)
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
        embedding_url=EMBEDDING_URL, embedding_model="qwen3-vl-embedding-8b",
    )
    harness.context.model = LimitedResponsesModel(harness.context, semaphore, metrics)
    harness.context.embedding_client = LimitedEmbeddingClient(harness.context.embedding_client, semaphore)
    return harness


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k not in ("semaphore", "metrics")})
async def rollout(question, node, phase, skill_version, semaphore, metrics):
    REQUEST_PRIORITY.set(0 if node["parent"] is not None else 2)
    directory = RUN / phase / question["timeline"] / f"question-{question['index']}" / f"rollout-{node['index']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    saved_answer = directory / "answer.json"
    if saved_answer.exists():
        return json.loads(saved_answer.read_text())
    if (directory / "skipped.json").exists():
        print(f"SKIP saved exhausted rollout: {directory}", flush=True)
        return None
    metadata = {"run": RUN.name, "timeline": question["timeline"], "question": str(question["index"]),
                "rollout": node["index"], "skill_version": skill_version, "phase": phase,
                "parent_rollout": node["parent"]}
    for attempt in range(1, 4):
        attempt_dir = directory / f"attempt-{attempt}"
        if (attempt_dir / "error.json").exists():
            continue
        if attempt_dir.exists():
            attempt_dir.rename(directory / f"interrupted-{attempt}-{uuid.uuid4().hex}")
        harness = make_harness(attempt_dir, RUN / "skills" / skill_version, semaphore, metrics, question["timeline"])
        harness.context.model_settings.extra_body = {
            "prompt_cache_key": f"{RUN.name}:{phase}:{question['timeline']}:{question['index']}"
        }
        started = time.monotonic()
        try:
            with weave.attributes({**metadata, "attempt": attempt}), trace("Trace2Skill QA rollout", metadata=metadata):
                if node["parent"] is None:
                    session = await harness.answer(question["question"])
                else:
                    parent = json.loads(Path(node["source_trace"]).read_text())
                    snapshot = parent["raw_responses"][node["response_index"]]["prefix_state"]
                    (attempt_dir / "prefix.json").write_text(json.dumps(snapshot, ensure_ascii=False) + "\n")
                    restored = await harness.restore_prefix(snapshot)
                    session = await harness.answer(
                        question["question"], input_items=restored,
                        instructions=snapshot["agent"]["instructions"],
                        base_instructions=snapshot["base_instructions"], compactions=snapshot["compactions"],
                    )
                if session.status != "answered" or session.answer is None:
                    raise InvalidTrace(f"QA ended without an answer: {session.status}")
            path_responses = harness.context.path_responses
            trajectory = attempt_dir / "trajectory.json"
            trajectory.write_text(json.dumps({"histories": harness.context.path_histories}, ensure_ascii=False, indent=2) + "\n")
            answer = {
                **metadata, "attempt": attempt, **dataclasses.asdict(session),
                "directory": str(directory), "trajectory_file": str(trajectory),
                "question_text": question["question"], "reference_answer": question["answer"],
                "media_files": sorted(str(p) for p in (attempt_dir / "media").glob("frame-*.json")),
                "trace_length": {
                    "generated_tokens": sum(r["usage"]["output_tokens"] for r in path_responses),
                    "model_calls": len(path_responses), "tool_calls": len(harness.context.tool_calls),
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in path_responses) / len(path_responses),
                    "peak_context_tokens": max(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in path_responses),
                },
                "rollout_time_seconds": time.monotonic() - started + sum(
                    json.loads(p.read_text())["seconds"] for p in directory.glob("*/rollout-time.json")),
            }
            saved_answer.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n")
            print(f"ANSWER phase={phase} timeline={question['timeline']} question={question['index']} rollout={node['index']}", flush=True)
            return answer
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            write_error(attempt_dir / "error.json", error)
            if not retryable(error):
                raise
        finally:
            (attempt_dir / "rollout-time.json").write_text(json.dumps({"seconds": time.monotonic() - started}) + "\n")
            await harness.close()
    (directory / "skipped.json").write_text(json.dumps({**metadata, "reason": "qa_attempts_exhausted"}, indent=2) + "\n")
    print(f"SKIP QA attempts exhausted: {directory}", flush=True)
    return None


@weave.op(postprocess_inputs=lambda x: {k: v for k, v in x.items() if k not in ("semaphore", "metrics")})
async def score_rollout(answer, semaphore, metrics):
    REQUEST_PRIORITY.set(1)
    directory = Path(answer["directory"])
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / "skipped.json").exists():
        return None
    attempt_dir = directory / f"attempt-{answer['attempt']}"
    harness = make_harness(attempt_dir, RUN / "skills" / answer["skill_version"], semaphore, metrics, answer["timeline"])
    try:
        judgment_path = directory / "judgment.json"
        if judgment_path.exists():
            score = json.loads(judgment_path.read_text())["score"]
        else:
            try:
                score = await judge_answer(harness, answer["question_text"], answer["reference_answer"], answer["answer"], 3)
            except JudgeAttemptsExhausted as error:
                (directory / "skipped.json").write_text(json.dumps({"reason": "judge_attempts_exhausted", "error": str(error)}) + "\n")
                print(f"SKIP judge attempts exhausted: {directory}", flush=True)
                return None
            judgment_path.write_text(json.dumps({"score": score}) + "\n")
        trace_files = sorted(str(p) for p in directory.rglob("traces/*.json"))
        billable = {}
        replacement_call_ids = set()
        for filename in trace_files:
            artifact = json.loads(Path(filename).read_text())
            for response in artifact["api_usage"]:
                billable[response["response_id"]] = response
            if artifact["agent"]["name"] != "judge":
                replacement_call_ids.update(call["call_id"] for call in artifact["tool_calls"] if call["name"] == "replace_tool")
        total_context_saved = 0
        for path in sorted(directory.glob("*/replace-*.json")):
            measurement = json.loads(path.read_text())
            if measurement["replace_call_id"] not in replacement_call_ids:
                continue
            for side in ("before", "after"):
                if f"{side}_tokens" in measurement:
                    continue
                items = copy.deepcopy(measurement[side])
                output = items[1]["output"]
                if isinstance(output, list):
                    for index, part in enumerate(output):
                        if part["type"] == "input_image":
                            recipe_name = json.loads(output[index - 1]["text"])["filename"]
                            recipe = path.parent / "media" / recipe_name
                            image_bytes = await asyncio.to_thread(frame_jpeg_bytes, json.loads(recipe.read_text()))
                            part["image_url"] = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                print(f"count replacement tokens side={side} artifact={path}", flush=True)
                async with semaphore:
                    token_count = await harness.context.client.responses.input_tokens.count(
                        model=MODEL, input=[{"role": "user", "content": answer["question_text"]}, *items],
                        reasoning=measurement["reasoning"],
                    )
                measurement[f"{side}_tokens"] = token_count.input_tokens
                path.write_text(json.dumps(measurement, ensure_ascii=False) + "\n")
            total_context_saved += measurement["before_tokens"] - measurement["after_tokens"]
        result = {
            **answer, "score": score, "success": score >= 4, "trace_files": trace_files,
            "cost_usd": sum(response_cost(r) for r in billable.values()),
            "replacement_metrics": {"total_context_saved_tokens": total_context_saved,
                                    "replace_tool_calls": len(replacement_call_ids)},
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"SCORE phase={answer['phase']} timeline={answer['timeline']} question={answer['question']} rollout={answer['rollout']} score={score}", flush=True)
        return result
    finally:
        await harness.close()


async def collect_trees(questions, semaphore, metrics):
    trees = {}
    results = []
    for question in questions:
        key = (question["timeline"], question["index"])
        tree_file = RUN / "collection" / key[0] / f"question-{key[1]}" / "tree.json"
        tree_file.parent.mkdir(parents=True, exist_ok=True)
        if tree_file.exists():
            tree = json.loads(tree_file.read_text())
        else:
            tree = {"timeline": key[0], "question": key[1],
                    "nodes": [{"index": 0, "parent": None, "expanded": False}]}
            tree_file.write_text(json.dumps(tree, indent=2) + "\n")
        trees[key] = (tree_file, tree)

    async def run_node(question, node):
        answer = await rollout(question, node, "collection", "initial", semaphore, metrics)
        tree_file, tree = trees[(question["timeline"], question["index"])]
        if answer is None:
            node["skipped"] = True
            node["expanded"] = True
            tree_file.write_text(json.dumps(tree, indent=2) + "\n")
            return
        if not node["expanded"]:
            candidates = []
            for filename in answer["trace_files"]:
                artifact = json.loads(Path(filename).read_text())
                if artifact["agent"]["name"] != "memory-retrieval":
                    continue
                for index, response in enumerate(artifact["raw_responses"]):
                    calls = [item for item in response["output"] if item["type"] == "function_call"]
                    if calls and all(call["name"] not in ("submit_answer", "agent_raise") for call in calls):
                        candidates.append((filename, index))
            randomizer = random.Random(f"0:{question['timeline']}:{question['index']}:{node['index']}")
            selected = randomizer.sample(candidates, min(5 - len(tree["nodes"]), len(candidates)))
            for filename, response_index in selected:
                child = {"index": len(tree["nodes"]), "parent": node["index"], "expanded": False,
                         "source_trace": filename, "response_index": response_index}
                tree["nodes"].append(child)
                group.create_task(run_node(question, child))
            node["expanded"] = True
            tree_file.write_text(json.dumps(tree, indent=2) + "\n")
        if len(tree["nodes"]) < 5 and all(n["expanded"] for n in tree["nodes"]) and not any("skipped" in n for n in tree["nodes"]):
            raise RuntimeError(f"No unused nonterminal tool points remain before four branches: {tree_file}")
        result = await score_rollout(answer, semaphore, metrics)
        if result is not None:
            results.append(result)
            metrics.record_rollout(result)

    async with asyncio.TaskGroup() as group:
        for question in questions:
            tree = trees[(question["timeline"], question["index"])][1]
            for node in list(tree["nodes"]):
                group.create_task(run_node(question, node))
    return sorted(results, key=lambda r: (r["timeline"], int(r["question"]), r["rollout"]))


async def evaluate_month(questions, version, semaphore, metrics):
    jobs = iter(questions)
    results = []

    async def worker():
        for question in jobs:
            answer = await rollout(question, {"index": 0, "parent": None}, f"month-{version}", version, semaphore, metrics)
            if answer is None:
                continue
            result = await score_rollout(answer, semaphore, metrics)
            if result is not None:
                results.append(result)
                metrics.record_rollout(result)

    async with asyncio.TaskGroup() as group:
        for _ in range(128):
            group.create_task(worker())
    return sorted(results, key=lambda r: int(r["question"]))


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
    # clip_qa is a fixed tool: analyze its caller-visible inputs and results,
    # without exposing the visual agent's internal trajectory or sampled frames.
    histories = []
    media_files = set()
    trajectory = json.loads(Path(result["trajectory_file"]).read_text())
    media_dir = Path(result["trajectory_file"]).parent / "media"
    for artifact in trajectory["histories"]:
        for item in artifact["history"]:
            if item.get("type") != "function_call_output" or not isinstance(item["output"], list):
                continue
            for index, part in enumerate(item["output"]):
                if part["type"] == "input_image":
                    frame = json.loads(item["output"][index - 1]["text"])
                    media_files.add(str(media_dir / frame["filename"]))
        # Analysis reads summaries; encrypted state remains in the replay artifacts.
        histories.append({"file": artifact["file"], "agent": artifact["agent"],
                          "history": [
                              {key: value for key, value in item.items() if key != "encrypted_content"}
                              if item.get("type") == "reasoning" else item
                              for item in artifact["history"]
                          ],
                          "tool_calls": [{k: v for k, v in call.items() if k != "original_output"}
                                         for call in artifact["tool_calls"]],
                          "error": artifact["error"]})
    result = {**result, "trace_files": [str(directory / "analysis_input.json")], "media_files": sorted(media_files)}
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
                instructions += "\nTreat clip_qa as a fixed, opaque tool. Analyze its caller's decisions and use of returned answers and evidence; do not analyze or learn the visual agent's internal reasoning, sampling or cropping. Images are limited to evidence returned to the caller."
                instructions += "\nReturn the Lean Solution Path and at most three Success Memory Items using the supplied JSON schema."
                content = [{"type": "input_text", "text": json.dumps(bundle, ensure_ascii=False)}]
                for filename in result["media_files"]:
                    image_bytes = await asyncio.to_thread(frame_jpeg_bytes, json.loads(Path(filename).read_text()))
                    content.extend([{"type": "input_text", "text": filename},
                                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(), "detail": "high"}])
                messages = [{"role": "user", "content": content}]
                for attempt in range(1, 4):
                    try:
                        (directory / f"success-request-{attempt}.json").write_text(json.dumps(stripped_json(messages), ensure_ascii=False) + "\n")
                        async with semaphore:
                            raw = await harness.context.client.responses.with_raw_response.parse(
                                model=MODEL, instructions=instructions, input=messages,
                                reasoning={"effort": "high", "summary": "auto"}, store=False,
                                include=["reasoning.encrypted_content"],
                                text_format=SuccessReport,
                            )
                        body = raw.content
                        (directory / f"success-response-{attempt}.json").write_bytes(body)
                        payload = json.loads(body)
                        metrics.record_response(payload)
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
                async def read_frame(file: str) -> list[ToolOutputText | ToolOutputImage]:
                    """Read a caller-visible evidence frame listed in media_files; internal video QA samples are not exposed."""
                    if file not in result["media_files"]:
                        raise ValueError("Choose a file from media_files in the task input")
                    recipe = json.loads(Path(file).read_text())
                    image_bytes = await asyncio.to_thread(frame_jpeg_bytes, recipe)
                    return [ToolOutputText(text=json.dumps({"frame_id": recipe["frame_id"], "filename": Path(file).name})),
                            ToolOutputImage(image_url="data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(), detail="high")]

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
              "source_file": str(directory / "analysis_input.json"), "items": items,
              "parent_rollout": result["parent_rollout"],
              "tree_file": str(Path(result["directory"]).parent / "tree.json")}
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"analysis {record['instance_id']} {record['record_source']}", flush=True)
    return record


def summarize(phase, questions, results, rollouts):
    present = {(r["timeline"], int(r["question"]), r["rollout"]) for r in results}
    expected = [(q["timeline"], q["index"], index) for q in questions for index in range(rollouts)]
    if not results:
        raise RuntimeError(f"No scored rollouts in {phase}")
    summary = {
        "requested_rollouts": len(expected), "completed_rollouts": len(results),
        "missing_rollouts": [key for key in expected if key not in present],
        "mean_score": sum(r["score"] for r in results) / len(results),
        "score_ge_4_fraction": sum(r["score"] >= 4 for r in results) / len(results),
        "cost_per_rollout_usd": sum(r["cost_usd"] for r in results) / len(results),
    }
    (RUN / f"{phase}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def main():
    RUN.mkdir(parents=True, exist_ok=True)
    init_weave()
    config = {
        "model": MODEL, "reasoning": "high", "reasoning_summary": "auto", "base_url": BASE_URL,
        "api": "responses", "reasoning_replay": True, "client_concurrency": 128, "replace_tool": True,
        "collection": {"split": "test", "timelines": ["day", "week"], "questions_per_timeline": 200,
                       "root_rollouts": 1, "branch_rollouts": 4, "branch_before": "model_tool_decision"},
        "evaluation": {"timeline": "month", "split": "val", "questions": 100, "rollouts": 1,
                       "arms": ["initial", "evolved"]},
        "seed": 0,
    }
    wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name,
                           name=RUN.name, resume="allow", dir=str(RUN), config=config)
    (RUN / "wandb-run.json").write_text(json.dumps({"id": wandb_run.id, "url": wandb_run.url}, indent=2) + "\n")
    try:
        initial = RUN / "skills/initial"
        if not initial.exists():
            shutil.copytree(ROOT / "skills/initial", initial)
        for source in (ROOT / "skills/initial").glob("*.md"):
            if (initial / source.name).read_bytes() != source.read_bytes():
                raise RuntimeError(f"Initial skill changed: {source.name}")
        selection = RUN / "questions.json"
        if selection.exists():
            questions = json.loads(selection.read_text())
        else:
            questions = [
                {**question, "timeline": timeline}
                for timeline in ("day", "week")
                for question in json.loads((DATASET / timeline / "test.json").read_text())
            ]
            selection.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n")
        month_selection = RUN / "month-questions.json"
        if month_selection.exists():
            month_questions = json.loads(month_selection.read_text())
        else:
            month_questions = [
                {**question, "timeline": "month"} for question in
                random.Random(0).sample(json.loads((DATASET / "month/val.json").read_text()), 100)
            ]
            month_selection.write_text(json.dumps(month_questions, ensure_ascii=False, indent=2) + "\n")
        semaphore = RequestSlots(wandb_run)
        metrics = RunMetrics(wandb_run, "collection", RUN)
        results = await collect_trees(questions, semaphore, metrics)
        summarize("collection", questions, results, 5)
        analysis_jobs = iter(results)
        metrics = RunMetrics(wandb_run, "analysis", RUN)
        records = []
        analysis_skipped = []

        async def analysis_worker():
            for result in analysis_jobs:
                record = await analyze(result, semaphore, metrics)
                if record is None:
                    analysis_skipped.append((result["timeline"], result["question"], result["rollout"]))
                else:
                    records.append(record)

        async with asyncio.TaskGroup() as group:
            for _ in range(128):
                group.create_task(analysis_worker())
        records.sort(key=lambda r: r["instance_id"])
        (RUN / "analysis-skipped.json").write_text(json.dumps(analysis_skipped, indent=2) + "\n")
        (RUN / "combined-records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        evolved = RUN / "skills/evolved"
        if not evolved.exists():
            weave.finish()
            wandb_run.finish()
            with (RUN / "evolution.log").open("a") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(UPSTREAM / "baseline_evolve.py"), str(RUN),
                    cwd=UPSTREAM, stdout=log, stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    status = await process.wait()
                except asyncio.CancelledError:
                    process.terminate()
                    await process.wait()
                    raise
            if status != 0:
                raise RuntimeError(f"Trace2Skill evolution exited {status}; see {RUN / 'evolution.log'}")
            init_weave()
            wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=RUN.name,
                                   name=RUN.name, resume="must", dir=str(RUN))
            semaphore.run = wandb_run
        evaluation = {}
        for version in ("initial", "evolved"):
            phase = f"month-{version}"
            metrics = RunMetrics(wandb_run, phase, RUN)
            evaluation[version] = await evaluate_month(month_questions, version, semaphore, metrics)
            summarize(phase, month_questions, evaluation[version], 1)
        indexed = {version: {r["question"]: r for r in values} for version, values in evaluation.items()}
        paired = sorted(set(indexed["initial"]) & set(indexed["evolved"]), key=int)
        if not paired:
            raise RuntimeError("Month evaluation has no complete pairs")
        comparison = {
            "requested_questions": len(month_questions), "paired_questions": len(paired),
            "missing_pairs": [q["index"] for q in month_questions if str(q["index"]) not in paired],
            "arms": {
                version: {
                    "mean_score": sum(indexed[version][q]["score"] for q in paired) / len(paired),
                    "score_ge_4_fraction": sum(indexed[version][q]["score"] >= 4 for q in paired) / len(paired),
                } for version in ("initial", "evolved")
            },
            "questions": [
                {"question_index": int(q), **{
                    version: {"answer": indexed[version][q]["answer"], "score": indexed[version][q]["score"],
                              "result_file": str(Path(indexed[version][q]["directory"]) / "result.json")}
                    for version in ("initial", "evolved")
                }} for q in paired
            ],
        }
        comparison["delta"] = {
            metric: comparison["arms"]["evolved"][metric] - comparison["arms"]["initial"][metric]
            for metric in ("mean_score", "score_ge_4_fraction")
        }
        (RUN / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
        wandb_run.log({f"comparison/delta_{key}": value for key, value in comparison["delta"].items()})
        print("TRACE2SKILL COMPLETE " + json.dumps({"paired_questions": len(paired),
              "arms": comparison["arms"], "delta": comparison["delta"]}), flush=True)
    finally:
        weave.finish()
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())
