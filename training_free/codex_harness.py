"""Codex answering client and per-attempt VideoSQL stdio MCP server."""

import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

import httpx
from agents import ModelSettings, OpenAIResponsesModel, RunConfig, ToolOutputText
from agents.tool_context import ToolContext
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from openai import APIStatusError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient
from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import ConfigReadResponse, ListMcpServerStatusResponse
from openai_codex.types import ReasoningEffort
from pydantic import BaseModel

from training_free.harness import AgentRaised, ClipSubmission, HarnessContext, InvalidTrace, make_tools, read_skill_catalog, stripped_json
from training_free.index_schema import SCHEMA
from training_free.mmlifelong_data import MM_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-terra"
QA_MODEL = "Qwen/Qwen3.8-27B-FP8"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
SKILLS = ROOT / "training_free/skills-vector"
TOOL_NAMES = ["run_sql", "semantic_search", "clip_qa", "add_note", "read_notes", "read_skill", "agent_raise", "submit_answer"]


class AttemptFailed(Exception):
    pass


def trace_json(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if dataclasses.is_dataclass(value):
        return {key: trace_json(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: trace_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [trace_json(item) for item in value]
    return value


def instructions_for(dataset):
    schema = MM_SCHEMA if dataset == "mmlifelong-month" else SCHEMA
    instructions = "Answer the user's question by retrieving evidence from the memory database. Use submit_answer when finished. Use agent_raise when the investigation cannot proceed. The complete database schema is:\n" + schema
    instructions += "\n\nRetrieval: run_sql and semantic_search return raw results with question=null. Their optional question parameter requests a separate QA answer from the retrieved evidence. semantic_search scores every eligible row of the chosen table by cosine similarity, then runs sql over semantic_hits. Use sql to filter, rank, order or aggregate. visual_clip searches visual embeddings; the other tables search text embeddings."
    instructions += "\n\nSQL scope: Restrict run_sql with a relevant outer WHERE, finite LIMIT, HAVING, QUALIFY, sampling or compact global COUNT/MIN/MAX/SUM/AVG aggregation. ORDER BY alone and a QA question do not restrict SQL. Natural-language columns are casefolded to lowercase; identifiers, time strings and metadata preserve their case."
    instructions += "\n\nVisual verification: Use clip_qa after retrieval locates an interval and a specific unresolved detail needs viewing. Choose fps for the detail, within the tool's sampling allowance. Preserve video identity, evidence IDs, times, uncertainty and contradictions."
    if dataset == "egolife-open":
        instructions += "\n\nThe questioner and camera wearer is Jake / A1_JAKE. I, me and my refer to Jake. Interpret relative times using the question's original query time. Database times use DHHMMSSFF; clip_qa uses seconds since DAY1 00:00:00. Use semantic_search start_s/end_s for media intervals. Evidence must precede the query time."
    instructions += "\n\n" + (SKILLS / "ROOT-SKILL.md").read_text()
    instructions += "\nAvailable skills:\n" + "\n".join(f"- {skill['name']}: {skill['description']}" for skill in read_skill_catalog(SKILLS))
    return instructions


class CodexContext(HarnessContext):
    def reserve_model_call(self, role):
        self.investigation_calls += 1
        print(f"QA_CALL role={role} count={self.investigation_calls}", file=sys.stderr, flush=True)

    async def summary_response(self, name, instructions, content, text_format):
        request = {"model": self.model.model, "instructions": instructions, "input": [{"role": "user", "content": content}], "reasoning": self.model_settings.reasoning, "store": False, "include": ["reasoning.encrypted_content"]}
        path = self.run_dir / "traces" / f"{name}-{uuid.uuid4().hex}.json"
        artifact = {"agent": name, "api": "responses", "request": stripped_json(request), "response": None, "error": None}
        path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")
        print(f"QA_START role={name} parts={len(content)} trace={path}", file=sys.stderr, flush=True)
        try:
            self.reserve_model_call(name)
            raw = await self.client.responses.with_raw_response.parse(**request, text_format=text_format)
            payload = json.loads(raw.content)
            artifact["response"] = payload
            recorded = {"response_id": payload["id"], "agent": name, "model": payload["model"], "usage": payload["usage"]}
            self.response_usage[payload["id"]] = recorded
            response = raw.parse()
            if response.status != "completed" or response.output_parsed is None:
                raise InvalidTrace(f"{name} returned no completed structured answer")
            return response.output_parsed, recorded
        except BaseException as error:
            artifact["error"] = {"type": type(error).__name__, "message": str(error)}
            print(f"QA_ERROR role={name} error={error} trace={path}", file=sys.stderr, flush=True)
            raise
        finally:
            path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")

    async def clip_response(self, request, call_id, frame_files):
        path = self.run_dir / "traces" / f"clip-{call_id}.json"
        artifact = {"agent": "clip-qa", "api": "chat.completions", "request": stripped_json(request), "frame_files": frame_files, "response": None, "error": None}
        path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")
        print(f"QA_START role=clip-qa frames={len(frame_files)} trace={path}", file=sys.stderr, flush=True)
        try:
            self.reserve_model_call("clip-qa")
            raw = await self.client.chat.completions.with_raw_response.parse(**request, response_format=ClipSubmission)
            payload = json.loads(raw.content)
            artifact["response"] = payload
            recorded = {"response_id": payload["id"], "agent": "clip-qa", "model": payload["model"], "usage": payload["usage"]}
            self.response_usage[payload["id"]] = recorded
            return raw, recorded
        except BaseException as error:
            artifact["error"] = {"type": type(error).__name__, "message": str(error)}
            print(f"QA_ERROR role=clip-qa error={error} trace={path}", file=sys.stderr, flush=True)
            raise
        finally:
            path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")


@contextlib.asynccontextmanager
async def tool_context(dataset, artifact_root, directory, question_id, query_time, qa_url, embedding_url):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "traces").mkdir(exist_ok=True)

    async def record_request(request):
        path = directory / "traces" / f"wire-{uuid.uuid4().hex}.json"
        request.extensions["trace_path"] = path
        body = json.loads(request.content) if request.content else None
        path.write_text(json.dumps({"url": str(request.url), "request": stripped_json(body), "response": None}, ensure_ascii=False) + "\n")

    async def record_response(response):
        await response.aread()
        path = response.request.extensions["trace_path"]
        record = json.loads(path.read_text())
        record.update({"status": response.status_code, "response": response.text})
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n")

    hooks = {"request": [record_request], "response": [record_response]}
    async with AsyncOpenAI(api_key="EMPTY", base_url=qa_url, timeout=None, max_retries=0, http_client=DefaultAsyncHttpxClient(event_hooks=hooks)) as client, httpx.AsyncClient(timeout=httpx.Timeout(connect=60, read=100000, write=100000, pool=100000), event_hooks=hooks) as embedding_client:
        for url, expected in ((qa_url, QA_MODEL), (embedding_url, EMBEDDING_MODEL)):
            print(f"CHECK_MODEL url={url} expected={expected}", file=sys.stderr, flush=True)
            response = await embedding_client.get(url.rstrip("/") + "/models", timeout=10)
            response.raise_for_status()
            if expected not in [item["id"] for item in response.json()["data"]]:
                raise RuntimeError(f"Unexpected model at {url}: {response.text}")
        month = dataset == "mmlifelong-month"
        database = artifact_root / "index.duckdb" if month else artifact_root / "questions" / f"{question_id}.duckdb"
        context = CodexContext(
            database=database,
            embedding_index=artifact_root / "embeddings",
            media_map=artifact_root / "media-map.json",
            skill_dir=SKILLS,
            run_dir=directory,
            model=OpenAIResponsesModel(model=QA_MODEL, openai_client=client),
            client=client,
            model_settings=ModelSettings(reasoning={"effort": "xhigh", "summary": "auto"} if month else {"summary": "auto"}, store=False, response_include=["reasoning.encrypted_content"], preserve_raw_usage=True),
            run_config=RunConfig(tracing_disabled=True, reasoning_item_id_policy="preserve"),
            embedding_url=embedding_url,
            embedding_model=EMBEDDING_MODEL,
            embedding_client=embedding_client,
            query_time=query_time,
            time_format="mmlifelong" if month else "egolife",
            semantic_search_mode="vector",
            clip_qa_video=True,
        )
        yield context


async def serve():
    dataset = os.environ["VIDEO_SQL_DATASET"]
    directory = Path(os.environ["VIDEO_SQL_ATTEMPT_DIR"])
    async with stdio_server() as (read_stream, write_stream):
        with contextlib.redirect_stdout(sys.stderr):
            async with tool_context(dataset, Path(os.environ["VIDEO_SQL_ARTIFACT_ROOT"]), directory, os.environ["VIDEO_SQL_QUESTION_ID"], json.loads(os.environ["VIDEO_SQL_QUERY_TIME"]), os.environ["VIDEO_SQL_QA_URL"], os.environ["VIDEO_SQL_EMBEDDING_URL"]) as context:
                functions, _ = make_tools(context)
                tools = {tool.name: tool for tool in functions}
                terminal = None

                async def list_tools(request_context, params):
                    return types.ListToolsResult(tools=[types.Tool(name=tool.name, description=tool.description, input_schema=tool.params_json_schema) for tool in tools.values()])

                async def call_tool(request_context, params):
                    nonlocal terminal
                    if terminal is not None:
                        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(terminal))], is_error=True)
                    context.tool_counter += 1
                    call_id = uuid.uuid4().hex
                    arguments = json.dumps({} if params.arguments is None else params.arguments, ensure_ascii=False)
                    path = directory / "traces" / f"tool-{call_id}.json"
                    record = {"tool_id": context.tool_counter, "name": params.name, "arguments": json.loads(arguments), "output": None, "error": None}
                    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"TOOL_START name={params.name} arguments={arguments} trace={path}", file=sys.stderr, flush=True)
                    is_error = False
                    try:
                        output = await tools[params.name].on_invoke_tool(ToolContext(context=context, tool_name=params.name, tool_call_id=call_id, tool_arguments=arguments), arguments)
                    except AgentRaised as error:
                        terminal = {"category": "ordinary", "type": "AgentRaised", "message": str(error)}
                        output = json.dumps({"error": terminal})
                        is_error = True
                    except BaseException as error:
                        ordinary = isinstance(error, (InvalidTrace, APITimeoutError, httpx.TimeoutException)) or isinstance(error, APIStatusError) and (error.status_code in (408, 429) or error.status_code >= 500)
                        terminal = {"category": "ordinary" if ordinary else "maintenance", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                        output = json.dumps({"error": terminal})
                        is_error = True
                        print(f"TOOL_ERROR name={params.name} error={terminal}", file=sys.stderr, flush=True)
                    if terminal is not None:
                        record["error"] = terminal
                        (directory / "tool-failure.json").write_text(json.dumps(terminal, ensure_ascii=False) + "\n")
                    content = [types.TextContent(type="text", text=json.dumps({"tool_id": record["tool_id"]}))]
                    if isinstance(output, str):
                        content.append(types.TextContent(type="text", text=output))
                    else:
                        for part in output:
                            if isinstance(part, ToolOutputText):
                                content.append(types.TextContent(type="text", text=part.text))
                            else:
                                content.append(types.ImageContent(type="image", data=part.image_url.split(",", 1)[1], mime_type="image/jpeg"))
                    result = types.CallToolResult(content=content, is_error=is_error)
                    record["output"] = result.model_dump(mode="json", by_alias=True)
                    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                    if params.name == "submit_answer" and not is_error:
                        terminal = {"answer": json.loads(output)["answer"]}
                    print(f"TOOL_DONE name={params.name} error={is_error}", file=sys.stderr, flush=True)
                    return result

                server = Server("video_sql", instructions=instructions_for(dataset), on_list_tools=list_tools, on_call_tool=call_tool)
                (directory / "mcp-context.json").write_text(json.dumps({"dataset": dataset, "database": str(context.database), "query_time": context.query_time, "query_time_s": context.query_time_s, "skills": str(SKILLS), "month_skills": str(ROOT / "training_free/skills-month") if dataset == "mmlifelong-month" else None, "tools": [tool.model_dump(mode="json", by_alias=True) for tool in (await list_tools(None, None)).tools]}, ensure_ascii=False, indent=2) + "\n")
                await server.run(read_stream, write_stream, server.create_initialization_options(), raise_exceptions=True)


def answer_question(dataset, artifact_root, directory, question, qa_url, embedding_url, stop):
    directory.mkdir(parents=True, exist_ok=True)
    working = directory / "workspace"
    working.mkdir(exist_ok=True)
    env = {
        "PYTHONPATH": str(ROOT), "HF_HOME": "/raid/huggingface",
        "VIDEO_SQL_DATASET": dataset, "VIDEO_SQL_ARTIFACT_ROOT": str(artifact_root),
        "VIDEO_SQL_ATTEMPT_DIR": str(directory), "VIDEO_SQL_QUESTION_ID": str(question["id"]),
        "VIDEO_SQL_QUERY_TIME": json.dumps(question["query_time"]),
        "VIDEO_SQL_QA_URL": qa_url, "VIDEO_SQL_EMBEDDING_URL": embedding_url,
        "OMP_NUM_THREADS": "10", "MKL_NUM_THREADS": "10", "OPENBLAS_NUM_THREADS": "10", "NUMEXPR_NUM_THREADS": "10",
    }
    overrides = ['web_search="disabled"', 'project_doc_max_bytes=0', 'skills.include_instructions=false', 'skills.bundled.enabled=false', 'tools.update_plan.enabled=false', 'tools.experimental_request_user_input.enabled=false', 'agents.enabled=false', 'features.use_legacy_landlock=true']
    for feature in ("shell_tool", "unified_exec", "view_image", "apps", "plugins", "remote_plugin", "skill_search", "skill_mcp_dependency_install", "multi_agent", "multi_agent_v2", "browser_use", "computer_use", "image_generation", "workspace_dependencies", "tool_suggest", "goals", "hooks", "memories", "sleep_tool", "code_mode", "code_mode_host"):
        overrides.append(f"features.{feature}=false")
    config = CodexConfig(cwd=str(working), config_overrides=tuple(overrides))
    instructions = instructions_for(dataset)
    prompt = f"Question ID: {question['id']}\nQuestion: {question['question']}"
    if dataset == "egolife-open":
        prompt += f"\nQuery time: {question['query_time']} (DHHMMSSFF). Questioner: Jake / A1_JAKE."
    request = {"model": MODEL, "effort": "max", "service_tier": "default", "ephemeral": False, "prompt": prompt, "instructions": instructions, "overrides": overrides}
    (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    answer = None
    terminal = None
    usage = None
    completed = None
    thread = None
    with (directory / "events.jsonl").open("w") as trace, Codex(config) as codex:
        try:
            inherited = codex._client.request("config/read", {"includeLayers": False}, response_model=ConfigReadResponse).config.model_dump(mode="json", by_alias=True)
            registrations = {name: {"enabled": False, "required": False} for name in inherited["mcp_servers"]}
            registrations["video_sql"] = {"command": sys.executable, "args": ["-m", "training_free.codex_harness"], "cwd": str(ROOT), "env": env, "enabled": True, "required": True, "default_tools_approval_mode": "approve", "enabled_tools": TOOL_NAMES, "tool_timeout_sec": 100000}
            request["mcp_servers"] = registrations
            (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
            thread = codex.thread_start(model=MODEL, service_tier="default", ephemeral=False, cwd=str(working), approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only, base_instructions=instructions, developer_instructions="", config={"mcp_servers": registrations})
            (directory / "thread.json").write_text(json.dumps(trace_json(thread.read().thread), ensure_ascii=False) + "\n")
            inventory = codex._client.request("mcpServerStatus/list", {"threadId": thread.id}, response_model=ListMcpServerStatusResponse)
            (directory / "mcp-inventory.json").write_text(json.dumps(trace_json(inventory), ensure_ascii=False) + "\n")
            turn = thread.turn(prompt, effort=ReasoningEffort.max)
            for event in turn.stream():
                record = trace_json(event)
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                if stop.is_set() and terminal is None and answer is None and event.method != "turn/completed":
                    terminal = {"category": "maintenance", "message": "Batch stopped after another worker's dependency/code failure"}
                    trace.write(json.dumps({"interrupt": trace_json(turn.interrupt()), "reason": terminal}) + "\n")
                if event.method == "thread/tokenUsage/updated":
                    usage = record["payload"]["tokenUsage"]["total"]
                    (directory / "usage.json").write_text(json.dumps(usage) + "\n")
                elif event.method == "item/completed":
                    item = record["payload"]["item"]
                    if item["type"] != "mcpToolCall":
                        continue
                    failure_path = directory / "tool-failure.json"
                    if terminal is None and answer is None and failure_path.exists():
                        terminal = json.loads(failure_path.read_text())
                        trace.write(json.dumps({"interrupt": trace_json(turn.interrupt()), "reason": terminal}) + "\n")
                    elif terminal is None and answer is None and item["server"] == "video_sql" and item["tool"] == "submit_answer" and item["status"] == "completed":
                        for block in item["result"]["content"]:
                            if block["type"] == "text":
                                data = json.loads(block["text"])
                                if "answer" in data:
                                    answer = data["answer"]
                        if answer is not None:
                            (directory / "submitted-answer.json").write_text(json.dumps({"answer": answer, "thread_id": thread.id}, ensure_ascii=False) + "\n")
                            trace.write(json.dumps({"interrupt": trace_json(turn.interrupt()), "reason": "submit_answer"}) + "\n")
                elif event.method == "turn/completed":
                    completed = record["payload"]["turn"]
            if terminal is not None:
                if terminal["category"] == "maintenance":
                    raise RuntimeError(f"MCP dependency/code failure: {terminal}")
                raise AttemptFailed(str(terminal))
            if answer is None:
                if completed is not None and completed["status"] == "failed":
                    info = completed["error"]["codexErrorInfo"]
                    if info in ("contextWindowExceeded", "rateLimitExceeded", "serverOverloaded", "internalServerError"):
                        raise AttemptFailed(f"Codex turn failed: {completed}")
                    if isinstance(info, dict):
                        status = next(iter(info.values()))["httpStatusCode"]
                        if status is None or status in (408, 429) or status >= 500:
                            raise AttemptFailed(f"Codex turn failed: {completed}")
                    raise RuntimeError(f"Codex turn failed: {completed}")
                raise AttemptFailed(f"Codex turn ended without submit_answer: {completed}")
            return {"id": question["id"], "candidate_answer": answer, "thread_id": thread.id, "usage": usage, "attempt_dir": str(directory)}
        except BaseException as error:
            record = {"category": "ordinary" if isinstance(error, AttemptFailed) else "maintenance", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            (directory / "error.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"ANSWER_ERROR id={question['id']} error={error} directory={directory}", flush=True)
            raise
        finally:
            if thread is not None:
                archived = codex.thread_archive(thread.id)
                (directory / "archive.json").write_text(json.dumps({"thread_id": thread.id, "response": trace_json(archived)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(serve())
