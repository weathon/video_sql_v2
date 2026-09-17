"""Codex answering client and per-question VideoSQL stdio MCP server."""

import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from pathlib import Path

import httpx
from agents import ModelSettings, OpenAIResponsesModel, RunConfig, ToolOutputText, function_tool
from agents.tool_context import ToolContext
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient, InternalServerError, LengthFinishReasonError, RateLimitError
from openai_codex import ApprovalMode, Sandbox
from openai_codex.types import ReasoningEffort
from pydantic import ValidationError

from src.analysis import trace_json
from src.harness import AgentRaised, HarnessContext, InvalidTrace, REQUEST_TRACE, make_tools, stripped_json, tool_error_json
from src.mcp import record_wire_request
from training_free.index_schema import SCHEMA
from training_free.mmlifelong_data import MM_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-terra"
QA_MODEL = "Qwen/Qwen3.8-27B-FP8"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, LengthFinishReasonError, httpx.TransportError, ValidationError, json.JSONDecodeError, InvalidTrace)
DATASETS = {
    "month": {"artifacts": ROOT / "runs/mmlifelong-month/artifacts", "baseline": ROOT / "runs/mmlifelong-month/results/month-sql-gpt5", "time_format": "mmlifelong", "denominator": 623, "reasoning_effort": "xhigh", "attempts": 2},
    "egolife": {"artifacts": Path("/raid/videosql-video-artifacts"), "baseline": ROOT / "runs/fair-qwen38/formal/fair-qwen38-sql-open", "time_format": "egolife", "denominator": 500, "reasoning_effort": None, "attempts": 4},
}


class CodexHarnessContext(HarnessContext):
    def reserve_model_call(self, role: str) -> None:
        self.investigation_calls += 1
        print(f"QA_CALL role={role} count={self.investigation_calls}", file=sys.stderr, flush=True)

    async def summary_response(self, name, instructions, content, text_format):
        request = {"model": QA_MODEL, "instructions": instructions, "input": [{"role": "user", "content": content}], "reasoning": self.model_settings.reasoning, "store": False, "include": ["reasoning.encrypted_content"]}
        path = self.run_dir / "traces" / f"{name}-{uuid.uuid4().hex}.json"
        artifact = {"input": stripped_json(request), "response": None, "error": None}
        path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")
        print(f"QA_START name={name} trace={path}", file=sys.stderr, flush=True)
        try:
            self.reserve_model_call(name)
            REQUEST_TRACE.set({"trace_dir": str(path.parent), "trace_file": str(path), "agent": name})
            async with self.model_limiter:
                raw = await self.client.responses.with_raw_response.parse(**request, text_format=text_format)
            payload = json.loads(raw.content)
            artifact["response"] = payload
            response = raw.parse()
            if response.status != "completed" or response.output_parsed is None:
                raise InvalidTrace(f"{name} returned no completed structured answer")
            recorded = {"response_id": payload["id"], "agent": name, "model": payload["model"], "usage": payload["usage"]}
            self.response_usage[payload["id"]] = recorded
            return response.output_parsed, recorded
        except BaseException as error:
            artifact["error"] = {"type": type(error).__name__, "message": str(error)}
            print(f"QA_ERROR name={name} error={error} trace={path}", file=sys.stderr, flush=True)
            raise
        finally:
            path.write_text(json.dumps(artifact, ensure_ascii=False) + "\n")


def instructions_for(dataset):
    text = (ROOT / "src/prompt.md").read_text()
    prefix, retrieval = text.split("\n\nRetrieval:", 1)
    text = prefix.split("CREATE TABLE", 1)[0] + (MM_SCHEMA if dataset == "month" else SCHEMA) + "\n\nRetrieval:" + retrieval
    if dataset == "month":
        text = text.replace("WHERE start_ts >= 200000000 AND end_ts < 300000000", "WHERE video_id = '14' AND start_ts >= 300 AND end_ts < 600").replace("requested day/time", "requested video/time")
    return text


async def serve_mcp(config_path):
    config = json.loads(config_path.read_text())
    dataset = DATASETS[config["dataset"]]
    artifacts = dataset["artifacts"]
    directory = config_path.parent
    async with stdio_server() as (read_stream, write_stream):
        async with AsyncOpenAI(api_key="EMPTY", base_url=config["qa_url"], timeout=None, max_retries=0, http_client=DefaultAsyncHttpxClient(event_hooks={"request": [record_wire_request]})) as client, httpx.AsyncClient(timeout=httpx.Timeout(connect=60, read=100000, write=100000, pool=100000)) as embedding_client:
            for url, expected in ((config["qa_url"], QA_MODEL), (config["embedding_url"], EMBEDDING_MODEL)):
                print(f"CHECK_MODEL url={url} expected={expected}", file=sys.stderr, flush=True)
                response = await embedding_client.get(url.rstrip("/") + "/models", timeout=10)
                response.raise_for_status()
                if expected not in [item["id"] for item in response.json()["data"]]:
                    raise RuntimeError(f"Unexpected model at {url}: {response.text}")
            reasoning = {"summary": "auto"}
            if dataset["reasoning_effort"] is not None:
                reasoning["effort"] = dataset["reasoning_effort"]
            context = CodexHarnessContext(
                database=artifacts / ("index.duckdb" if config["dataset"] == "month" else f"questions/{config['id']}.duckdb"),
                embedding_index=artifacts / "embeddings", media_map=artifacts / "media-map.json", skill_dir=ROOT / "training_free/skills-vector", run_dir=directory,
                model=OpenAIResponsesModel(model=QA_MODEL, openai_client=client), client=client,
                model_settings=ModelSettings(reasoning=reasoning, store=False, response_include=["reasoning.encrypted_content"], preserve_raw_usage=True),
                run_config=RunConfig(tracing_disabled=True, reasoning_item_id_policy="preserve"),
                embedding_url=config["embedding_url"], embedding_model=EMBEDDING_MODEL, embedding_client=embedding_client,
                query_time=config["query_time"], model_limiter=asyncio.Semaphore(10), time_format=dataset["time_format"], semantic_search_mode="vector", clip_qa_video=True,
            )
            functions, catalog = make_tools(context)
            filenames = {skill["name"]: skill["filename"] for skill in catalog}

            @function_tool(failure_error_function=tool_error_json)
            async def read_skill(name: str) -> str:
                """Read the complete Markdown for one listed skill."""
                if name not in filenames:
                    raise ValueError(f"unknown skill: {name}")
                skill_dir = ROOT / "training_free/skills-month" if config["dataset"] == "month" and name in ("sql-schema", "semantic-search") else context.skill_dir
                return (skill_dir / filenames[name]).read_text().replace(" and consumes a shared investigation model call", "")

            tools = {tool.name: tool for tool in functions}
            tools["read_skill"] = read_skill

            async def list_tools(request_context, params):
                return types.ListToolsResult(tools=[types.Tool(name=tool.name, description=tool.description, input_schema=tool.params_json_schema) for tool in tools.values()])

            async def call_tool(request_context, params):
                context.tool_counter += 1
                call_id = uuid.uuid4().hex
                arguments = json.dumps({} if params.arguments is None else params.arguments, ensure_ascii=False)
                path = directory / "traces" / f"tool-{call_id}.json"
                record = {"tool_id": context.tool_counter, "name": params.name, "arguments": arguments, "output": None, "error": None}
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"TOOL_START name={params.name} arguments={arguments} trace={path}", file=sys.stderr, flush=True)
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        output = await tools[params.name].on_invoke_tool(ToolContext(context=context, tool_name=params.name, tool_call_id=call_id, tool_arguments=arguments), arguments)
                    record["output"] = stripped_json(output)
                    content = [types.TextContent(type="text", text=json.dumps({"tool_id": record["tool_id"]}) + "\n")]
                    if isinstance(output, str):
                        content.append(types.TextContent(type="text", text=output))
                    else:
                        for part in output:
                            if isinstance(part, ToolOutputText):
                                content.append(types.TextContent(type="text", text=part.text))
                            else:
                                content.append(types.ImageContent(type="image", data=part.image_url.split(",", 1)[1], mime_type="image/jpeg"))
                    return types.CallToolResult(content=content, is_error=False)
                except BaseException as error:
                    record["error"] = {"type": type(error).__name__, "message": str(error), "retryable": isinstance(error, RETRYABLE)}
                    print(f"TOOL_ERROR name={params.name} error={error} trace={path}", file=sys.stderr, flush=True)
                    raise
                finally:
                    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")

            server = Server("video_sql", instructions=instructions_for(config["dataset"]), on_list_tools=list_tools, on_call_tool=call_tool)
            print(f"MCP_READY directory={directory} database={context.database}", file=sys.stderr, flush=True)
            await server.run(read_stream, write_stream, server.create_initialization_options(), raise_exceptions=True)


def answer_question(codex, dataset, question, directory, qa_url, embedding_url):
    directory.mkdir(parents=True, exist_ok=True)
    context_path = directory / "context.json"
    context_path.write_text(json.dumps({"dataset": dataset, "id": question["id"], "query_time": question["query_time"], "qa_url": qa_url, "embedding_url": embedding_url}, ensure_ascii=False) + "\n")
    prompt = f"Question ID: {question['id']}\nQuestion: {question['question']}"
    if dataset == "egolife":
        prompt += f"\nQuery time: {question['query_time']} (EgoLife DHHMMSSFF)."
    config = {
        "mcp_servers.video_sql.command": sys.executable,
        "mcp_servers.video_sql.args": ["-m", "training_free.codex_harness", "--serve", str(context_path)],
        "mcp_servers.video_sql.required": True,
        "mcp_servers.video_sql.default_tools_approval_mode": "approve",
        "mcp_servers.video_sql.tool_timeout_sec": 100000,
        "features.use_legacy_landlock": True,
    }
    thread = None
    with (directory / "events.jsonl").open("w") as trace:
        trace.write(json.dumps({"request": {"model": MODEL, "effort": "max", "service_tier": "default", "prompt": prompt, "config": config}}, ensure_ascii=False) + "\n")
        trace.flush()
        try:
            thread = codex.thread_start(approval_mode=ApprovalMode.deny_all, model=MODEL, service_tier="default", ephemeral=False, cwd=str(ROOT), sandbox=Sandbox.read_only, developer_instructions=instructions_for(dataset), config=config)
            trace.write(json.dumps({"thread": trace_json(thread.read().thread)}, ensure_ascii=False) + "\n")
            turn = thread.turn(prompt, effort=ReasoningEffort.max)
            final_response = None
            completed = None
            for event in turn.stream():
                record = trace_json(event)
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                if event.method == "item/completed":
                    item = record["payload"]["item"]
                    print(f"CODEX_ITEM id={question['id']} type={item['type']} trace={directory}", flush=True)
                    if item["type"] == "agentMessage" and item["phase"] in (None, "final_answer"):
                        final_response = item["text"]
                elif event.method == "turn/completed":
                    completed = record["payload"]["turn"]
            submitted = None
            tool_records = [json.loads(path.read_text()) for path in (directory / "traces").glob("tool-*.json")]
            for record in sorted(tool_records, key=lambda item: item["tool_id"]):
                if record["error"] is not None:
                    error = record["error"]
                    if error["retryable"]:
                        raise InvalidTrace(f"Tool {record['name']}: {error}")
                    raise RuntimeError(f"Tool {record['name']}: {error}")
                if record["name"] == "submit_answer":
                    submitted = json.loads(record["output"])["answer"]
            if completed is not None and completed["status"] == "failed":
                error_info = completed["error"]["codexErrorInfo"]
                if error_info in ("contextWindowExceeded", "rateLimitExceeded", "serverOverloaded", "internalServerError") or isinstance(error_info, dict) and any(key in error_info for key in ("httpConnectionFailed", "responseStreamConnectionFailed", "responseStreamDisconnected", "responseTooManyFailedAttempts")):
                    raise InvalidTrace(f"Codex turn failed: {completed['error']}")
            if completed is None or completed["status"] != "completed":
                raise RuntimeError(f"Codex turn did not complete: {completed}")
            answer = submitted if submitted is not None else final_response
            if answer is None or not answer.strip():
                raise InvalidTrace("Codex returned no answer")
            return {"id": question["id"], "candidate_answer": answer, "final_response": final_response, "thread_id": thread.id, "trace_dir": str(directory)}
        except BaseException as error:
            trace.write(json.dumps({"error": {"type": type(error).__name__, "message": str(error)}}, ensure_ascii=False) + "\n")
            print(f"CODEX_ERROR id={question['id']} error={error} trace={directory}", flush=True)
            raise
        finally:
            if thread is not None:
                archived = codex.thread_archive(thread.id)
                trace.write(json.dumps({"archive": {"thread_id": thread.id, "response": trace_json(archived)}}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path, required=True)
    asyncio.run(serve_mcp(parser.parse_args().serve))
