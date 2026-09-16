"""Serve the Week harness tools over MCP stdio without an agent loop."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import httpx
from agents import ModelSettings, OpenAIResponsesModel, RunConfig, ToolOutputText
from agents.tool_context import ToolContext
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from src.harness import AgentRaised, HarnessContext, REQUEST_TRACE, make_tools, stripped_json


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/raid/videosql-video-artifacts")
MODEL = "Qwen/Qwen3.8-27B-FP8"


class MCPHarnessContext(HarnessContext):
    def reserve_model_call(self, role: str) -> None:
        self.investigation_calls += 1
        print(f"QA_CALL role={role} count={self.investigation_calls}", file=sys.stderr, flush=True)


async def record_wire_request(request):
    info = REQUEST_TRACE.get()
    path = Path(info["trace_dir"]) / f"wire-{uuid.uuid4().hex}.json"
    record = {"url": str(request.url), "trace_file": info["trace_file"], "agent": info["agent"], "body": stripped_json(json.loads(request.content))}
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")


async def main():
    directory = ROOT / "runs/mcp" / uuid.uuid4().hex
    qa_url = os.environ["VIDEO_SQL_QA_URL"]
    embedding_url = os.environ["VIDEO_SQL_EMBEDDING_URL"]
    async with stdio_server() as (read_stream, write_stream):
        async with AsyncOpenAI(api_key="EMPTY", base_url=qa_url, timeout=None, max_retries=0, http_client=DefaultAsyncHttpxClient(event_hooks={"request": [record_wire_request]})) as client, httpx.AsyncClient(timeout=httpx.Timeout(connect=60, read=100000, write=100000, pool=100000)) as embedding_client:
            for url, expected in ((qa_url, MODEL), (embedding_url, "qwen3-vl-embedding-8b")):
                print(f"CHECK_MODEL url={url} expected={expected}", file=sys.stderr, flush=True)
                response = await embedding_client.get(url.rstrip("/") + "/models", timeout=10)
                response.raise_for_status()
                if expected not in [item["id"] for item in response.json()["data"]]:
                    raise RuntimeError(f"Unexpected model at {url}: {response.text}")
            context = MCPHarnessContext(database=ARTIFACTS / "index.duckdb", embedding_index=ARTIFACTS / "embeddings", media_map=ARTIFACTS / "media-map.json", skill_dir=ROOT / "training_free/skills-vector", run_dir=directory, model=OpenAIResponsesModel(model=MODEL, openai_client=client), client=client, model_settings=ModelSettings(reasoning={"summary": "auto"}, store=False, response_include=["reasoning.encrypted_content"], preserve_raw_usage=True), run_config=RunConfig(tracing_disabled=True, reasoning_item_id_policy="preserve"), embedding_url=embedding_url, embedding_model="qwen3-vl-embedding-8b", embedding_client=embedding_client, query_time=None, model_limiter=asyncio.Semaphore(10), semantic_search_mode="vector", clip_qa_video=True)
            functions, _ = make_tools(context)
            tools = {tool.name: tool for tool in functions}
            print(f"MCP_READY traces={directory}", file=sys.stderr, flush=True)

            async def list_tools(request_context, params):
                return types.ListToolsResult(tools=[types.Tool(name=tool.name, description=tool.description, input_schema=tool.params_json_schema) for tool in tools.values()])

            async def call_tool(request_context, params):
                tool = tools[params.name]
                context.tool_counter += 1
                tool_id = context.tool_counter
                call_id = uuid.uuid4().hex
                arguments = json.dumps({} if params.arguments is None else params.arguments, ensure_ascii=False)
                path = directory / "traces" / f"tool-{call_id}.json"
                record = {"tool_id": tool_id, "call_id": call_id, "name": params.name, "arguments": arguments, "output": None, "error": None}
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"TOOL_START name={params.name} arguments={arguments} trace={path}", file=sys.stderr, flush=True)
                is_error = False
                try:
                    output = await tool.on_invoke_tool(ToolContext(context=context, tool_name=params.name, tool_call_id=call_id, tool_arguments=arguments), arguments)
                except AgentRaised as error:
                    record["error"] = {"type": type(error).__name__, "message": str(error)}
                    output = json.dumps({"error": str(error)}, ensure_ascii=False)
                    is_error = True
                except BaseException as error:
                    record["error"] = {"type": type(error).__name__, "message": str(error)}
                    print(f"TOOL_ERROR name={params.name} error={type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    raise
                finally:
                    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                record["output"] = stripped_json(output)
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                header = json.dumps({"tool_id": tool_id}) + "\n"
                if isinstance(output, str):
                    content = [types.TextContent(type="text", text=header + output)]
                else:
                    content = [types.TextContent(type="text", text=header)]
                    for part in output:
                        if isinstance(part, ToolOutputText):
                            content.append(types.TextContent(type="text", text=part.text))
                        else:
                            content.append(types.ImageContent(type="image", data=part.image_url.split(",", 1)[1], mime_type="image/jpeg"))
                print(f"TOOL_DONE name={params.name} tool_id={tool_id} is_error={is_error}", file=sys.stderr, flush=True)
                return types.CallToolResult(content=content, is_error=is_error)

            server = Server("video_sql", instructions=(ROOT / "src/prompt.md").read_text(), on_list_tools=list_tools, on_call_tool=call_tool)
            await server.run(read_stream, write_stream, server.create_initialization_options(), raise_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
