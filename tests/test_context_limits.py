import asyncio
import copy
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import duckdb
import numpy as np
import pytest
from agents.tool_context import ToolContext
from openai.types.responses import Response, ResponseCompletedEvent, ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText, ResponseUsage

from src import harness as module
from src.harness import SEARCH_TABLES, VideoSQLHarness


@pytest.fixture
def harness(tmp_path):
    database = tmp_path / "index.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE event_chains (row_id VARCHAR, name VARCHAR, start_s DOUBLE)")
        connection.executemany("INSERT INTO event_chains VALUES (?, ?, ?)", [("a", "package", 0), ("b", "quiz", 60), ("c", "auction", 120)])
    index = tmp_path / "embedding_index"
    index.mkdir()
    for table in SEARCH_TABLES:
        np.save(index / f"{table}.row_ids.npy", np.array([], dtype=str))
        np.save(index / f"{table}.vectors.npy", np.empty((0, 4096), dtype=np.float32))
    media = tmp_path / "media-map.json"
    media.write_text('{"sources": []}')
    instance = VideoSQLHarness(database=database, embedding_index=index, media_map=media, skill_dir=Path(__file__).resolve().parents[1] / "skills/initial", run_dir=tmp_path / "run", model_name="unit-test", api_key="unit-test", base_url="http://127.0.0.1:1/v1", embedding_url="http://127.0.0.1:1/v1", embedding_model="unit-test")
    instance.context.run_config.tracing_disabled = True
    instance.context.client.responses.input_tokens.count = AsyncMock(return_value=SimpleNamespace(input_tokens=100))
    yield instance
    asyncio.run(instance.close())


@pytest.fixture
def scripted_model(harness, monkeypatch):
    responses = []
    requests = []

    async def stream_response(instructions, input, model_settings, tools, output_schema, *args, **kwargs):
        requests.append({"instructions": instructions, "input": copy.deepcopy(input), "settings": model_settings, "tools": [tool.name for tool in tools], "schema": output_schema})
        output, input_tokens, output_tokens = responses.pop(0)
        usage = ResponseUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens, input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0}, output_tokens_details={"reasoning_tokens": 0})
        response = Response.model_construct(id=f"resp_{len(requests)}", created_at=0, model="unit-test", object="response", status="completed", output=output, usage=usage)
        yield ResponseCompletedEvent(type="response.completed", sequence_number=0, response=response)

    monkeypatch.setattr(harness.context.model, "stream_response", stream_response)
    return responses, requests


@pytest.mark.parametrize("question", [None, "Summarize all events"])
@pytest.mark.parametrize("sql", [
    "SELECT * FROM event_chains",
    "SELECT * FROM event_chains ORDER BY start_s",
    "SELECT * FROM event_chains LIMIT ALL",
    "SELECT * FROM event_chains LIMIT -1",
    "SELECT * FROM event_chains WHERE row_id = 'a'; SELECT * FROM event_chains LIMIT 1",
    "SELECT * FROM event_chains, (SELECT 1 WHERE TRUE) AS unrelated",
    "SELECT name, COUNT(*) FROM event_chains GROUP BY name",
])
def test_unscoped_sql_rejected_before_database_or_qa(harness, monkeypatch, sql, question):
    execute = Mock(side_effect=AssertionError("Unscoped SQL reached the database"))
    summarize = AsyncMock(side_effect=AssertionError("Unscoped SQL reached QA"))
    monkeypatch.setattr(module, "execute_sql", execute)
    monkeypatch.setattr(harness.context, "summary_response", summarize)
    tool = next(tool for tool in harness.answer_tools if tool.name == "run_sql")
    arguments = json.dumps({"sql": sql, "question": question})
    context = ToolContext(context=harness.context, tool_name="run_sql", tool_call_id="sql", tool_arguments=arguments)
    result = json.loads(asyncio.run(tool.on_invoke_tool(context, arguments)))
    assert "error" in result
    assert "rewrite" in result["error"].lower()
    execute.assert_not_called()
    summarize.assert_not_awaited()


@pytest.mark.parametrize("sql, rows", [
    ("SELECT row_id FROM event_chains ORDER BY start_s LIMIT 2", [{"row_id": "a"}, {"row_id": "b"}]),
    ("SELECT row_id FROM event_chains WHERE start_s >= 60 AND start_s < 120", [{"row_id": "b"}]),
    ("SELECT COUNT(*) AS n FROM event_chains", [{"n": 3}]),
    ("SELECT MIN(start_s) AS first, MAX(start_s) AS last FROM event_chains", [{"first": 0.0, "last": 120.0}]),
    ("SELECT name, COUNT(*) AS n FROM event_chains GROUP BY name HAVING name = 'quiz'", [{"name": "quiz", "n": 1}]),
    ("SELECT row_id FROM event_chains QUALIFY ROW_NUMBER() OVER (ORDER BY start_s) = 1", [{"row_id": "a"}]),
])
def test_scoped_sql_returns_real_database_rows(harness, sql, rows):
    tool = next(tool for tool in harness.answer_tools if tool.name == "run_sql")
    arguments = json.dumps({"sql": sql, "question": None})
    context = ToolContext(context=harness.context, tool_name="run_sql", tool_call_id="sql", tool_arguments=arguments)
    result = json.loads(asyncio.run(tool.on_invoke_tool(context, arguments)))
    assert result["rows"] == rows
    assert result["has_more"] is False


@pytest.mark.parametrize("tokens, forced", [(127999, False), (128000, True), (128001, True)])
def test_context_boundary_before_model_call(harness, scripted_model, tokens, forced):
    responses, requests = scripted_model
    count = harness.context.client.responses.input_tokens.count
    count.return_value = SimpleNamespace(input_tokens=tokens)
    if not forced:
        output = ResponseFunctionToolCall(type="function_call", name="submit_answer", call_id="answer", arguments='{"answer":"package, quiz, auction"}')
    else:
        output = ResponseOutputMessage(type="message", id="final", role="assistant", status="completed", content=[ResponseOutputText(type="output_text", text='{"answer":"package, quiz, auction"}', annotations=[])])
    responses.append(([output], 100, 20))
    result = asyncio.run(harness.answer("Order the events"))
    assert result.answer == "package, quiz, auction"
    assert len(requests) == 1
    assert harness.context.answer_context_reached == forced
    if forced:
        assert requests[0]["tools"] == []
        assert requests[0]["settings"].tool_choice == "none"
        assert requests[0]["schema"] is not None
        assert harness.context.tool_calls == {}
        events = [json.loads(line) for path in (harness.context.run_dir / "traces").glob("*events*.jsonl") for line in path.read_text().splitlines()]
        assert any(event["type"] == "answer_context_limit" and event["tokens"] == tokens for event in events)
    else:
        assert "run_sql" in requests[0]["tools"]
        assert requests[0]["schema"] is None
    assert "run_sql" in [tool["name"] for tool in count.call_args.kwargs["tools"]]
    assert count.call_args.kwargs["instructions"]
    assert count.call_args.kwargs["reasoning"] == harness.context.model_settings.reasoning


@pytest.mark.parametrize("trigger", ["response_usage", "tool_result"])
def test_context_crossing_stops_investigation_and_preserves_history(harness, scripted_model, monkeypatch, trigger):
    responses, requests = scripted_model
    execute = Mock(wraps=module.execute_sql)
    monkeypatch.setattr(module, "execute_sql", execute)
    count = harness.context.client.responses.input_tokens.count
    count.side_effect = [SimpleNamespace(input_tokens=100), SimpleNamespace(input_tokens=128000)]
    call = ResponseFunctionToolCall(type="function_call", name="run_sql", call_id="lookup", arguments='{"sql":"SELECT row_id FROM event_chains LIMIT 1","question":null}')
    responses.append(([call], 127999 if trigger == "response_usage" else 100, 1))
    final = ResponseOutputMessage(type="message", id="final", role="assistant", status="completed", content=[ResponseOutputText(type="output_text", text='{"answer":"package"}', annotations=[])])
    responses.append(([final], 128000, 20))
    result = asyncio.run(harness.answer("What happened first?"))
    assert result.answer == "package"
    assert len(requests) == 2
    assert requests[-1]["tools"] == []
    assert requests[-1]["settings"].tool_choice == "none"
    history = requests[-1]["input"]
    calls = [item for item in history if item.get("type") == "function_call"]
    outputs = [item for item in history if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in calls] == [item["call_id"] for item in outputs] == ["lookup"]
    returned = json.loads(outputs[0]["output"].partition("\n")[2])
    if trigger == "response_usage":
        execute.assert_not_called()
        assert "not executed" in returned["error"]
    else:
        execute.assert_called_once()
        assert returned["rows"] == [{"row_id": "a"}]
    assert harness.context.answer_context_reached is True
    assert harness.context.compactions == 0


def test_small_context_can_continue_past_twelve_turns(harness, scripted_model):
    responses, requests = scripted_model
    for turn in range(13):
        call = ResponseFunctionToolCall(type="function_call", name="run_sql", call_id=f"lookup-{turn}", arguments=json.dumps({"sql": f"SELECT row_id FROM event_chains LIMIT 1 OFFSET {turn % 3}", "question": None}))
        responses.append(([call], 100, 20))
    responses.append(([ResponseFunctionToolCall(type="function_call", name="submit_answer", call_id="answer", arguments='{"answer":"package, quiz, auction"}')], 100, 20))
    result = asyncio.run(harness.answer("Order the events"))
    assert result.answer == "package, quiz, auction"
    assert len(requests) == harness.context.answer_turns == 14
    assert all("run_sql" in request["tools"] for request in requests)
    assert harness.context.answer_context_reached is False
    trace = json.loads(Path(result.trace_files[0]).read_text())
    assert trace["run_config"]["max_turns"] is None
    assert len(trace["tool_calls"]) == 14


def test_token_count_uses_replacement_without_removed_reasoning(harness, scripted_model):
    responses, requests = scripted_model
    original = {"type": "function_call", "call_id": "lookup", "name": "run_sql", "arguments": '{"sql":"SELECT row_id FROM event_chains LIMIT 1","question":null}'}
    replacement = {"type": "function_call", "call_id": "replace", "name": "replace_tool", "arguments": json.dumps({"tool_id": 1, "replacement": "First event: package (row a)", "reason": "Keep only relevant evidence"})}
    history = [{"role": "user", "content": "What happened first?"}, original, {"type": "function_call_output", "call_id": "lookup", "output": '{"tool_id":1}\n{"rows":[{"row_id":"a","name":"package"}]}'}, {"type": "reasoning", "id": "removed-reasoning", "summary": []}, replacement, {"type": "function_call_output", "call_id": "replace", "output": '{"tool_id":2}\n{"replaced_tool_id":1}'}]
    responses.append(([ResponseFunctionToolCall(type="function_call", name="submit_answer", call_id="answer", arguments='{"answer":"package"}')], 100, 20))
    result = asyncio.run(harness.answer("What happened first?", input_items=history))
    assert result.answer == "package"
    counted = harness.context.client.responses.input_tokens.count.call_args.kwargs["input"]
    assert counted == requests[0]["input"]
    assert len(counted) == 3
    assert counted[1] == original
    assert counted[2]["output"] == '{"tool_id": 1}\nReplaced with:\nFirst event: package (row a)\nReason: Keep only relevant evidence'
    assert "removed-reasoning" not in json.dumps(counted)
    assert not any(item.get("call_id") == "replace" for item in counted)


def test_token_count_failure_stops_before_model_generation(harness, scripted_model):
    responses, requests = scripted_model
    harness.context.client.responses.input_tokens.count.side_effect = RuntimeError("tokenizer unavailable")
    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        asyncio.run(harness.answer("What happened first?"))
    assert requests == []
    assert harness.context.tool_calls == {}
    trace = json.loads((harness.context.run_dir / "traces/answer-segment-00.json").read_text())
    assert trace["error"] == {"type": "RuntimeError", "message": "tokenizer unavailable"}


def test_mtp_context_length_matches_main_model(monkeypatch):
    execute = Mock()
    monkeypatch.setattr(sys, "argv", ["trace2skill_service.py", "pd"])
    monkeypatch.setattr(os, "execvp", execute)
    monkeypatch.setattr(Path, "mkdir", Mock())
    monkeypatch.setattr(Path, "write_text", Mock())
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/trace2skill_service.py"))
    execute.assert_called_once()
    command = execute.call_args.args[1]
    main_length = int(command[command.index("--max-model-len") + 1])
    mtp = json.loads(command[command.index("--speculative-config") + 1])
    assert mtp["method"] == "mtp"
    assert mtp["max_model_len"] == main_length
    assert command[command.index("--max-num-seqs") + 1] == "30"
