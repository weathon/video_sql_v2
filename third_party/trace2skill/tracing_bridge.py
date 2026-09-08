"""Keep raw structured-parse Chat responses visible in the Weave API call spans."""

import json

import weave
from agents import set_trace_processors
from openai._legacy_response import LegacyAPIResponse
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk
from weave.integrations.openai import openai_sdk
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor


CHAT_OUTPUT_PROCESSOR = openai_sdk.openai_on_finish_post_processor


def chat_accumulator(acc, value, skip_last=False, stream_start_time=None):
    # Weave's accumulator consumes only tool_calls[0] from each chunk. Use the
    # SDK accumulator to retain every parallel call and reasoning_content delta.
    payload = value.model_dump(exclude_none=True)
    if acc is not None:
        snapshot = acc.current_completion_snapshot
        for choice in payload["choices"]:
            if choice["index"] >= len(snapshot.choices):
                continue
            previous = snapshot.choices[choice["index"]].message
            delta = choice["delta"]
            # DeepInfra repeats metadata and sends nulls in later chunks. These
            # are not text deltas; leave accumulated metadata intact.
            if "role" in delta and delta["role"] == previous.role:
                del delta["role"]
            if "tool_calls" in delta and previous.tool_calls is not None:
                for tool in delta["tool_calls"]:
                    if tool["index"] >= len(previous.tool_calls):
                        continue
                    prior = previous.tool_calls[tool["index"]]
                    for key in ("id", "type"):
                        if key in tool and tool[key] == getattr(prior, key):
                            del tool[key]
                    if "function" in tool and "name" in tool["function"] and tool["function"]["name"] == prior.function.name:
                        del tool["function"]["name"]
    else:
        acc = ChatCompletionStreamState()
        openai_sdk.init_stream_tracking(acc, stream_start_time)
    acc.handle_chunk(ChatCompletionChunk.model_validate(payload))
    openai_sdk.calculate_time_to_first_token(acc, value, openai_sdk._openai_chunk_has_content)
    return acc


def response_output(value):
    # Save raw output without triggering schema validation inside Weave. The caller
    # retains the actual response and supplies validation errors to the model.
    if isinstance(value, LegacyAPIResponse):
        return json.loads(value.content)
    if isinstance(value, ChatCompletionStreamState):
        payload = value.current_completion_snapshot.model_dump(exclude_unset=True, exclude_none=True)
        return openai_sdk.add_time_to_first_token_to_dict(payload, value)
    return CHAT_OUTPUT_PROCESSOR(value)


def init_weave():
    openai_sdk.openai_accumulator = chat_accumulator
    openai_sdk.openai_on_finish_post_processor = response_output
    client = weave.init("3dsmile/video-sql-v2")
    set_trace_processors([WeaveTracingProcessor()])
    return client
