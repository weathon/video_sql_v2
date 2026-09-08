"""Keep raw structured-parse Responses visible in the Weave API call spans."""

import json

import weave
from agents import set_trace_processors
from openai._legacy_response import LegacyAPIResponse
from weave.integrations.openai import openai_sdk
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor


RESPONSES_OUTPUT_PROCESSOR = openai_sdk.responses_on_finish_post_processor


def response_output(value):
    # Save raw output without triggering schema validation inside Weave. The caller
    # retains the actual response and supplies validation errors to the model.
    if isinstance(value, LegacyAPIResponse):
        return json.loads(value.content)
    return RESPONSES_OUTPUT_PROCESSOR(value)


def init_weave():
    openai_sdk.responses_on_finish_post_processor = response_output
    client = weave.init("3dsmile/video-sql-v2")
    set_trace_processors([WeaveTracingProcessor()])
    return client
