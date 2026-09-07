"""Keep raw structured-parse Responses visible in the Weave API call spans."""

import json

import weave
from agents import set_trace_processors
from openai._legacy_response import LegacyAPIResponse
from weave.integrations.openai import openai_sdk
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor


RESPONSE_OUTPUT_PROCESSOR = openai_sdk.responses_on_finish_post_processor


def response_output(value):
    # The SDK raw-response wrapper intentionally defers structured validation. Weave's
    # default Responses serializer calls model_dump(), which this wrapper does not have.
    if isinstance(value, LegacyAPIResponse):
        return json.loads(value.content)
    return RESPONSE_OUTPUT_PROCESSOR(value)


def init_weave():
    openai_sdk.responses_on_finish_post_processor = response_output
    client = weave.init("3dsmile/video-sql-v2")
    set_trace_processors([WeaveTracingProcessor()])
    return client
