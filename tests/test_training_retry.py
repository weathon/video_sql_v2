from agents.exceptions import ModelBehaviorError, UserError
import httpx
from openai import APIError, BadRequestError

from src.training import is_retryable_rollout_error


def test_only_retryable_causes_inside_user_error_are_retried():
    transient = ModelBehaviorError("temporary model failure")
    wrapped_transient = UserError("clip tool failed")
    wrapped_transient.__cause__ = transient
    assert is_retryable_rollout_error(transient) is True
    assert is_retryable_rollout_error(wrapped_transient) is True

    deterministic = RuntimeError("missing video")
    wrapped_deterministic = UserError("clip tool failed")
    wrapped_deterministic.__cause__ = deterministic
    assert is_retryable_rollout_error(wrapped_deterministic) is False

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    stream_error = APIError("stream failed", request, body=None)
    wrapped_stream_error = UserError("clip tool failed")
    wrapped_stream_error.__cause__ = stream_error
    assert is_retryable_rollout_error(stream_error) is True
    assert is_retryable_rollout_error(wrapped_stream_error) is True

    bad_request = BadRequestError("invalid request", response=httpx.Response(400, request=request), body=None)
    wrapped_bad_request = UserError("clip tool failed")
    wrapped_bad_request.__cause__ = bad_request
    assert is_retryable_rollout_error(bad_request) is False
    assert is_retryable_rollout_error(wrapped_bad_request) is False
