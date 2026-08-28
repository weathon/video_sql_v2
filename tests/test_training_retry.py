from agents.exceptions import ModelBehaviorError, UserError

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
