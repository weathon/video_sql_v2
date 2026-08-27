# Skip or crash

Choose based on the scope of invalidity, not on whether an exception is inconvenient.

Skip the whole sample when the failure is local to one independent sample and does not change the inputs, ordering, or results of other samples. Record the stable row ID, source path, and exact error. Do not create a substitute result or a completed shard. A resumed run may attempt the missing sample again.

Crash when the failure can invalidate more than one sample or the run configuration: a missing service or dataset, database or schema mismatch, shard row mismatch, invalid model response indexes, deterministic code error, or an unavailable dependency.

For concurrent work, catch only the classified sample-local exception inside that sample's task. Unknown exceptions must still propagate and terminate the run.

The 2026-08-26 video embedding run established one project-specific case: an independent one-clip video embedding request returning HTTP 500 with `exception_type=RuntimeError` and `detail=Requested next frame while there are no more frames left to decode.` is a sample-local TorchCodec failure. Skip that whole clip, leave its embedding shard absent, and continue other clips. A local vLLM middleware exposes the exception type and message so no other HTTP 500 is classified as this case. Every other HTTP error and all non-HTTP errors still crash.
