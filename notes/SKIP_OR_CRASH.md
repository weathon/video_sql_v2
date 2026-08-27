# Skip or crash

Choose based on the scope of invalidity, not on whether an exception is inconvenient.

Skip the whole sample when the failure is local to one independent sample and does not change the inputs, ordering, or results of other samples. Record the stable row ID, source path, and exact error. Do not create a substitute result or a completed shard. A resumed run may attempt the missing sample again.

Crash when the failure can invalidate more than one sample or the run configuration: a missing service or dataset, database or schema mismatch, shard row mismatch, invalid model response indexes, deterministic code error, or an unavailable dependency.

For concurrent work, catch only the classified sample-local exception inside that sample's task. Unknown exceptions must still propagate and terminate the run.

The 2026-08-26 video embedding run established one project-specific case: an independent one-clip video embedding request returning HTTP 500 because TorchCodec requested a frame past the end of that clip is a sample-local failure. Skip that whole clip, leave its embedding shard absent, and continue other clips. Non-500 HTTP errors and all non-HTTP errors still crash.
