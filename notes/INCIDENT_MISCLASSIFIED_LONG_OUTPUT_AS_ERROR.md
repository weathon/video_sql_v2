# Incident: misclassified long model output as an error

## Date
2026-08-28

## Summary
During the Qwen3.8 Flash Next indexing probe, I treated a long-running generation as a failed request even though the server continued reporting one running request, positive generation throughput, and no API or model error. I cancelled the probe requests, stopped the Qwen service, and incorrectly made a fixed output-token limit a prerequisite for starting indexing. The user had not requested a fixed budget, and long output by itself was not an error.

## What went wrong
1. I equated the absence of a quick final response with failure instead of classifying the request from the actual server state.
2. The server trace showed active generation, nonzero throughput, no waiting requests, and no error response, but I still described the generation as abnormal and cancelled it.
3. The rendered prompt confirmed that `reasoning_effort=low` reached the model. I incorrectly treated the fact that low effort remained long as evidence that the request could not be allowed to finish.
4. I introduced a new requirement for a request-level output limit even though the user had not asked for one and the project explicitly avoids completion caps.
5. I stopped the correctly configured Qwen service before any formal indexing request failed, timed out, exhausted retries, or produced an invalid artifact.

## Impact
Formal indexing did not start. Two probe requests were cancelled while the server was still generating, the Qwen service was stopped and must be restarted, and the user was asked to make an unnecessary output-budget decision. This delayed the requested Stage 3 indexing work.

## Root cause
I misclassified duration and output length as an error condition. I ignored the distinction between an active long generation and an actual failure such as an API error, server termination, exhausted retry, schema failure, or invalid result. The user had specifically warned that this model can produce long repetitive reasoning, but I treated the warned behavior as a reason to terminate rather than as expected model behavior to observe without changing the experiment.

## Corrective actions
1. Do not classify a request as failed solely because it runs for a long time or generates many tokens.
2. While the server reports an active request with positive throughput and no error, allow the request to continue until the model emits its own end marker or reaches the existing server-side context limit.
3. Apply the configured retries only after a real API, service, parsing, validation, or result failure; elapsed time alone is not one of those failures.
4. Do not add or request a completion-token cap unless the user explicitly chooses one.
5. Do not stop the model service without a concrete failure signal or an explicit user instruction.

## Files touched
This report adds `notes/INCIDENT_MISCLASSIFIED_LONG_OUTPUT_AS_ERROR.md`. `CODEX_PLAN.md` records the corrected interpretation and report path.
