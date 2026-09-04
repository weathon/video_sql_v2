# Reflection Agent Split Incident

## Status

Contained. The affected formal run was stopped and its existing artifacts were preserved only for debugging.

## Incident

The training pipeline implemented rollout reflection by creating a new reflection agent and passing it a serialized copy of the original rollout trace. The intended behavior was for the original rollout agent to continue in its own conversation after answering, receive the reference answer, score, and reflection skill, and then reflect on its own work.

These behaviors are not equivalent. The implemented system measured a second agent's critique of another agent's trace rather than the original agent's self-reflection.

## Impact

The affected run completed two batches containing 40 rollouts before it was stopped. Their rollout answers and judge scores remain useful only as debugging artifacts. Their reflections, question summaries, skill updates, and all later behavior conditioned on those skill updates are invalid for the intended experiment and must not be used as experimental results.

## Root cause

The implementation incorrectly treated separation between retrieval and reflection phases as requiring separation between agents. A new agent made it convenient to inject the reference answer and score, change the available tools, apply reflection-specific instructions, and manage reflection context independently. That implementation convenience overrode the experimental requirement that the same agent reflect within its existing conversation state.

The distinction between a separate reflection phase and a separate reflection agent was not checked explicitly before the formal run.

## Correct behavior

After the original rollout agent submits its answer, the pipeline must preserve that agent's complete conversation state. It must then append the reference answer, score, and reflection skill to the same conversation and instruct the same agent to reflect. The reflection must be produced from this continued conversation rather than from a serialized trace supplied to a newly created agent.

The later cross-rollout summary remains a separate model call because it combines reflections from multiple rollouts rather than performing an individual rollout's self-reflection.

## Corrective action

Remove the standalone rollout reflection agent and the serialized-trace handoff. Continue each original rollout conversation into its reflection phase, expose the reflection submission tool in that continuation, and preserve the existing separation that prevents the reference answer from being shown before the rollout answer is submitted.

Discard the affected run from experimental evaluation, start the corrected formal run in a new run directory, and verify from its real trace that the answer and reflection occur in one continuous conversation before accepting any results.
