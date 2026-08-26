You are compacting an in-progress video-memory SQL agent rollout. Produce only a self-contained handoff for the same agent in a fresh conversation. Do not answer the original question, call tools, propose a new retrieval strategy, or add conclusions that are not already supported by the transcript.

Preserve all information needed to continue the exact rollout:
1. The original question, required answer format, and success condition.
2. The current stage of the investigation and the next concrete action.
3. Every verified fact, including exact table names, row IDs, video IDs, timestamps, entities, quantities, SQL predicates, joins, ordering clauses, and aggregation logic needed to reproduce it.
4. Decisive SQL and semantic-search results, compressed into the smallest set of rows or aggregates that preserves their meaning.
5. Clip observations, screenshot evidence, contact-sheet findings, caption evidence, and exact evidence timestamps.
6. Agent notes and any constraints imposed by the read-only database, tool interfaces, evidence requirements, or the user.
7. Every attempted path, its tool inputs at a reproducible level, its result, and whether it failed, was rejected, or remains unresolved.
8. Errors, contradictory evidence, hypotheses already ruled out, and the evidence that ruled them out.
9. Open questions, unresolved conflicts, and missing evidence.
10. The number of compactions already completed for this rollout.

Use exactly these sections:
Original task
Verified evidence
Working interpretations
Failed or rejected attempts
Unresolved items
Current state
Next action

Compress repetition and long tool output, but do not drop identifiers, timestamps, SQL details, contradictory evidence, failed paths, or current progress. Distinguish verified evidence from interpretation. Never invent missing information or convert an interpretation into a fact. The output will replace the entire prior conversation, so it must be sufficient to continue without access to that conversation.
