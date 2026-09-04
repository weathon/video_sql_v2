---
name: general-retrieval
description: Use this first for most video memory questions. It explains the full retrieval loop, when to read narrower skills, and when to stop with a final answer.
---

# General Retrieval

## Introduction

Answer from evidence in the SQL database, semantic index, and visual clips. Keep the final answer concise and submit it with `submit_answer`.

## When to use

Use this skill at the start of a question, after a failed search path, or when deciding whether the answer is ready.

## Main content

1. Normalize the question into the entities, actions, objects, and time constraints it asks about. **Reason:** SQL and semantic search need different phrasing; separating constraints prevents a single vague query from controlling the whole rollout.
2. Start broad with `semantic_search` when the wording is visual or approximate; start with `run_sql` when the question names a table-like constraint such as counts, ordering, dates, or durations. **Reason:** semantic search finds candidates, while SQL verifies exact relational claims.
3. Use `sql_qa` for large candidate sets that would flood context, then query exact rows with `run_sql` before finalizing. **Reason:** the summary call can compress rows, but exact row checks keep the answer traceable.
4. Read narrower skills when the task is mainly schema navigation, semantic search, time localization, counting/order, or visual confirmation. **Reason:** the prompt lists skills by description so detailed tactics should be loaded only when relevant.
5. Submit an answer only after the trace has a concrete row, search hit, or visual evidence supporting it. Use `status="unknown"` when the available evidence does not answer the question. **Reason:** a short unknown is better than filling gaps from memory or names.

