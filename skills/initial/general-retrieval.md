---
name: general-retrieval
description: Overall workflow for answering a question from the indexed video memory. Read this first for every question before choosing more specific skills.
---

# General Retrieval

## Introduction

The memory is a DuckDB database built from long videos plus embedding indexes over the same rows. Every row carries `video_id`, `start_s` and `end_s`, so any text hit can be turned into a video interval and inspected visually.

**Reason:** This establishes how indexed text evidence connects to exact locations in the source video.

## When to use

Every question. Use it to decide which of the other skills to read next.

**Reason:** A shared starting workflow helps select specialized guidance without loading every skill.

## Main content

1. Use SQL for exact filtering, joins, ordering, aggregation and counting.
2. Use semantic search to locate likely rows when the exact wording is unknown, then confirm and expand the hit with SQL on the returned `row_id`, `video_id` and time range.
3. Inspect the corresponding video interval with `clip_qa` when the text evidence is ambiguous or the question asks about something only visible in the frames.
4. Record intermediate findings with `add_note` so nested clip agents and later steps can reuse them.
5. Continue until the available evidence supports a concise final answer, then call `submit_answer` once with the final answer. Do not call any other tool after deciding on the answer.

**Reason:** These steps keep retrieval grounded in evidence and give the agent an explicit termination action.
