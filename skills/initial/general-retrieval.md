---
name: general-retrieval
description: Plan and execute the complete video-memory retrieval loop. Read this at the start of every question to choose the relevant SQL, semantic-search, time, counting, and visual-verification skills.
---

# General Retrieval

## Introduction

Treat the answer as the end of an evidence chain. Identify the requested fact, time scope, entities, and operation before choosing tools, then preserve the row IDs, intervals, and source facts that support the answer.

**Reason:** An explicit evidence chain keeps plausible recollection separate from facts retrieved from the indexed memory.

## When to use

Use this skill for every question. Read the specialized skills whose descriptions match the question before performing their parts of the search.

**Reason:** The catalog stays small while detailed instructions are loaded only when relevant.

## Main content

1. Classify the question into lookup, temporal localization, counting or ordering, and visual confirmation requirements.
2. Use semantic search to discover candidate rows when the wording is uncertain; use SQL to filter, join, aggregate, and order exact candidates.
3. Fetch the actual database rows behind semantic hits by `row_id`; never infer content from a score or index position.
4. Use `sql_qa` when the SQL result should be interpreted by its dedicated single LLM call instead of being returned raw to this agent.
5. Use `clip_qa` when the answer depends on appearance, action, state, sequence, or another fact that text rows do not establish.
6. Revisit the search when sources conflict or the evidence covers only part of the question. Submit only the concise answer supported by the final evidence chain.

**Reason:** This ordering separates candidate discovery, exact computation, and source verification so each claim remains traceable.
