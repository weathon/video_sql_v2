---
name: general-retrieval
description: Use for planning and coordinating a complete memory-retrieval investigation.
---

# Introduction

Use the searchable memory database to answer the question from evidence rather than guessing.

# When to use

Read this skill when beginning an investigation or when the search has stalled.

# Main content

Start with the most distinctive people, objects, actions, quoted phrases, or outcomes in the question. Use run_sql for known filters, joins, and aggregates. Use semantic_search only when semantic matching is needed; it is not a default or mandatory first step. Inspect candidate evidence, then query neighboring rows and related tables with SQL when needed. semantic_search returns raw evidence; run_sql returns raw results with question=null and optionally requests a separate QA answer when question is supplied. Move between broad summaries and 30-second evidence instead of assuming the first semantic match is correct. Keep notes when an investigation spans several branches. Use visual verification when the answer depends on visible identity, state, order, text, or counting that stored descriptions do not establish precisely. Submit only the final answer requested by the question.
