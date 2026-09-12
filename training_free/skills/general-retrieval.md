---
name: general-retrieval
description: Use for planning and coordinating a complete memory-retrieval investigation.
---

# Introduction

Use the searchable memory database to answer the question from evidence rather than guessing.

# When to use

Read this skill when beginning an investigation or when the search has stalled.

# Main content

Start with the most distinctive people, objects, actions, quoted phrases, or outcomes in the question. Prefer sql_ppr_search to combine semantic and graph retrieval with SQL. Usually choose one SQL stage: pre_sql imposes a known hard scope before PPR (for example yesterday), whereas post_sql selects, orders, or aggregates the useful PPR hits after retrieval. These solve different problems; do not automatically use both. Both remain available when necessary. Use original run_sql for direct database questions or follow-up joins, and semantic_search for a specific embedded table. With question=null, semantic_search and run_sql return raw results; supplying question optionally requests a separate QA answer. Move between broad summaries and 30-second evidence instead of assuming the first semantic match is correct. Keep notes when an investigation spans several branches. Use visual verification when the answer depends on visible identity, state, order, text, or counting that stored descriptions do not establish precisely. Submit only the final answer requested by the question.
