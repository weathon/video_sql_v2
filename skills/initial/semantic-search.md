---
name: semantic-search
description: Use this when exact keywords may miss relevant moments, when translating a natural-language clue into candidate rows, or when broad recall is needed before SQL verification.
---

# Semantic Search

## Introduction

`semantic_search` embeds the query with Qwen3-VL and searches one indexed table. It returns row IDs and scores; use SQL afterward to inspect exact content.

## When to use

Use semantic search for visual descriptions, paraphrases, activities, object states, or uncertain wording.

## Main content

1. Query the table that matches the evidence type: `episodes` for captions, `event_steps` for actions, `topic_facts` for summarized facts, `object_states` for object/status pairs, and `visual_clips` for video similarity. **Reason:** each table embeds a different unit of memory.
2. Search in rank windows when broad results are needed: after checking top 10, ask for ranks 10 to 20 instead of repeating top 20. **Reason:** non-overlapping windows save context and avoid rereading the same candidates.
3. Apply `video_id` when the question names `day`, `week`, or `month`. **Reason:** the three logical timelines are independent.
4. Treat scores as candidate ranking, not proof. Read the actual row text and, for visual claims, call `clip_qa`. **Reason:** a high score can come from similar wording without the needed fact.

