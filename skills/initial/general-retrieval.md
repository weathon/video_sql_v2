---
name: general-retrieval
description: Use at the start of any video-memory question to choose evidence sources and plan the retrieval loop.
---

# Introduction

Use a hypothesis-driven loop: identify what the answer must establish, retrieve candidate evidence, test competing interpretations, and stop only when the final answer is supported.

# When to use

Read this skill first for every question, then read the narrower skills that match the required operation.

# Main content

1. Identify the logical timeline from the prompt and keep every query restricted to that `video_id`. **Reason:** Day, Week, and Month are independent timelines whose seconds are not interchangeable.
2. Translate each clause of the question into an evidence requirement. Search broad summaries for orientation, then descend to 30-second rows and visual clips for decisive facts. **Reason:** High-level summaries locate candidates but can omit or merge details.
3. Keep at least two plausible interpretations when wording is ambiguous. Use SQL, semantic retrieval, and video inspection to discriminate them. **Reason:** A plausible first hit is not proof that alternatives are false.
4. Use `add_note` for intermediate facts with row IDs and times, then reread notes before answering. **Reason:** Multi-hop questions require stable bookkeeping across tool calls.
5. Call `submit_answer` with the concise semantic answer after the evidence requirements are satisfied. **Reason:** Identifiers are evidence aids, not part of the evaluated answer unless the question asks for them.
