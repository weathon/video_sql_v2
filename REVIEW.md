# Review Guideline

This file is read and executed directly by the review subagent only. It is not for the main agent: the main agent must not read, summarize, interpret, or relay it. The review subagent must read the complete file itself before every review.

The target is deliberately **anti-perfect research code**. The review must reject changes that make code polished, generalized, defensive, production-ready, or conventionally "better" beyond the experiment's requirements. Rough, direct, purpose-built code is the desired result, not an imperfection to clean up.

## When to review
Before launching any big/expensive run (rollouts, training, large API sweeps).

## Process
1. **Read this file directly and completely.** Do not rely on a summary, quoted checklist, or instructions relayed by the main agent.
2. Review **only the changed lines, not pre-existing code**, in the supplied diff against the checklist below.
3. **Report only** — do NOT edit anything.

## Scope
- Review **only the changed lines** in the diff. Do not flag pre-existing code the diff didn't touch (mark it `original code already like that`).
- When a change preserves a pre-existing user-sanctioned behavior verbatim, mark it `user override`, not a failure.
- When a diff claims to be behavior-preserving (e.g. inlining a helper), **verify behavioral equivalence** against the old code and report any regression.
- **Mitigations apply only to lines the diff did NOT change.** A newly-added or modified line is judged on its merits, full stop. `original code already like that` and `user override` are valid ONLY for untouched pre-existing lines. If the diff introduces a line, it does not matter that the same idiom exists elsewhere — in another file, in the reference implementation, or in a sibling function. Copying a defensive pattern from somewhere else (`x or ""`, `.get(k, default)`, a `try/except` fallback) does NOT make it pre-existing or sanctioned in the new location. "It mirrors the reference" is not a mitigation.
- Review only what is listed in this document plus simple correctness. This is explicitly an **anti-perfect-code review**: do not perform a SaaS-style review, pursue completeness or polish, or recommend conventional best practices outside the experiment's requirements.

## Two axes for every finding

## Checklist
For the changed code only, answer each item. Allowed answers: **no** / **doesn't apply** / **user override** / **original code already like that** / **this is an exception** (use the last sparingly). Only a plain **yes** with none of those mitigations requires fixing before the run (unless the user says otherwise).

0. Is there basic correctness errors in the code? Do NOT consider extream edge cases. 
1. Does it report success/failure from a proxy (metric / gate / exit-0) without checking the real artifact? (e.g. calls a run "success" on a cosine threshold while the gen
erated image is blurred/broken; claims a restart while W&B `resume="allow"` silently continued the old run; computes a metric under wrong preprocessing — unaligned/wrong-resolution — so the number isn't comparable.)
2. Does it rely on assumed semantics instead of the actual source? (implements a spec from a grepped phrase rather than the full text; runs a third-party method on default objective without confirming dodging-vs-impersonation; sweeps/selects on a proxy metric that isn't the true training objective.)
3. Does it silently change the experimental protocol or data? (drops/demotes/reorders samples; adds early stopping, a different batch size, a context-batching scheme, or a self-created goal the user didn't ask for; absolutizes a conclusion — "all failed" — from a single checkpoint/partial sample.)
4. Does the script not including wandb for a training or eval? (wandb is not needed for quick processing but is required for any long running jobs)
5. Is the code over engineered and added functions that are not in the requirement doc?
6. Did it use a default value when something is wrong? (`return results or ""`, `data.get("key", 0)`) — or add any fallback that substitutes a stand-in for a missing dependency (disabling a required service, a mock/placeholder in a data path)?
7. Is the code defensive (try/except fallbacks, validation for impossible cases, `.get` with defaults, swallowing errors)?
8. Is the code production-ready/polished rather than research/toy-style? Any unrequested machinery (resume, manifest hash, atomic writes, completion markers, per-sample JSONL, CLI generalization, whole-`args` passing, extra asserts)?
9. Is there any input/output chunking (truncating/splitting inputs or outputs, a prompt/completion cap)? e.g. `output[:5000]`, `input_msg[:4000]`.
8. Does it downgrade quality for time/cost/memory (thinking off, smaller model, lower setting, a speed optimization that could change results)?
10. Do the LLM sampling params follow the model's OFFICIAL recommendation (model card / generation_config) rather than a forced/ad-hoc value? Flag setting only temperature and leaving top_p/top_k at API defaults, or any value not traceable to the model's docs.
11. Is the edit minimal?
12. Does it seem to make a repo for a SAAS company instead of a research project?
13. Review it against the requirement spec sheet, is there anything NOT the same as the requirement doc? Even if the current apporch might seem better/makes more sense, all the requirement should stick to original design doc strictly. If there is fuzzy/ambitious, ask user. 
14. Did the code/text/prompt/comments leaked user prompt into context? For example, if user asked "no X here" you should not put without X in the comment. Also global design should not be exposed to agent. (eg. do not tell agent the outside env)
15. Is there hard line breaks? 
Example
```
            instructions=(
                "Answer the visual question by sampling the supplied interval. "
                "Use sample_video, crop_frame for fine details, notes when useful, and "
                "submit_clip_answer with evidence_frame_ids from returned images. "
                f"Sampling and crops share {CLIP_IMAGE_LIMIT} images across this entire investigation. "
                "Each image is at most 854 by 480 pixels; crop_frame reads its original resolution. "
                "Do not infer an answered result without visible evidence."
            ),
```

Including
```
            """Crop a sampled frame reloaded from its source video at the recorded PTS.

            Coordinates always refer to the original full frame, even when frame_id
            names an earlier crop. Units are 0-1000; full frame is 0,0,1000,1000.
            Require 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000.
            Each returned crop consumes one image from this clip's shared budget.
            """
```
16. Is the results not judged and logged in real time but judge/upload all at once at the end?

Does not include lists, paragraphes, etc. 

## Principles the review enforces
Anti-perfect research code: offensive programming, direct and purpose-built implementation, crash on the unexpected, no token/output caps, no speed-for-quality or speed-for-correctness downgrades. Correctness above "it runs"; polish, generality, defensive completeness, and production readiness are review failures when the experiment did not require them.

- **Verify before claim.** Success/failure is judged on the real artifact, never a proxy alone.
- **Raise or skip, never silently fall back.** Missing/malformed dependency → **raise** (abort when it invalidates the run) or **skip** the **whole sample** (let resume rerun it). Never substitute a different signal for a missing one.
- **No silent input/output mangling.** No truncating to fit context, no dropping unknown fields, no silent type coercion, no caps beyond a user-sanctioned one.
- **Do exactly what was asked.** No unrequested features/protocol changes.
- **No `.get` with defaults, no `(x or 0)`.** Use `[key]` and `x`; crash if absent.
- **Resumable, disk-backed.** Never hold a dataset's results only in memory; write each sample to disk.
- **Don't catch broad `Exception` to keep a loop alive.** If you don't know the failure, you're hiding it.

## Report format
List only genuine plain-yes violations (or behavioral regressions), each with `file:line`, a one-line reason, and its **Axis B label (TRUST-FATAL / ANNOYING)**. If none, say **"no blocking issues"**. Do not edit anything. Only report what you can point to concrete evidence for — do not manufacture findings, and do not flag a correct idiom as a bug (that causes needless reruns).

