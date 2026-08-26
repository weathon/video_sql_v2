# Review Guideline (global)

The single standard every pre-run code review must follow. Derived from CLAUDE.md. Whenever CLAUDE.md says to review changes before a run, follow this file exactly.

## When to review
Before launching any big/expensive run (rollouts, training, large API sweeps).

## Process
1. **Commit first.** Commit the current changes before each run. The review is a `git diff` against the previous commit, so each run's review covers exactly what changed since the last run.
2. **Spawn a subagent to review the diff.** The main agent spawns a subagent that runs `git diff <last commit>` and reviews **only the changed lines, not pre-existing code**, against the checklist below.
3. The subagent **reports only** — it does NOT edit anything.
4. Fix every plain-**yes** item, re-commit, re-review. Only then launch.

## Scope
- Review **only the changed lines** in the diff. Do not flag pre-existing code the diff didn't touch (mark it `original code already like that`).
- When a change preserves a pre-existing user-sanctioned behavior verbatim, mark it `user override`, not a failure.
- When a diff claims to be behavior-preserving (e.g. inlining a helper), **verify behavioral equivalence** against the old code and report any regression.
- **Mitigations apply only to lines the diff did NOT change.** A newly-added or modified line is judged on its merits, full stop. `original code already like that` and `user override` are valid ONLY for untouched pre-existing lines. If the diff introduces a line, it does not matter that the same idiom exists elsewhere — in another file, in the reference implementation, or in a sibling function. Copying a defensive pattern from somewhere else (`x or ""`, `.get(k, default)`, a `try/except` fallback) does NOT make it pre-existing or sanctioned in the new location. "It mirrors the reference" is not a mitigation.

## Two axes for every finding
For each finding, report **both**:
- **Axis A — is it a real fault?** (yes / mitigated). Uses the mitigation labels below.
- **Axis B — does it silently make results untrustworthy?** Label **TRUST-FATAL** (poisons results/state/conclusion → that batch must be rerun) or **ANNOYING** (real but doesn't affect result validity → rollback / delete the line is enough). This tells the user "rerun" vs "just fix". Items 1–3 below are the usual TRUST-FATAL sources; items 4–9 are usually ANNOYING unless they change the protocol.

## Checklist
For the changed code only, answer each item. Allowed answers: **no** / **doesn't apply** / **user override** / **original code already like that** / **this is an exception** (use the last sparingly). Only a plain **yes** with none of those mitigations requires fixing before the run (unless the user says otherwise).

**Usually TRUST-FATAL — check these first:**
1. Does it report success/failure from a proxy (metric / gate / exit-0) without checking the real artifact? (e.g. calls a run "success" on a cosine threshold while the generated image is blurred/broken; claims a restart while W&B `resume="allow"` silently continued the old run; computes a metric under wrong preprocessing — unaligned/wrong-resolution — so the number isn't comparable.)
2. Does it rely on assumed semantics instead of the actual source? (implements a spec from a grepped phrase rather than the full text; runs a third-party method on default objective without confirming dodging-vs-impersonation; sweeps/selects on a proxy metric that isn't the true training objective.)
3. Does it silently change the experimental protocol or data? (drops/demotes/reorders samples; adds early stopping, a different batch size, a context-batching scheme, or a self-created goal the user didn't ask for; absolutizes a conclusion — "all failed" — from a single checkpoint/partial sample.)
4. Does the script not including wandb for a training or eval? (wandb is not needed for quick processing but is required for any long running jobs)

**Usually ANNOYING (rollback fixes it) — still flag:**
4. Did it use a default value when something is wrong? (`return results or ""`, `data.get("key", 0)`) — or add any fallback that substitutes a stand-in for a missing dependency (disabling a required service, a mock/placeholder in a data path)?
5. Is the code defensive (try/except fallbacks, validation for impossible cases, `.get` with defaults, swallowing errors)?
6. Is the code production-ready/polished rather than research/toy-style? Any unrequested machinery (resume, manifest hash, atomic writes, completion markers, per-sample JSONL, CLI generalization, whole-`args` passing, extra asserts)?
7. Is there any input/output chunking (truncating/splitting inputs or outputs, a prompt/completion cap)? e.g. `output[:5000]`, `input_msg[:4000]`.
8. Does it downgrade quality for time/cost/memory (thinking off, smaller model, lower setting, a speed optimization that could change results)?
9. Do the LLM sampling params follow the model's OFFICIAL recommendation (model card / generation_config) rather than a forced/ad-hoc value? Flag setting only temperature and leaving top_p/top_k at API defaults, or any value not traceable to the model's docs.
10. Is the edit minimal?

## Principles the review enforces
Research code: offensive programming, crash on the unexpected, no token/output caps, no speed-for-quality or speed-for-correctness downgrades. Correctness above "it runs".

- **Verify before claim.** Success/failure is judged on the real artifact, never a proxy alone.
- **Raise or skip, never silently fall back.** Missing/malformed dependency → **raise** (abort when it invalidates the run) or **skip** the **whole sample** (let resume rerun it). Never substitute a different signal for a missing one.
- **No silent input/output mangling.** No truncating to fit context, no dropping unknown fields, no silent type coercion, no caps beyond a user-sanctioned one.
- **Do exactly what was asked.** No unrequested features/protocol changes.
- **No single-use or <10-line helper functions** unless the helper has a functional purpose (callable for `to_thread`/executor, `key=`/callback, retry/timeout/context wrapper). Inline cosmetic wrappers.
- **No `.get` with defaults, no `(x or 0)`.** Use `[key]` and `x`; crash if absent.
- **Resumable, disk-backed.** Never hold a dataset's results only in memory; write each sample to disk.
- **Don't catch broad `Exception` to keep a loop alive.** If you don't know the failure, you're hiding it.
- **A denial is a full stop.** No reformulating a denied command to route around it.

## Report format
List only genuine plain-yes violations (or behavioral regressions), each with `file:line`, a one-line reason, and its **Axis B label (TRUST-FATAL / ANNOYING)**. If none, say **"no blocking issues"**. Do not edit anything. Only report what you can point to concrete evidence for — do not manufacture findings, and do not flag a correct idiom as a bug (that causes needless reruns).
