# Video-Memory QA Operating Guideline

Grounded in a cascaded study of 300 questions x 5 rollouts. The two clearest calibration
points are `day-1` (Entity Recognition, all 5 rollouts scored 5/5 -- "all_high") and
`month-167` (Counting, all 5 rollouts scored 0/5 -- "all_low"). Every non-obvious rule
below names which of these (or which general pattern) it comes from.

## 1. How to plan the investigation and choose the first move

- Start from the named entity/anchor in the question (a boss name, a location, an action
  like "took an ice cream and got into a taxi"), not from a blind time scan. In `day-1`,
  every rollout anchored on the boss name 牯护院 (via `semantic_search` or `run_sql LIKE`)
  before touching a time window, and all converged on the correct interval quickly. In
  `month-167`, all 5 rollouts also correctly anchored on "ice cream + taxi entry" — so
  anchor-first retrieval is not what separated the successes from the failure; it is
  necessary but not sufficient.
- Once you have a candidate window, deliberately query more than one table for the same
  fact (episodes dense captions AND topic_facts/semantic_triples structured rows). In
  `day-1`, the same explicit fact ("the player learns the 定身法 skill") existed as both
  an episode caption and a `topic_facts` row; cross-checking cost nothing and added
  redundancy. Treat single-table confirmation as weaker evidence than two independent
  tables agreeing.
- Decide early whether the question is answerable from text alone or requires vision.
  Signal it needs vision: the DB text around your anchor window is only qualitative
  ("dense crowd of fans holding up smartphones") with no explicit number, name, or state
  — that is exactly the situation in `month-167`, where no row in episodes, visual_clips,
  event_steps, topic_facts, or semantic_triples ever states a count. If a keyword/LIKE
  sweep across all the relevant tables for the specific fact you need (a count, an exact
  wording, a specific state) comes back empty or only generic, stop searching text and
  move to `clip_qa` — but budget for more than one `clip_qa` call (see Section 4).

## 2. semantic_search vs run_sql vs sql_qa

- Use `semantic_search` first to locate the right entity name and rough time window when
  you don't yet know exact timestamps or table row IDs — it is what let every `day-1`
  rollout find 牯护院/Bullguard without knowing the interval in advance.
- Once you have a rough window (from semantic_search, or from the question's own
  temporal anchor), switch to `run_sql` with an explicit `start_s BETWEEN ... AND ...`
  or `LIKE '%keyword%'` filter over episodes/topic_facts/event_steps. This is more
  reliable than depending on embedding quality alone once you know roughly where to
  look: in `day-1`, plain time-range or keyword-LIKE scans over episodes/topic_facts
  were what actually surfaced the decisive row in every rollout, semantic_search having
  done the job of narrowing the window first.
- Read the full row text, not just a snippet or score, before trusting a match — the
  decisive evidence in `day-1` was the full caption text ("pointing one's finger and
  silently saying 定 freezes an opponent's movement") and the full topic_facts row
  ("the player learns the 定身法 (immovable technique) skill"), not just that a row
  matched a query.
- When a query needs to join tables that share column names (e.g. `event_chains` and
  `event_steps` both have `start_s`/`end_s`/`description`), alias columns proactively.
  In `day-1` rollout 4, an unaliased join errored with "duplicate SQL column names"; the
  agent recovered by reissuing with `AS` aliases, but that wasted a step. Write the
  alias in the first attempt.
- If a specific fact you need (a number, a named state, an exact quote) is not found by
  any reasonable combination of semantic_search + run_sql across episodes, topic_facts,
  event_steps, visual_clips, and semantic_triples, treat that absence itself as a
  finding — it is the signal from `month-167` that the DB simply does not store the
  answer as text and you must fall back to visual verification rather than keep
  re-querying the same tables with slightly different keywords.

## 3. Anchoring and widening/narrowing time

- Anchor the interval from the entity/action named in the question, then confirm it
  against the DB's own adjacent episode boundaries rather than an arbitrary fixed
  padding. In `day-1`, the fight episode (1140-1170s) and the unlock episode
  (1200-1230s) were adjacent DB rows, so a `BETWEEN 600 AND 1500` sweep safely captured
  both without needing precision tuning.
- For counting/visual questions in a chaotic scene, do not trust the first narrow window
  you localize. `month-167` rollout 0 scoped `clip_qa` to only a 30-second slice
  (176838-176868), "the first third of the actual exit/crowd scene," and scored 0. Widen
  to cover the full described event (anchor start through the point the scene visibly
  ends, e.g. car door closing) before running the visual check, not just the first
  interval that seems plausible.
- If you already have a clue interval or temporal certificate hint (Short vs Long), let
  it directly bound your SQL time-range width: Short-certificate questions (like
  `day-1`) do not need broad multi-hundred-second sweeps to be safe, but Long-certificate
  or ambiguous-boundary questions (like `month-167`'s 176838-176928 crowd scene) warrant
  checking that your window reaches the natural end of the scene, not just the start.

## 4. When clip_qa is worth it, how to scope and phrase it, and how far to trust it

- Use `clip_qa` when text tables only describe a scene qualitatively and the question
  needs a quantitative or fine visual detail (count, specific person identification,
  exact visual state) — this is the correct routing seen in `month-167` itself (skill
  sequence correctly moved from text search to a mandatory visual-verification step).
  The failure there was not calling `clip_qa`, it was trusting a single call's output.
- Scope the `clip_qa` window to the full scene, not a truncated slice: `month-167`
  rollout 0's 30s window undercounted worst (1 person) with the sparsest coverage;
  rollouts 1-4 used a fuller ~60s window (176838-176898) and still undercounted (1-2 vs
  reference 5). This shows widening the window some helps but is not sufficient on its
  own — the deeper problem is frame-sampling density (below).
- Do not accept a single `clip_qa` call's count as final for a dense/chaotic scene. All
  5 `month-167` rollouts issued exactly one `clip_qa` call, sampled only "a handful of
  frames" (2-3 stills), and every one converged on 1-2 distinct people versus the
  reference of 5 — a systematic, not random, undercount. When the scene is described in
  the DB text as "dense crowd" / "chaotic mob" / multiple overlapping people, treat a
  low `clip_qa` count as a *presumptively unreliable* first answer, not a verified fact:
  issue at least a second `clip_qa` call (overlapping or denser-sampled window), and
  explicitly ask the sub-agent to track/enumerate each individual across consecutive
  frames with frame-id evidence rather than return only a final number. Compare the
  described crowd density in the text against the returned count; if they visibly
  disagree ("dense crowd of fans" text but a count of 1), that disagreement is itself
  the signal to re-verify before submitting, not to submit.
- Phrase the `clip_qa` question to ask for an enumerated list of distinct people/events
  with their evidence (what makes each one distinct, e.g. clothing/position/frame id),
  not just "how many X happened" — a bare count from a sub-agent sampling 2-3 frames of
  a mob cannot be sanity-checked by the calling agent, whereas an enumerated list can be.

## 5. Counting, ordering, and dedup

- For counting questions, distinguish "came up to interact/pose for a photo" from
  "filmed/pressed nearby but did not individually approach" — this inclusion-criterion
  ambiguity is exactly what `month-167`'s DB text left unresolved ("dense crowd of fans
  holding up smartphones" doesn't say how many *approached specifically*), and different
  rollouts drew the line differently (1 vs 2 people) without ever getting close to 5.
  When the question's inclusion criterion is ambiguous, get the sub-agent to describe
  *why* each candidate counts (approached vs. stayed in place) rather than deciding the
  criterion yourself from a gut read of 2-3 frames.
- Cross-check any count against the DB's own qualitative language before finalizing: if
  episodes/visual_clips describe the scene as "dense," "many," or "crowd," and your
  count is 1-2, that mismatch is a concrete pre-submission check to run, not just a
  vague warning — literally diff your count against the density adjectives in the
  retrieved rows.
- For ordering/sequence questions, prefer the DB's own adjacent-episode ordering (as in
  `day-1`'s fight-episode → unlock-episode adjacency) over inferring order purely from
  timestamps returned out of table order, since adjacent episode rows already encode the
  narrative sequence.

## 6. Handling causal / entity / temporal question types differently

- **Entity Recognition** (e.g. `day-1`: "what spell was unlocked"): these succeed best
  when you look for an explicit, named fact in structured tables (topic_facts,
  semantic_triples) in addition to dense captions — the win condition is redundant
  textual confirmation, not visual inspection. Do not invoke `clip_qa` for these unless
  a keyword/entity sweep across all text tables comes back empty.
- **Counting**: treat text tables as only useful for locating the scene, not for
  producing the number itself unless a row explicitly states a count (search for it, but
  expect not to find it, per `month-167`). Budget for multiple `clip_qa` passes as the
  default plan, not a fallback only invoked if the first pass looks wrong.
- **Temporal/causal** questions (e.g. "what happened after X", "why did Y happen"):
  anchor on the cause/trigger event first via semantic_search/run_sql, then read the
  next 1-2 adjacent episode rows rather than re-querying broadly — this mirrors the
  `day-1` pattern of using episode adjacency (fight episode immediately followed by
  unlock episode) to establish causal/temporal sequence cheaply.

## 7. Deciding "good enough" and phrasing the answer for the judge

- An answer is good enough when it is corroborated by at least two independent
  retrieval paths (two tables, or one table + one direct read of adjacent context) that
  agree verbatim or near-verbatim on the key fact, as in every `day-1` rollout. A single
  `clip_qa` call's number, by contrast, should not be treated as good enough for a
  counting question in a described "dense/chaotic" scene — that overconfidence is the
  entire failure mode of `month-167`.
- Phrase the answer to lead with the specific fact/number the judge will match against
  (e.g. "定身法 (Immobilize)" first, context second), matching the pattern of every
  `day-1` answer, which stated the term plainly before adding supporting timestamps —
  don't bury the key term in narrative.
- If, after a genuine widen-and-recheck pass (Section 4), the visual count still looks
  low relative to the text's density language, say so honestly rather than reporting a
  falsely confident number — but first actually do the extra `clip_qa` pass; none of the
  `month-167` rollouts did, so it's unverified whether a second pass would have closed
  the gap to 5, only that a single narrow pass reliably does not.

## 8. Mistakes that most often cost points, and the signal you're about to make one

- **Trusting a single narrow-window clip_qa count as final.** Signal: you scoped
  `clip_qa` to less than the full described scene (e.g. the first 30-60s of a longer
  interaction) and got a count that "feels" plausible — stop and widen before
  submitting. (`month-167`, all 5 rollouts.)
- **Submitting a visual count that contradicts the DB's own density language.** Signal:
  the retrieved episode/visual_clips text uses words like "dense crowd," "many people,"
  "chaotic," but your running count is 1-2 — that gap is the check you skipped.
  (`month-167`.)
- **Only issuing one clip_qa call for a counting question in a crowd scene.** Signal:
  your tool-call sequence has exactly one `clip_qa` before `submit` and the scene
  involves more than one or two people — that is under-sampling by construction, since a
  single call only inspects a handful of stills. (`month-167`, every rollout.)
- **Leaving joins unaliased when tables share column names.** Signal: you're about to
  join `event_chains` and `event_steps` (or any two tables both having
  `start_s`/`end_s`/`description`) without `AS` aliases — this will error with duplicate
  column names; alias proactively instead of reactively. (`day-1` rollout 4.)
- **Treating "I found a plausible entity/time window" as equivalent to "I found the
  fact."** Signal: you've anchored the right window (as all `month-167` rollouts did)
  but haven't yet confirmed the *specific* thing the question asks about (a count, a
  name, a state) is present as explicit text — anchoring correctly is necessary but was
  never the differentiator between the two studied cases; explicit-fact confirmation
  (`day-1`) versus its absence (`month-167`) was.
