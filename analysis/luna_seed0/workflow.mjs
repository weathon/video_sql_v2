export const meta = {
  name: 'luna-seed0-rollout-diagnosis',
  description: 'Per-question rollout diagnosis of the Luna seed0 eval run, cascaded 5-way into a final memory-task SKILL',
  phases: [
    { title: 'Leaf', detail: 'one sonnet agent per question, reads its 5 rollouts through memcli.py' },
    { title: 'Merge', detail: 'cascading 5-way merge of diagnosis files' },
    { title: 'Skill', detail: 'final guideline written to analysis/luna_seed0/SKILL.md' },
  ],
}

const ROOT = '/raid/wg25r/video_sql_v2'
const OUT = ROOT + '/analysis/luna_seed0'
const CLI = ROOT + '/analysis/luna_seed0/.venv/bin/python ' + ROOT + '/analysis/luna_seed0/memcli.py'

const CLI_DOC = `Use this CLI (run from ${ROOT}) for everything; never open a segment JSON with Read, they are huge.

  ${CLI} question --timeline T --index I
      question text, reference answer, question_type, temporal_certificate, and the evidence windows.
      total_intervals are LOGICAL timeline seconds, the same axis every tool and trace uses.
      clue_interval / clue_intervals may instead be offsets inside one source video (they carry a video_id) and
      then do NOT match trace times; when the two disagree, trust total_intervals for the logical axis.
  ${CLI} rollouts --timeline T --index I
      the 5 rollouts: status, score (0-5), submitted answer, and every segment JSON path per attempt
  ${CLI} trace-list --file <segment.json> [--max-chars 300]
      tool-call skeleton of one segment: index, tool, arguments, output preview, final_output, per-request token usage
  ${CLI} trace-item --file <segment.json> --index N [--max-chars 20000]
      full arguments and full tool output of one history item
  ${CLI} sql --timeline T --sql "SELECT ..." [--max-rows 50] [--max-chars 800]
      read-only query against that timeline's memory DB (the same DB the agent had)
  ${CLI} schema --timeline T
      tables, row counts, columns
  ${CLI} frame --timeline T --time <logical_seconds> [--count N --step S] --out-dir /tmp/luna-<yourid>
      extract source-video frames at logical timeline times; then Read the printed jpg paths
  ${CLI} copy-frame --frame-id <id> --rollout-dir <rollout dir> --out-dir /tmp/luna-<yourid>
      copy a frame the visual agent actually saw; then Read the printed jpg path

Segment file naming inside an attempt's traces/ dir: answer-segment-*.json is the main retrieval agent (several
segments means compaction happened), clip-<id>-segment-*.json is a visual agent call, sql-qa-segment-*.json is a
SQL question-answering sub-call, judge-attempt-*-segment-*.json is the judge.
Reasoning text is encrypted and unreadable; judge only from tool calls, tool outputs and submitted answers.`

const LEAF_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['question_id', 'file', 'headline'],
  properties: {
    question_id: { type: 'string' },
    file: { type: 'string' },
    headline: { type: 'string', description: 'one or two sentences: score spread and the dominant cause' },
  },
}

const MERGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['file', 'headline'],
  properties: {
    file: { type: 'string' },
    headline: { type: 'string' },
  },
}

function limiter(n) {
  let active = 0
  const queue = []
  const next = () => {
    if (active >= n || queue.length === 0) return
    active++
    const { fn, resolve, reject } = queue.shift()
    fn().then(resolve, reject).finally(() => { active--; next() })
  }
  return (fn) => new Promise((resolve, reject) => { queue.push({ fn, resolve, reject }); next() })
}

function chunk(items, size) {
  const groups = []
  for (let i = 0; i < items.length; i += size) groups.push(items.slice(i, i + size))
  return groups
}

const limit = limiter(1)
const TL = { d: 'day', m: 'month', w: 'week' }
const questions = typeof args === 'string'
  ? args.split(',').map((code) => ({ timeline: TL[code[0]], index: Number(code.slice(1)) }))
  : args
log(`${questions.length} questions, 5 rollouts each, 2 agents at a time`)

phase('Leaf')

const leafPromises = questions.map((q) => {
  const id = `${q.timeline}-${q.index}`
  const file = `${OUT}/leaf/${id}.json`
  const prompt = `You are diagnosing how a video-memory agent answered ONE question across its 5 independent rollouts.

Run directory: ${ROOT}/runs/training_free_luna_seed0_7510e535
Question id: ${id}  (timeline=${q.timeline}, index=${q.index})

The agent under study answers long-video questions from a pre-built memory database (episodes / entities /
semantic_triples / topic_chains+topic_facts / event_chains+event_steps / visual_clips / object_states / graph_edges)
plus embedding search, and can call a visual sub-agent (clip_qa) that samples frames from the source video.
Its tools were: semantic_search, run_sql, sql_qa, clip_qa, add_note, read_notes, read_skill, agent_raise, submit_answer.
A judge scored each submitted answer 0-5 against the reference answer.

${CLI_DOC}

If ${file} already exists and already contains a valid diagnosis for all 5 rollouts, do nothing and return it.

Method (do NOT dump whole traces into your context):
1. ${CLI} question --timeline ${q.timeline} --index ${q.index}  -> question, reference answer, question_type,
   temporal_certificate, and the evidence windows (total_intervals are the logical-timeline seconds to compare
   against trace times).
2. ${CLI} rollouts --timeline ${q.timeline} --index ${q.index}  -> the 5 scores/answers/segment paths.
3. For EACH of the 5 rollouts: trace-list the answer segment(s) to get the tool-call skeleton, then trace-item ONLY
   the few steps that decide the outcome (the search or SQL that found or missed the clue interval, the clip_qa call,
   the last steps before submit_answer). List the clip/sql-qa segment files of that attempt and trace-list them when
   the outcome depended on them.
4. Check the ground truth yourself when it matters: query the memory DB around the clue_intervals with sql to see
   whether the needed fact was retrievable at all (use total_intervals for the logical-time window), and pull frames (frame / copy-frame) when the disagreement is
   visual. Whether to look at images is your call.
5. Explain WHY high-scoring rollouts scored high and low-scoring ones scored low, in terms of concrete retrieval
   behaviour, not vibes. Distinguish: evidence missing from the DB, evidence present but not retrieved, retrieved but
   misread, wrong time window, counting/dedup error, visual sub-agent wrong, answer format/specificity vs judge,
   compaction or execution failure, reference answer itself questionable.

Write your diagnosis with the Write tool to exactly ${file}, as a single JSON object with these keys:
  question_id, timeline, index, question, reference_answer, question_type, temporal_certificate,
  total_intervals (logical seconds), clue_interval (as given in the question file), scores (array of 5 ints or nulls), score_spread ("all_high"|"all_low"|"mixed"),
  rollouts: array of 5 objects {rollout, status, score, answer, tool_call_sequence (compact string like
    "read_skill x4 -> semantic_search(episodes) -> run_sql(topic_facts) -> clip_qa(176838-176898) -> submit"),
    decisive_steps (array of {segment_file, item_index, what_happened}), failure_point (string or null),
    evidence_checked (what you verified yourself with sql/frames)},
  db_evidence: {clue_retrievable (true/false/"partial"), notes},
  root_causes: array of strings, what_worked: array of strings,
  transferable_lessons: array of strings (concrete, actionable advice for answering this kind of question well),
  confidence: "high"|"medium"|"low", open_questions: array of strings.

Then return the structured result: question_id, the file path, and a one-or-two sentence headline.`
  return limit(() => agent(prompt, { label: `leaf:${id}`, phase: 'Leaf', model: 'sonnet', schema: LEAF_SCHEMA }))
    .then((r) => r || { question_id: id, file, headline: 'agent failed' })
})

phase('Merge')

let level = leafPromises
let layer = 1
while (level.length > 5) {
  const groups = chunk(level, 5)
  level = groups.map((group, gi) => {
    const file = `${OUT}/merge-L${layer}/${gi}.json`
    const layerNow = layer
    return (async () => {
      const kids = await Promise.all(group)
      const listing = kids.map((k) => `- ${k.file}: ${k.headline}`).join('\n')
      const prompt = `You are merging ${kids.length} diagnosis files from a study of a video-memory agent's rollouts.

Child files (read each one fully with Read; they are small JSON):
${listing}

Produce a merged JSON that keeps what generalizes and drops what is specific to a single question, but ALWAYS keeps
the question ids that support each claim so the evidence chain stays traceable. Write it with the Write tool to
exactly ${file}, with these keys:
  layer: ${layerNow}, group: ${gi}, source_files: array of the child file paths,
  question_ids: array of every question id covered underneath (union of the children's coverage),
  score_profile: {counts of all_high / mixed / all_low and any pattern by question_type or temporal_certificate},
  failure_patterns: array of {pattern, why_it_happens, question_types_affected, supporting_question_ids, frequency},
  success_patterns: array of {pattern, why_it_works, supporting_question_ids, frequency},
  tool_usage_findings: array of strings about semantic_search / run_sql / sql_qa / clip_qa / notes / compaction usage,
  time_anchoring_findings: array of strings about locating the right interval,
  judge_and_answer_format_findings: array of strings,
  open_questions: array of strings.
Do not invent patterns that no child file supports. If children disagree, record the disagreement.
If ${file} already exists and already covers these same child files, do nothing and return it.

Then return the file path and a one-or-two sentence headline.`
      return limit(() => agent(prompt, { label: `merge-L${layerNow}:${gi}`, phase: 'Merge', model: 'sonnet', schema: MERGE_SCHEMA }))
        .then((r) => r || { file, headline: 'agent failed' })
    })()
  })
  layer++
}

phase('Skill')

const finals = await Promise.all(level)
const finalListing = finals.map((k) => `- ${k.file}: ${k.headline}`).join('\n')
const skillFile = `${OUT}/SKILL.md`
const skillPrompt = `You are writing the final operating guideline for the video-memory question-answering task,
based on a cascaded study of 300 questions x 5 rollouts of a real evaluation run.

Top-level merged files (read each fully with Read):
${finalListing}

The agent this guideline is for answers long-video questions using a pre-built memory database and embedding search
(tables: episodes with 30sec/3min/... granularities, entities, semantic_triples, topic_chains/topic_facts,
event_chains/event_steps, visual_clips, object_states, graph_edges, plus evidence link tables), with tools
semantic_search, run_sql, sql_qa, clip_qa (visual sub-agent over the source video), add_note/read_notes, read_skill,
submit_answer. Answers are judged 0-5 for semantic equivalence to a reference answer.

Write ${skillFile} with the Write tool: a practical guideline in Markdown, no frontmatter. It must be grounded in the
merged findings, and every non-obvious recommendation should name the failure or success pattern it comes from.
Cover at least: how to plan an investigation and choose the first retrieval move; how to use semantic_search vs
run_sql vs sql_qa (query phrasing, tables, granularity, score ranges, when to read full rows); how to anchor time and
widen or narrow intervals; when clip_qa is worth it, how to scope the interval and phrase the visual question, and
how far to trust its answer; counting, ordering and dedup; how to handle causal / entity / temporal question types
differently; how to decide the answer is good enough and how to phrase it for the judge; and the mistakes that most
often cost points, with the concrete signal that you are about to make each one.

Return the path to the file you wrote and a two-sentence summary of the guideline.`

const skill = await limit(() => agent(skillPrompt, { label: 'final-skill', phase: 'Skill', model: 'sonnet' }))

return { leaves: questions.length, top_level_files: finals.map((f) => f.file), skill_file: skillFile, skill_summary: skill }
