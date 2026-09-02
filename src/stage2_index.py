import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import wandb
from openai import APIError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from src.preprocess.normalize import normalize_text
from src.stage2_prepare import Chunk, QWEN_MODEL, SourceGroup, SourceVideo, WORKERS, load_jsonl


class EntityItem(BaseModel):
    name: str
    evidence_caption_row_ids: list[str]


class SemanticTripleItem(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence_caption_row_ids: list[str]


class TopicFactItem(BaseModel):
    fact: str
    evidence_caption_row_ids: list[str]


class TopicChainItem(BaseModel):
    entity: str
    has_lifecycle: bool
    summary: str
    facts: list[TopicFactItem]


class EventStepItem(BaseModel):
    description: str
    evidence_caption_row_ids: list[str]


class EventChainItem(BaseModel):
    name: str
    chain_type: str
    key_entities: list[str]
    summary: str
    steps: list[EventStepItem]


class ObjectStateItem(BaseModel):
    object: str
    status: str
    text: str
    caption_30sec_row_id: str


class WindowOutput(BaseModel):
    summary: str
    entities: list[EntityItem]
    semantic_triples: list[SemanticTripleItem]
    topic_chains: list[TopicChainItem]
    event_chains: list[EventChainItem]
    object_states: list[ObjectStateItem]


class HierarchyOutput(BaseModel):
    summary: str
    topic_chains: list[TopicChainItem]
    event_chains: list[EventChainItem]


class ModelOutputError(RuntimeError):
    pass


class ContextWindowExceeded(RuntimeError):
    pass


def checkpoint_path(run_root: Path, level: str, scope_id: str) -> Path:
    return run_root / "structure" / level / f"{scope_id.replace(':', '__')}.json"


def validate_evidence(output: WindowOutput | HierarchyOutput, caption_row_ids: set[str]) -> None:
    if not output.summary.strip():
        raise ModelOutputError("summary is empty")
    evidence_lists = []
    if isinstance(output, WindowOutput):
        evidence_lists.extend(item.evidence_caption_row_ids for item in output.entities)
        evidence_lists.extend(item.evidence_caption_row_ids for item in output.semantic_triples)
        for item in output.object_states:
            evidence_lists.append([item.caption_30sec_row_id])
    for chain in output.topic_chains:
        if not chain.facts:
            raise ModelOutputError(f"topic chain has no facts: {chain.entity}")
        evidence_lists.extend(fact.evidence_caption_row_ids for fact in chain.facts)
    for chain in output.event_chains:
        if not chain.steps:
            raise ModelOutputError(f"event chain has no steps: {chain.name}")
        evidence_lists.extend(step.evidence_caption_row_ids for step in chain.steps)
    for evidence in evidence_lists:
        if not evidence:
            raise ModelOutputError("record has no 30-second evidence row ID")
        unknown = set(evidence) - caption_row_ids
        if unknown:
            raise ModelOutputError(f"record cites unavailable 30-second rows: {sorted(unknown)}")


def is_context_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in ["maximum context length", "context length", "input is too long", "input length", "too many tokens"])


async def call_structured(client: AsyncOpenAI, semaphore: asyncio.Semaphore, run_root: Path, trace_kind: str, scope_id: str, messages: list[dict], response_model: type[BaseModel], caption_row_ids: set[str]) -> BaseModel:
    last_error = None
    context_failures = 0
    repair_message = None
    for attempt in range(1, 6):
        attempt_messages = messages if repair_message is None else messages + [{"role": "user", "content": repair_message}]
        trace_path = run_root / "traces" / trace_kind / scope_id.replace(":", "__") / f"attempt-{attempt}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        request = {"model": QWEN_MODEL, "reasoning_effort": "low", "messages": attempt_messages, "response_format": response_model.model_json_schema()}
        try:
            async with semaphore:
                response = await client.beta.chat.completions.parse(model=QWEN_MODEL, messages=attempt_messages, response_format=response_model, reasoning_effort="low")
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ModelOutputError(f"missing parsed output: {response.choices[0].message.model_dump(mode='json')}")
            validate_evidence(parsed, caption_row_ids)
        except (APIError, LengthFinishReasonError, ValidationError, ModelOutputError) as error:
            last_error = error
            if is_context_error(error):
                context_failures += 1
            else:
                repair_message = f"Your preceding structured response was rejected by validation: {error}. Return the complete corrected object using only the supplied 30-second row IDs."
            trace_path.write_text(json.dumps({"request": request, "error": str(error)}, ensure_ascii=False, indent=2))
            print(json.dumps({"stage": trace_kind, "scope_id": scope_id, "attempt": attempt, "error": str(error)}))
            continue
        trace_path.write_text(json.dumps({"request": request, "response": response.model_dump(mode="json")}, ensure_ascii=False, indent=2))
        return parsed
    if context_failures == 5:
        raise ContextWindowExceeded(f"Context exceeded on all 5 attempts for {scope_id}: {last_error}")
    raise RuntimeError(f"Structured call failed after 5 attempts for {scope_id}: {last_error}")


async def build_structure(run_root: Path, base_url: str) -> dict[str, int]:
    chunks = load_jsonl(run_root / "chunks.jsonl", Chunk)
    source_groups = load_jsonl(run_root / "source_groups.jsonl", SourceGroup)
    dense = {}
    for chunk in chunks:
        dense_path = run_root / "dense" / f"{chunk.row_id.replace(':', '__')}.json"
        payload = json.loads(dense_path.read_text())
        if payload["row_id"] != chunk.row_id or not str(payload["text"]).strip():
            raise RuntimeError(f"Dense checkpoint mismatch in {dense_path}")
        dense[chunk.row_id] = str(payload["text"]).strip()
    client = AsyncOpenAI(base_url=base_url.rstrip("/") + "/v1", max_retries=0, default_headers={"Authorization": ""}, _enforce_credentials=False)
    semaphore = asyncio.Semaphore(WORKERS)
    window_system = "You build a factual video-memory index from ordered 30-second dense captions. Extract stable entities, directly supported semantic triples, topic lifecycle facts, chronological local event chains, important object or interface states, and a faithful rollup summary. Every output record must cite one or more supplied caption row IDs. Object states must cite the exact single 30-second row where that state is visible or stated. Keep concrete details and do not infer unsupported facts."
    hierarchy_system = "You merge child video-memory indexes into a higher temporal level. Produce a faithful summary and merged topic and event chains. Preserve concrete details and chronology. Every fact and event step must cite only 30-second caption row IDs present in the child inputs. Do not add entities, facts, states, or events unsupported by those inputs."

    chunks_by_group = defaultdict(list)
    for chunk in chunks:
        chunks_by_group[chunk.source_group_id].append(chunk)
    for group_chunks in chunks_by_group.values():
        group_chunks.sort(key=lambda chunk: (chunk.start_s, chunk.row_id))

    async def process_fragment(scope_id: str, fragment_chunks: list[Chunk]) -> WindowOutput:
        fragment_checkpoint = checkpoint_path(run_root, "fragments", scope_id)
        if fragment_checkpoint.exists():
            payload = json.loads(fragment_checkpoint.read_text())
            if payload["caption_row_ids"] != [chunk.row_id for chunk in fragment_chunks]:
                raise RuntimeError(f"Fragment checkpoint row mismatch in {fragment_checkpoint}")
            return WindowOutput.model_validate(payload["output"])
        caption_rows = [{"row_id": chunk.row_id, "start_s": chunk.start_s, "end_s": chunk.end_s, "text": dense[chunk.row_id]} for chunk in fragment_chunks]
        messages = [{"role": "system", "content": window_system}, {"role": "user", "content": json.dumps({"scope_id": scope_id, "captions": caption_rows}, ensure_ascii=False)}]
        try:
            output = await call_structured(client, semaphore, run_root, "30min", scope_id, messages, WindowOutput, {chunk.row_id for chunk in fragment_chunks})
        except ContextWindowExceeded:
            if len(fragment_chunks) == 1:
                raise RuntimeError(f"30-second input exceeded context after 5 attempts for {fragment_chunks[0].row_id}")
            midpoint = len(fragment_chunks) // 2
            left = await process_fragment(f"{scope_id}:left", fragment_chunks[:midpoint])
            right = await process_fragment(f"{scope_id}:right", fragment_chunks[midpoint:])
            output = WindowOutput(summary=f"{left.summary} {right.summary}", entities=left.entities + right.entities, semantic_triples=left.semantic_triples + right.semantic_triples, topic_chains=left.topic_chains + right.topic_chains, event_chains=left.event_chains + right.event_chains, object_states=left.object_states + right.object_states)
            validate_evidence(output, {chunk.row_id for chunk in fragment_chunks})
        fragment_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fragment_checkpoint.write_text(json.dumps({"level": "fragment", "scope_id": scope_id, "video_id": fragment_chunks[0].video_id, "start_s": fragment_chunks[0].start_s, "end_s": fragment_chunks[-1].end_s, "caption_row_ids": [chunk.row_id for chunk in fragment_chunks], "output": output.model_dump(mode="json")}, ensure_ascii=False, indent=2))
        return output

    async def process_window(scope_id: str, window_chunks: list[Chunk]) -> dict:
        output_path = checkpoint_path(run_root, "30min", scope_id)
        if output_path.exists():
            payload = json.loads(output_path.read_text())
            if payload["caption_row_ids"] != [chunk.row_id for chunk in window_chunks]:
                raise RuntimeError(f"30-minute checkpoint row mismatch in {output_path}")
            WindowOutput.model_validate(payload["output"])
            return payload
        output = await process_fragment(scope_id, window_chunks)
        payload = {"level": "30min", "scope_id": scope_id, "video_id": window_chunks[0].video_id, "source_group_id": window_chunks[0].source_group_id, "start_s": window_chunks[0].start_s, "end_s": window_chunks[-1].end_s, "caption_row_ids": [chunk.row_id for chunk in window_chunks], "output": output.model_dump(mode="json")}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    window_specs = []
    for group in source_groups:
        group_chunks = chunks_by_group[group.source_group_id]
        for index in range(0, len(group_chunks), 60):
            window_specs.append((f"{group.source_group_id}:30min:{index // 60:04d}", group_chunks[index:index + 60]))
    window_tasks = [asyncio.create_task(process_window(scope_id, window_chunks)) for scope_id, window_chunks in window_specs]
    windows = []
    for completed, task in enumerate(asyncio.as_completed(window_tasks), 1):
        windows.append(await task)
        if completed % 50 == 0 or completed == len(window_tasks):
            print(json.dumps({"stage": "30min", "completed": completed, "total": len(window_tasks)}))
    windows.sort(key=lambda payload: (payload["video_id"], payload["start_s"], payload["scope_id"]))

    async def process_scale_fragment(scope_id: str, children: list[dict]) -> dict:
        output_path = checkpoint_path(run_root, "scale_fragments", scope_id)
        caption_row_ids = list(dict.fromkeys(row_id for child in children for row_id in child["caption_row_ids"]))
        if output_path.exists():
            payload = json.loads(output_path.read_text())
            if payload["caption_row_ids"] != caption_row_ids:
                raise RuntimeError(f"Scale fragment checkpoint row mismatch in {output_path}")
            output = HierarchyOutput.model_validate(payload["output"])
            validate_evidence(output, set(caption_row_ids))
            return payload
        child_inputs = [{"level": child["level"], "scope_id": child["scope_id"], "start_s": child["start_s"], "end_s": child["end_s"], "output": child["output"]} for child in children]
        messages = [{"role": "system", "content": hierarchy_system}, {"role": "user", "content": json.dumps({"target_level": "scale", "scope_id": scope_id, "children": child_inputs}, ensure_ascii=False)}]
        try:
            output = await call_structured(client, semaphore, run_root, "scale_fragments", scope_id, messages, HierarchyOutput, set(caption_row_ids))
        except ContextWindowExceeded:
            if len(children) == 1:
                raise RuntimeError(f"Single-child scale fragment exceeded context after 5 attempts for {scope_id}")
            midpoint = len(children) // 2
            left, right = await asyncio.gather(process_scale_fragment(f"{scope_id}:left", children[:midpoint]), process_scale_fragment(f"{scope_id}:right", children[midpoint:]))
            fragment_children = [left, right]
            child_inputs = [{"level": child["level"], "scope_id": child["scope_id"], "start_s": child["start_s"], "end_s": child["end_s"], "output": child["output"]} for child in fragment_children]
            messages = [{"role": "system", "content": hierarchy_system}, {"role": "user", "content": json.dumps({"target_level": "scale", "scope_id": scope_id, "children": child_inputs}, ensure_ascii=False)}]
            try:
                output = await call_structured(client, semaphore, run_root, "scale_fragments", f"{scope_id}:merge", messages, HierarchyOutput, set(caption_row_ids))
            except ContextWindowExceeded as error:
                raise RuntimeError(f"Layered scale fragment exceeded context after 5 attempts for {scope_id}") from error
        payload = {"level": "scale_fragment", "scope_id": scope_id, "video_id": children[0]["video_id"], "source_group_id": None, "start_s": children[0]["start_s"], "end_s": children[-1]["end_s"], "caption_row_ids": caption_row_ids, "output": output.model_dump(mode="json")}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    async def process_hierarchy(level: str, scope_id: str, children: list[dict]) -> dict:
        output_path = checkpoint_path(run_root, level, scope_id)
        caption_row_ids = list(dict.fromkeys(row_id for child in children for row_id in child["caption_row_ids"]))
        if output_path.exists():
            payload = json.loads(output_path.read_text())
            if payload["caption_row_ids"] != caption_row_ids:
                raise RuntimeError(f"{level} checkpoint row mismatch in {output_path}")
            HierarchyOutput.model_validate(payload["output"])
            return payload
        child_inputs = [{"level": child["level"], "scope_id": child["scope_id"], "start_s": child["start_s"], "end_s": child["end_s"], "caption_row_ids": child["caption_row_ids"], "output": child["output"]} for child in children]
        messages = [{"role": "system", "content": hierarchy_system}, {"role": "user", "content": json.dumps({"target_level": level, "scope_id": scope_id, "children": child_inputs}, ensure_ascii=False)}]
        direct_context_exhausted = False
        trace_dir = run_root / "traces" / level / scope_id.replace(":", "__")
        attempt_paths = sorted(trace_dir.glob("attempt-*.json"), key=lambda path: int(path.stem.removeprefix("attempt-")))
        if attempt_paths and len(attempt_paths) == 5:
            attempt_numbers = [int(path.stem.removeprefix("attempt-")) for path in attempt_paths]
            if attempt_numbers != [1, 2, 3, 4, 5]:
                raise RuntimeError(f"Hierarchy trace attempts are incomplete for {scope_id}")
            trace_payloads = [json.loads(path.read_text()) for path in attempt_paths]
            if level == "scale" and all(is_context_error(RuntimeError(payload["error"])) for payload in trace_payloads):
                direct_context_exhausted = True
            else:
                raise RuntimeError(f"Hierarchy call already failed after 5 attempts for {scope_id}")
        if not direct_context_exhausted:
            try:
                output = await call_structured(client, semaphore, run_root, level, scope_id, messages, HierarchyOutput, set(caption_row_ids))
            except ContextWindowExceeded as error:
                if level != "scale":
                    raise RuntimeError(f"{level} hierarchy input exceeded context after 5 attempts for {scope_id}") from error
                direct_context_exhausted = True
        if direct_context_exhausted:
            if len(children) == 1:
                raise RuntimeError(f"Single-child scale input exceeded context after 5 attempts for {scope_id}")
            midpoint = len(children) // 2
            left, right = await asyncio.gather(process_scale_fragment(f"{scope_id}:left", children[:midpoint]), process_scale_fragment(f"{scope_id}:right", children[midpoint:]))
            fragment_children = [left, right]
            child_inputs = [{"level": child["level"], "scope_id": child["scope_id"], "start_s": child["start_s"], "end_s": child["end_s"], "output": child["output"]} for child in fragment_children]
            messages = [{"role": "system", "content": hierarchy_system}, {"role": "user", "content": json.dumps({"target_level": level, "scope_id": scope_id, "children": child_inputs}, ensure_ascii=False)}]
            try:
                output = await call_structured(client, semaphore, run_root, "scale_layered", scope_id, messages, HierarchyOutput, set(caption_row_ids))
            except ContextWindowExceeded as error:
                raise RuntimeError(f"Layered scale input exceeded context after 5 attempts for {scope_id}") from error
        source_group_id = None if level == "scale" else children[0]["source_group_id"]
        payload = {"level": level, "scope_id": scope_id, "video_id": children[0]["video_id"], "source_group_id": source_group_id, "start_s": children[0]["start_s"], "end_s": children[-1]["end_s"], "caption_row_ids": caption_row_ids, "output": output.model_dump(mode="json")}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    async def build_level(level: str, grouped_children: list[tuple[str, list[dict]]]) -> list[dict]:
        tasks = [asyncio.create_task(process_hierarchy(level, scope_id, children)) for scope_id, children in grouped_children]
        results = []
        for completed, task in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await task)
            if completed % 20 == 0 or completed == len(tasks):
                print(json.dumps({"stage": level, "completed": completed, "total": len(tasks)}))
        return sorted(results, key=lambda payload: (payload["video_id"], payload["start_s"], payload["scope_id"]))

    windows_by_group = defaultdict(list)
    for payload in windows:
        windows_by_group[payload["source_group_id"]].append(payload)
    one_hour_specs = []
    for source_group_id, children in windows_by_group.items():
        for index in range(0, len(children), 2):
            one_hour_specs.append((f"{source_group_id}:1h:{index // 2:04d}", children[index:index + 2]))
    one_hours = await build_level("1h", one_hour_specs)
    one_hours_by_group = defaultdict(list)
    for payload in one_hours:
        one_hours_by_group[payload["source_group_id"]].append(payload)
    six_hour_specs = []
    for source_group_id, children in one_hours_by_group.items():
        for index in range(0, len(children), 6):
            six_hour_specs.append((f"{source_group_id}:6h:{index // 6:04d}", children[index:index + 6]))
    six_hours = await build_level("6h", six_hour_specs)
    six_hours_by_group = defaultdict(list)
    for payload in six_hours:
        six_hours_by_group[payload["source_group_id"]].append(payload)
    video_specs = [(f"{source_group_id}:video", children) for source_group_id, children in six_hours_by_group.items()]
    videos = await build_level("video", video_specs)
    videos_by_scale = defaultdict(list)
    for payload in videos:
        videos_by_scale[payload["video_id"]].append(payload)
    scale_specs = [(f"{video_id}:scale", sorted(children, key=lambda payload: payload["start_s"])) for video_id, children in videos_by_scale.items()]
    scales = await build_level("scale", scale_specs)
    await client.close()
    return {"windows_30min": len(windows), "episodes_1h": len(one_hours), "episodes_6h": len(six_hours), "video_rollups": len(videos), "scale_rollups": len(scales)}


def build_database(run_root: Path, output_path: Path) -> dict[str, int]:
    if output_path.exists():
        raise RuntimeError(f"Database output already exists: {output_path}")
    sources = load_jsonl(run_root / "source_videos.jsonl", SourceVideo)
    source_groups = load_jsonl(run_root / "source_groups.jsonl", SourceGroup)
    chunks = load_jsonl(run_root / "chunks.jsonl", Chunk)
    chunks_by_id = {chunk.row_id: chunk for chunk in chunks}
    dense = {}
    for chunk in chunks:
        payload = json.loads((run_root / "dense" / f"{chunk.row_id.replace(':', '__')}.json").read_text())
        if payload["row_id"] != chunk.row_id or not str(payload["text"]).strip():
            raise RuntimeError(f"Dense checkpoint mismatch for {chunk.row_id}")
        dense[chunk.row_id] = normalize_text(payload["text"])
    structure = {}
    for level in ["30min", "1h", "6h", "video", "scale"]:
        structure[level] = [json.loads(path.read_text()) for path in sorted((run_root / "structure" / level).glob("*.json"))]
        for payload in structure[level]:
            model = WindowOutput if level == "30min" else HierarchyOutput
            output = model.model_validate(payload["output"])
            validate_evidence(output, set(payload["caption_row_ids"]))

    chunks_by_group = defaultdict(list)
    for chunk in chunks:
        chunks_by_group[chunk.source_group_id].append(chunk)
    for group_chunks in chunks_by_group.values():
        group_chunks.sort(key=lambda chunk: (chunk.start_s, chunk.row_id))
    expected = {level: {} for level in ["30min", "1h", "6h", "video", "scale"]}
    for group in source_groups:
        group_chunks = chunks_by_group[group.source_group_id]
        for index in range(0, len(group_chunks), 60):
            scope_id = f"{group.source_group_id}:30min:{index // 60:04d}"
            expected["30min"][scope_id] = [chunk.row_id for chunk in group_chunks[index:index + 60]]
        window_scopes = list(expected["30min"])[-((len(group_chunks) + 59) // 60):]
        one_hour_scopes = []
        for index in range(0, len(window_scopes), 2):
            scope_id = f"{group.source_group_id}:1h:{index // 2:04d}"
            child_scopes = window_scopes[index:index + 2]
            expected["1h"][scope_id] = [row_id for child_scope in child_scopes for row_id in expected["30min"][child_scope]]
            one_hour_scopes.append(scope_id)
        six_hour_scopes = []
        for index in range(0, len(one_hour_scopes), 6):
            scope_id = f"{group.source_group_id}:6h:{index // 6:04d}"
            child_scopes = one_hour_scopes[index:index + 6]
            expected["6h"][scope_id] = [row_id for child_scope in child_scopes for row_id in expected["1h"][child_scope]]
            six_hour_scopes.append(scope_id)
        video_scope_id = f"{group.source_group_id}:video"
        expected["video"][video_scope_id] = [row_id for child_scope in six_hour_scopes for row_id in expected["6h"][child_scope]]
    groups_by_video = defaultdict(list)
    for group in source_groups:
        groups_by_video[group.video_id].append(group)
    for video_id, video_groups in groups_by_video.items():
        video_groups.sort(key=lambda group: (group.timeline_start_s, group.source_group_id))
        expected["scale"][f"{video_id}:scale"] = [row_id for group in video_groups for row_id in expected["video"][f"{group.source_group_id}:video"]]
    for level in expected:
        actual = {payload["scope_id"]: payload["caption_row_ids"] for payload in structure[level]}
        if actual != expected[level]:
            raise RuntimeError(f"Incomplete or stale {level} checkpoints: expected={len(expected[level])}, actual={len(actual)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(output_path))
    connection.execute((Path(__file__).parent / "stage2_schema.sql").read_text())
    connection.execute("BEGIN TRANSACTION")
    edge_counter = 0
    def insert_node(row_id: str, node_type: str, video_id: str, start_s: float, end_s: float, text: str) -> None:
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", [row_id, node_type, video_id, start_s, end_s, text])
    def insert_edge(video_id: str, src_row_id: str, dst_row_id: str, edge_type: str, weight: float) -> None:
        nonlocal edge_counter
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)", [f"{video_id}:graph_edges:{edge_counter:09d}", video_id, src_row_id, dst_row_id, edge_type, weight])
        edge_counter += 1

    sources_by_video = defaultdict(list)
    for source in sources:
        sources_by_video[source.video_id].append(source)
    scale_names = {"day": "Day", "week": "Week", "month": "Month"}
    for video_id in ["day", "week", "month"]:
        video_sources = sources_by_video[video_id]
        connection.execute("INSERT INTO videos VALUES (?, ?, ?, ?)", [video_id, scale_names[video_id], max(source.timeline_end_s for source in video_sources), len(video_sources)])
    for group in source_groups:
        connection.execute("INSERT INTO source_groups VALUES (?, ?, ?, ?, ?)", [group.source_group_id, group.video_id, group.name, group.timeline_start_s, group.timeline_end_s])
    for source in sources:
        connection.execute("INSERT INTO source_videos VALUES (?, ?, ?, ?, ?, ?, ?)", [source.source_video_id, source.video_id, source.source_group_id, source.filename, source.duration_s, source.timeline_start_s, source.timeline_end_s])

    for chunk in chunks:
        text = dense[chunk.row_id]
        insert_node(chunk.row_id, "episode", chunk.video_id, chunk.start_s, chunk.end_s, text)
        connection.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)", [chunk.row_id, chunk.video_id, chunk.source_video_id, chunk.start_s, chunk.end_s, "30sec", text])
        insert_node(chunk.visual_row_id, "visual_clip", chunk.video_id, chunk.start_s, chunk.end_s, text)
        connection.execute("INSERT INTO visual_clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [chunk.visual_row_id, chunk.video_id, chunk.source_video_id, chunk.start_s, chunk.end_s, chunk.source_start_s, chunk.source_end_s, chunk.clip_filename, text])
        insert_edge(chunk.video_id, chunk.visual_row_id, chunk.row_id, "co_clip", 1.0)
        insert_edge(chunk.video_id, chunk.row_id, chunk.visual_row_id, "co_clip", 1.0)

    caption_to_3min = {}
    episode_rows_by_level = defaultdict(list)
    three_min_indexes = defaultdict(int)
    for group in source_groups:
        group_chunks = chunks_by_group[group.source_group_id]
        for index in range(0, len(group_chunks), 6):
            batch = group_chunks[index:index + 6]
            video_id = group.video_id
            row_id = f"{video_id}:episodes:3min:{three_min_indexes[video_id]:06d}"
            three_min_indexes[video_id] += 1
            source_video_ids = {chunk.source_video_id for chunk in batch}
            source_video_id = next(iter(source_video_ids)) if len(source_video_ids) == 1 else None
            text = normalize_text(" ".join(dense[chunk.row_id] for chunk in batch))
            insert_node(row_id, "episode", video_id, batch[0].start_s, batch[-1].end_s, text)
            connection.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, source_video_id, batch[0].start_s, batch[-1].end_s, "3min", text])
            episode_rows_by_level[(video_id, "3min")].append((row_id, batch[0].start_s, batch[-1].end_s))
            for chunk in batch:
                caption_to_3min[chunk.row_id] = row_id
                insert_edge(video_id, row_id, chunk.row_id, "contains", 1.0)

    for level in ["1h", "6h", "video", "scale"]:
        for index, payload in enumerate(sorted(structure[level], key=lambda item: (item["video_id"], item["start_s"], item["scope_id"]))):
            video_id = payload["video_id"]
            row_id = f"{video_id}:episodes:{level}:{index:06d}"
            text = normalize_text(payload["output"]["summary"])
            insert_node(row_id, "episode", video_id, payload["start_s"], payload["end_s"], text)
            connection.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, None, payload["start_s"], payload["end_s"], level, text])
            episode_rows_by_level[(video_id, level)].append((row_id, payload["start_s"], payload["end_s"]))
            for caption_row_id in payload["caption_row_ids"]:
                insert_edge(video_id, row_id, caption_row_id, "contains", 0.5)

    for (video_id, level), rows in episode_rows_by_level.items():
        rows.sort(key=lambda row: (row[1], row[0]))
        for current, following in zip(rows, rows[1:]):
            gap_s = max(0.0, following[1] - current[2])
            weight = 1.0 / (gap_s + 1.0)
            insert_edge(video_id, current[0], following[0], "temporal_next", weight)
            insert_edge(video_id, following[0], current[0], "temporal_next", weight)

    entity_state = {}
    for payload in structure["30min"]:
        output = WindowOutput.model_validate(payload["output"])
        for entity in output.entities:
            normalized_name = normalize_text(entity.name)
            key = (payload["video_id"], normalized_name)
            state = entity_state.setdefault(key, {"name": entity.name, "evidence": set()})
            state["evidence"].update(entity.evidence_caption_row_ids)
    for entity_index, ((video_id, normalized_name), state) in enumerate(sorted(entity_state.items())):
        evidence = sorted(state["evidence"], key=lambda row_id: (chunks_by_id[row_id].start_s, row_id))
        row_id = f"{video_id}:entities:{entity_index:06d}"
        name = normalize_text(state["name"])
        start_s = chunks_by_id[evidence[0]].start_s
        end_s = chunks_by_id[evidence[-1]].end_s
        insert_node(row_id, "entity", video_id, start_s, end_s, name)
        connection.execute("INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, name, normalized_name, start_s, end_s, len(evidence)])
        for caption_row_id in evidence:
            insert_edge(video_id, row_id, caption_row_id, "mentioned_in", 1.0)
            insert_edge(video_id, row_id, chunks_by_id[caption_row_id].visual_row_id, "appeared_in", 0.8)

    semantic_count = 0
    object_state_count = 0
    for payload in sorted(structure["30min"], key=lambda item: (item["video_id"], item["start_s"], item["scope_id"])):
        video_id = payload["video_id"]
        output = WindowOutput.model_validate(payload["output"])
        for triple in output.semantic_triples:
            evidence = sorted(dict.fromkeys(triple.evidence_caption_row_ids), key=lambda row_id: (chunks_by_id[row_id].start_s, row_id))
            row_id = f"{video_id}:semantic_triples:{semantic_count:07d}"
            subject = normalize_text(triple.subject)
            predicate = normalize_text(triple.predicate)
            object_value = normalize_text(triple.object)
            text = normalize_text(f"{subject} {predicate} {object_value}")
            start_s = chunks_by_id[evidence[0]].start_s
            end_s = chunks_by_id[evidence[-1]].end_s
            insert_node(row_id, "semantic", video_id, start_s, end_s, text)
            connection.execute("INSERT INTO semantic_triples VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, evidence[0], end_s, subject, predicate, object_value, text])
            for caption_row_id in evidence:
                insert_edge(video_id, row_id, caption_row_id, "has_property", 1.0)
            semantic_count += 1
        for state in output.object_states:
            chunk = chunks_by_id[state.caption_30sec_row_id]
            row_id = f"{video_id}:object_states:{object_state_count:07d}"
            object_name = normalize_text(state.object)
            status = normalize_text(state.status)
            text = normalize_text(state.text)
            insert_node(row_id, "object_state", video_id, chunk.start_s, chunk.end_s, text)
            connection.execute("INSERT INTO object_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [row_id, video_id, chunk.source_video_id, caption_to_3min[chunk.row_id], chunk.row_id, chunk.start_s, chunk.end_s, object_name, status, text])
            insert_edge(video_id, row_id, chunk.row_id, "state_in", 1.0)
            object_state_count += 1

    topic_chain_count = 0
    topic_fact_count = 0
    event_chain_count = 0
    event_step_count = 0
    for level in ["30min", "1h", "6h", "video", "scale"]:
        for payload in sorted(structure[level], key=lambda item: (item["video_id"], item["start_s"], item["scope_id"])):
            video_id = payload["video_id"]
            output = WindowOutput.model_validate(payload["output"]) if level == "30min" else HierarchyOutput.model_validate(payload["output"])
            for chain in output.topic_chains:
                chain_id = f"{video_id}:topic_chains:{topic_chain_count:07d}"
                connection.execute("INSERT INTO topic_chains VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [chain_id, video_id, normalize_text(chain.entity), chain.has_lifecycle, level, payload["scope_id"], payload["start_s"], payload["end_s"], normalize_text(chain.summary)])
                for fact in chain.facts:
                    evidence = sorted(dict.fromkeys(fact.evidence_caption_row_ids), key=lambda row_id: (chunks_by_id[row_id].start_s, row_id))
                    fact_id = f"{video_id}:topic_facts:{topic_fact_count:08d}"
                    connection.execute("INSERT INTO topic_facts VALUES (?, ?, ?, ?, ?, ?)", [fact_id, chain_id, video_id, evidence[0], chunks_by_id[evidence[0]].start_s, normalize_text(fact.fact)])
                    for evidence_index, caption_row_id in enumerate(evidence):
                        connection.execute("INSERT INTO topic_fact_evidence VALUES (?, ?, ?)", [f"{fact_id}:evidence:{evidence_index:03d}", fact_id, caption_row_id])
                    topic_fact_count += 1
                topic_chain_count += 1
            for chain in output.event_chains:
                chain_id = f"{video_id}:event_chains:{event_chain_count:07d}"
                connection.execute("INSERT INTO event_chains VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [chain_id, video_id, normalize_text(chain.name), normalize_text(chain.chain_type), level, payload["scope_id"], payload["start_s"], payload["end_s"], normalize_text(chain.summary)])
                for entity_index, entity in enumerate(chain.key_entities):
                    connection.execute("INSERT INTO event_chain_entities VALUES (?, ?, ?, ?)", [f"{chain_id}:entities:{entity_index:03d}", chain_id, video_id, normalize_text(entity)])
                for step_index, step in enumerate(chain.steps):
                    evidence = sorted(dict.fromkeys(step.evidence_caption_row_ids), key=lambda row_id: (chunks_by_id[row_id].start_s, row_id))
                    step_id = f"{video_id}:event_steps:{event_step_count:08d}"
                    connection.execute("INSERT INTO event_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [step_id, chain_id, video_id, evidence[0], evidence[-1], step_index, chunks_by_id[evidence[0]].start_s, chunks_by_id[evidence[-1]].end_s, normalize_text(step.description)])
                    for evidence_index, caption_row_id in enumerate(evidence):
                        connection.execute("INSERT INTO event_step_evidence VALUES (?, ?, ?)", [f"{step_id}:evidence:{evidence_index:03d}", step_id, caption_row_id])
                    event_step_count += 1
                event_chain_count += 1
    connection.execute("COMMIT")
    counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ["videos", "source_videos", "episodes", "visual_clips", "entities", "semantic_triples", "object_states", "topic_chains", "topic_facts", "event_chains", "event_steps", "graph_edges"]}
    connection.close()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["structure", "database"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    args = parser.parse_args()
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name, job_type=f"stage2-{args.stage}", config={"stage": args.stage, "workers": WORKERS, "qwen_model": QWEN_MODEL, "run_root": str(args.run_root), "output": str(args.output) if args.output else None})
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    if args.stage == "structure":
        if args.base_url is None:
            raise RuntimeError("--base-url is required for structure")
        metrics = asyncio.run(build_structure(args.run_root, args.base_url))
    else:
        if args.output is None:
            raise RuntimeError("--output is required for database")
        metrics = build_database(args.run_root, args.output)
    run.log(metrics)
    run.finish()
    print(json.dumps({"stage": args.stage, "status": "completed", **metrics}))
