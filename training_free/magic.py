"""MAGIC's unified graph and narrative chains with local Qwen endpoints."""

import base64
import copy
import json
import pickle
import sys
import threading
from pathlib import Path
from string import Template
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

import av
import numpy as np
from openai import APIError, ContentFilterFinishReasonError, LengthFinishReasonError, OpenAI
from pydantic import BaseModel, ValidationError, model_validator

from training_free.normalize import normalize_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/magic_video/src"))

from worldmm.llm.templates.memory_reasoning_unified_egolife import prompt_template as reasoning_template
from worldmm.llm.templates.qa_egolife_chain import prompt_template as qa_template
from worldmm.memory.episodic.memory import CaptionEntry
from worldmm.memory.memory import WorldMemory, _TIME_EXTRACT_TEMPLATE
from worldmm.memory.unified.memory import UnifiedMemory
from worldmm.memory.unified.nodes import NodeType
from worldmm.memory.utils import RetrievedItem, transform_timestamp
from worldmm.memory.visual.memory import FrameEntry


MODEL = "Qwen/Qwen3.8-27B-FP8"
EMBEDDING_MODEL = "qwen3-vl-embedding-8b"
ARTIFACTS = {}
ARTIFACT_LOCK = threading.Lock()


class MagicDecision(BaseModel):
    decision: Literal["search", "answer"]
    search_query: str | None

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "search" and (self.search_query is None or not self.search_query.strip()):
            raise ValueError("A search decision requires a nonempty search_query")
        if self.decision == "answer" and self.search_query is not None:
            raise ValueError("An answer decision requires search_query=null")
        return self


class TimeReference(BaseModel):
    start: float | None
    end: float | None

    @model_validator(mode="after")
    def validate_interval(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be null or both be seconds")
        if self.start is not None and self.start >= self.end:
            raise ValueError("start must be less than end")
        return self


class MagicOutputError(ValueError):
    """A model response that the next native reasoning round can correct."""


class MagicAPI:
    def __init__(self, run_dir, base_url, embedding_url):
        self.run_dir = run_dir
        self.traces = run_dir / "traces"
        self.traces.mkdir(parents=True, exist_ok=True)
        self.images = run_dir / "frames"
        self.images.mkdir(parents=True, exist_ok=True)
        self.llm = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=0)
        self.embeddings = OpenAI(api_key="EMPTY", base_url=embedding_url, max_retries=0)
        self.last_assistant = None

    def complete(self, messages, response_format=None):
        self.last_assistant = None
        transport = copy.deepcopy(messages)
        recorded = copy.deepcopy(messages)
        for sent, saved in zip(transport, recorded):
            if not isinstance(sent["content"], list):
                continue
            for index, part in enumerate(sent["content"]):
                if part["type"] == "image":
                    frame = part["image"]
                    image_path = Path(frame.info["magic_image_path"])
                    image_bytes = image_path.read_bytes()
                    sent["content"][index] = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")}}
                    saved["content"][index] = {"type": "image_url", "image_url": {"url": str(image_path)}, "transport": "data:image/jpeg;base64"}
        request = {"model": MODEL, "messages": recorded, "reasoning_effort": "xhigh"}
        if response_format is not None:
            request["response_format"] = {"type": "json_schema", "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema(), "strict": True}}
        trace = {"request": request, "response": None, "error": None}
        print(f"MAGIC LLM request: model={MODEL}, messages={len(messages)}, schema={response_format.__name__ if response_format else None}", flush=True)
        try:
            if response_format is None:
                raw = self.llm.chat.completions.with_raw_response.create(model=MODEL, messages=transport, reasoning_effort="xhigh")
            else:
                raw = self.llm.chat.completions.with_raw_response.parse(model=MODEL, messages=transport, reasoning_effort="xhigh", response_format=response_format)
            trace["response"] = raw.json()
            message = trace["response"]["choices"][0]["message"]
            if not isinstance(message.get("reasoning_content"), str) or not message["reasoning_content"].strip():
                raise RuntimeError("MAGIC model response is missing actual reasoning_content")
            self.last_assistant = {"role": "assistant", "content": message["content"], "reasoning_content": message["reasoning_content"]}
            completion = raw.parse()
            if completion.choices[0].message.refusal:
                raise MagicOutputError(completion.choices[0].message.refusal)
            if not message["content"] or not message["content"].strip():
                raise MagicOutputError("MAGIC model returned empty content")
            if response_format is None:
                return message["content"]
            return completion.choices[0].message.parsed
        except Exception as exc:
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, APIError):
                trace["error"]["body"] = exc.body
            print(f"MAGIC LLM error: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            (self.traces / f"magic-{uuid4().hex}.json").write_text(json.dumps(trace, ensure_ascii=False))

    def encode_text(self, texts):
        single = isinstance(texts, str)
        inputs = [normalize_text(texts)] if single else [normalize_text(text) for text in texts]
        request = {"model": EMBEDDING_MODEL, "input": inputs, "encoding_format": "float"}
        trace = {"request": request, "response": None, "error": None}
        print(f"MAGIC embeddings request: inputs={len(inputs)}", flush=True)
        try:
            raw = self.embeddings.embeddings.with_raw_response.create(**request)
            trace["response"] = raw.json()
            result = raw.parse()
            vectors = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in vectors] != list(range(len(inputs))):
                raise ValueError("MAGIC embedding response indices do not match requested inputs")
            matrix = np.asarray([item.embedding for item in vectors], dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
                raise ValueError("MAGIC embedding response contains invalid vectors")
            return matrix[0] if single else matrix
        except Exception as exc:
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, APIError):
                trace["error"]["body"] = exc.body
            print(f"MAGIC embeddings error: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            (self.traces / f"magic-{uuid4().hex}.json").write_text(json.dumps(trace, ensure_ascii=False))

    encode_vis_query = encode_text


class MagicFrames:
    def __init__(self, api, artifact_root):
        self.api = api
        self.artifact_root = artifact_root

    def _extract_frames(self, video_path, fps=1.0, max_frames=64, start_sec=None, end_sec=None):
        trace = {"request": {"tool": "extract_frames", "video_path": video_path, "fps": fps, "max_frames": max_frames, "start_sec": start_sec, "end_sec": end_sec}, "response": None, "error": None}
        print(f"MAGIC frames request: {video_path}, max_frames={max_frames}", flush=True)
        try:
            physical_path = self.artifact_root / "videos" / Path(video_path).relative_to("data/EgoLife")
            with av.open(str(physical_path)) as container:
                stream = container.streams.video[0]
                video_fps = float(stream.average_rate)
                total_frames = stream.frames
                if video_fps <= 0 or total_frames <= 0:
                    raise ValueError(f"Invalid video frame metadata: {physical_path}, fps={video_fps}, frames={total_frames}")
                duration = total_frames / video_fps
                start = 0.0 if start_sec is None else start_sec
                end = duration if end_sec is None else end_sec
                start = max(0.0, min(start, duration))
                end = max(start, min(end, duration))
                start_frame = int(start * video_fps)
                end_frame = min(int(end * video_fps), total_frames - 1)
                if start_frame >= end_frame:
                    raise ValueError(f"Empty video sampling interval: {physical_path}, frames={start_frame}:{end_frame}")
                frame_interval = max(1, int(video_fps / fps) if fps > 0 else int(video_fps))
                frame_indices = list(range(start_frame, end_frame + 1, frame_interval))
                if max_frames is not None and len(frame_indices) > max_frames:
                    frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int).tolist()
                wanted = set(frame_indices)
                decoded = {}
                metadata = {}
                for frame_index, frame in enumerate(container.decode(stream)):
                    if frame_index in wanted:
                        image = frame.to_image()
                        image_path = self.api.images / f"magic-{uuid4().hex}.jpg"
                        image.save(image_path, format="JPEG", quality=90)
                        image.info["magic_image_path"] = str(image_path)
                        decoded[frame_index] = FrameEntry(video_path=video_path, frame_index=frame_index, timestamp_sec=frame_index / video_fps, frame=image)
                        metadata[frame_index] = {"frame_index": frame_index, "timestamp_sec": frame_index / video_fps, "pts": frame.pts, "time_base": str(frame.time_base), "image_path": str(image_path), "size": list(image.size)}
                    if frame_index == frame_indices[-1]:
                        break
                frames = [decoded[index] for index in frame_indices]
                trace["response"] = {"physical_path": str(physical_path), "video_fps": video_fps, "total_frames": total_frames, "frames": [metadata[index] for index in frame_indices]}
                return frames
        except Exception as exc:
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"MAGIC frames error: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            (self.api.traces / f"magic-{uuid4().hex}.json").write_text(json.dumps(trace, ensure_ascii=False))


def load_magic_artifacts(artifact_root):
    with ARTIFACT_LOCK:
        if artifact_root in ARTIFACTS:
            return ARTIFACTS[artifact_root]
        source = artifact_root / "magic"
        print(f"MAGIC loading artifacts: {source}", flush=True)
        memory = UnifiedMemory(None, str(source / "graph.pkl"), chain_mode="facts")
        memory.load(topic_chain_facts_path=str(source / "topic_chains/topic_chains.json"), event_chain_path=str(source / "event_chains/step3_enriched_chains.json"))
        captions = {}
        for granularity in ("30sec", "3min", "10min", "1h"):
            rows = json.loads((source / "captions" / f"A1_JAKE_{granularity}.json").read_text())
            captions[granularity] = [CaptionEntry(id=f"{granularity}_{index}", text=row["text"], start_time=str(row["start_time"]), end_time=str(row["end_time"]), date=row["date"], granularity=granularity, video_path=row["video_path"]) for index, row in enumerate(rows)]
        with (source / "visual_embeddings.pkl").open("rb") as handle:
            visual_embeddings = pickle.load(handle)
        for node in memory._full_graph.nodes.values():
            if node.node_type == NodeType.VISUAL_CLIP:
                original_path = node.metadata["video_path"]
                if not np.array_equal(node.visual_embedding, visual_embeddings[original_path]):
                    raise ValueError(f"Graph visual embedding differs from offline visual_embeddings.pkl: {node.id}")
        ARTIFACTS[artifact_root] = {"graph": memory._full_graph, "topics": memory._topic_facts, "events": memory._event_chains, "captions": captions}
        return ARTIFACTS[artifact_root]


class MagicMemory(WorldMemory):
    """Independent online state; inherited formatting reads shared artifact data."""

    def __init__(self, api, artifact_root, artifacts):
        self.episodic_memory = SimpleNamespace(captions=artifacts["captions"])
        self.unified_memory = UnifiedMemory(api, str(artifact_root / "magic/graph.pkl"), chain_mode="facts")
        self.unified_memory._full_graph = artifacts["graph"]
        self.unified_memory._topic_facts = artifacts["topics"]
        self.unified_memory._event_chains = artifacts["events"]
        self.visual_memory = MagicFrames(api, artifact_root)


def run_magic(question: str, query_time: int, run_dir: Path, artifact_root: Path, base_url: str, embedding_url: str) -> str:
    api = MagicAPI(Path(run_dir), base_url, embedding_url)
    artifact_root = Path(artifact_root).resolve()
    question = normalize_text(question)
    initialization = {"request": {"tool": "index", "artifact_root": str(artifact_root), "query_time": query_time}, "response": None, "error": None}
    try:
        memory = MagicMemory(api, artifact_root, load_magic_artifacts(artifact_root))
        memory.unified_memory.index(query_time)
        active_graph = memory.unified_memory._active_graph
        initialization["response"] = {"graph_stats": active_graph.stats(), "semantic_snapshot_timestamps": sorted({node.start_ts for node in active_graph.nodes.values() if node.node_type == NodeType.SEMANTIC}), "text_matrix_shape": list(active_graph._text_emb_matrix.shape), "visual_matrix_shape": list(active_graph._vis_emb_matrix.shape)}
    except Exception as exc:
        initialization["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"MAGIC initialization error: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        (api.traces / f"magic-{uuid4().hex}.json").write_text(json.dumps(initialization, ensure_ascii=False))
    reasoning_prompt = copy.deepcopy(reasoning_template)
    reasoning_prompt[0]["content"] = reasoning_prompt[0]["content"].replace('If decision = "answer": Do NOT include "search_query".', 'If decision = "answer": Set "search_query" to null.')
    final_prompt = copy.deepcopy(qa_template)
    final_prompt[0]["content"] = final_prompt[0]["content"].replace("Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.", "Your task is to answer open-ended questions based on this accumulated context.").replace("Provide your answer as a single letter (A, B, C, or D) based on the evidence.", "Provide a direct, brief, factual answer to the question based on the evidence.")
    full_query = f"Query: {question}\nCurrent time: {transform_timestamp(str(query_time))}"
    conversation = []
    round_history = []
    retrieved_items = []
    retrieved_node_ids = set()
    injected_chain_facts = set()
    total_visual_frames = 0
    error_count = 0
    model_errors = (ValidationError, json.JSONDecodeError, MagicOutputError, LengthFinishReasonError, ContentFilterFinishReasonError)
    try:
        for round_num in range(1, 6):
            if error_count >= 5:
                raise RuntimeError("MAGIC exhausted max_errors=5")
            history = memory._format_round_history(round_history)
            conversation.append({"role": "user", "content": f"{full_query}\n\nRound History:\n{history}\n\nTask:\nDecide whether to 'search' (retrieve more evidence) or 'answer' (sufficient evidence accumulated)."})
            try:
                decision = api.complete(reasoning_prompt + conversation, MagicDecision)
            except model_errors as exc:
                if api.last_assistant is not None:
                    conversation.append(api.last_assistant)
                conversation.append({"role": "user", "content": f"The reasoning response failed validation: {type(exc).__name__}: {exc}. Correct this error on the next round."})
                error_count += 1
                if round_num == 5:
                    raise
                continue
            conversation.append(api.last_assistant)
            if decision.decision == "answer":
                break
            search_query = normalize_text(decision.search_query)
            time_prompt = Template(_TIME_EXTRACT_TEMPLATE).substitute(query=search_query).replace("null", '{"start": null, "end": null}')
            conversation.append({"role": "user", "content": time_prompt})
            try:
                time_reference = api.complete([{"role": "system", "content": "Extract the video time reference using the supplied schema."}] + conversation, TimeReference)
            except model_errors as exc:
                if api.last_assistant is not None:
                    conversation.append(api.last_assistant)
                conversation.append({"role": "user", "content": f"Time extraction failed validation: {type(exc).__name__}: {exc}. Correct this error on the next round."})
                error_count += 1
                if round_num == 5:
                    raise
                continue
            conversation.append(api.last_assistant)
            request = {"tool": "retrieve", "query": search_query, "query_time": query_time, "time_reference": time_reference.model_dump(), "top_k": 16, "alpha": 0.7, "seed_top_k": 5, "damping": 0.85, "bm25_weight": 0.3, "chain_min_hits": 1, "chain_max_topics": 3, "chain_max_events": 1, "chain_topic_sim": 0.7, "chain_event_sim": 0.7, "chain_event_min_hits": 1, "chain_event_granularities": ["30sec", "3min"]}
            trace = {"request": request, "response": None, "error": None}
            print(f"MAGIC retrieval: round={round_num}, query={search_query}", flush=True)
            try:
                if time_reference.start is not None:
                    content = memory.retrieve_by_time(time_reference.start, time_reference.end)
                    if content:
                        retrieved_items.append(RetrievedItem("episodic", content, search_query, round_num))
                    round_history.append({"round_num": round_num, "decision": "search", "memory_type": "temporal", "search_query": search_query, "retrieved_content": content if content else "[No results]"})
                    trace["response"] = {"temporal_content": content}
                    continue
                raw_results = memory.unified_memory.retrieve(search_query, top_k=16, target_node_types=[NodeType.EPISODE, NodeType.SEMANTIC, NodeType.VISUAL_CLIP], alpha=0.7, seed_top_k=5, damping=0.85, bm25_weight=0.3)
                trace["response"] = {"raw_results": [{"id": node.id, "node_type": node.node_type.value, "start_ts": node.start_ts, "end_ts": node.end_ts, "text": node.text, "metadata": node.metadata, "score": score} for node, score in raw_results]}
                new_results = [(node, score) for node, score in raw_results if node.id not in retrieved_node_ids]
                trace["response"]["new_node_ids"] = [node.id for node, _ in new_results]
                retrieved_node_ids.update(node.id for node, _ in new_results)
                if not new_results:
                    round_history.append({"round_num": round_num, "decision": "search", "memory_type": "unified", "search_query": search_query, "retrieved_content": "[No new results]"})
                    continue
                new_items = memory.unified_memory.format_results(new_results, memory.visual_memory, max_frames_per_clip=16, max_total_frames=64, _current_total_frames=total_visual_frames)
                round_frame_count = sum(len(item.content) for item in new_items if item.memory_type == "visual")
                total_visual_frames += round_frame_count
                chain_facts = memory.unified_memory.get_chain_facts_for_results(new_results, query_time=query_time, query_text=question, embedding_model=api, min_hits=1, max_topic_chains=3, max_event_chains=1, topic_sim_threshold=0.7, event_chain_sim_threshold=0.7, event_chain_min_hits=1, event_chain_granularities=("30sec", "3min"))
                trace["response"]["chain_facts"] = chain_facts
                fresh_facts = [fact for fact in chain_facts if (fact["label"], fact["time"], fact["fact"]) not in injected_chain_facts]
                for fact in fresh_facts:
                    injected_chain_facts.add((fact["label"], fact["time"], fact["fact"]))
                    content = f"[{fact['time']} - {fact['end_time']}] {fact['fact']}" if fact["chain_type"] == "event" else f"[{fact['time']}] {fact['fact']}"
                    new_items.append(RetrievedItem(f"{fact['chain_type']}_chain", content, search_query, round_num, fact["label"]))
                for item in new_items:
                    item.query = search_query
                    item.round_num = round_num
                retrieved_items.extend(new_items)
                trace["response"]["items"] = [{"memory_type": item.memory_type, "content": item.content if isinstance(item.content, str) else [image.info["magic_image_path"] for image in item.content], "label": item.label} for item in new_items]
                summary_parts = memory._render_retrieved_items_interleaved(new_items)
                summary = "\n".join(part["text"] for part in summary_parts if part["type"] == "text")
                if round_frame_count:
                    summary += f"\n[{round_frame_count} image frames retrieved]"
                round_history.append({"round_num": round_num, "decision": "search", "memory_type": "unified", "search_query": search_query, "retrieved_content": summary})
                trace["response"]["round_history"] = round_history[-1]
            except Exception as exc:
                trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"MAGIC retrieval error: {type(exc).__name__}: {exc}", flush=True)
                raise
            finally:
                (api.traces / f"magic-{uuid4().hex}.json").write_text(json.dumps(trace, ensure_ascii=False))
        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
        qa_content.extend(memory._render_retrieved_items_interleaved(retrieved_items))
        return api.complete(final_prompt + conversation + [{"role": "user", "content": qa_content}])
    finally:
        memory.unified_memory.cleanup()
        api.llm.close()
        api.embeddings.close()
