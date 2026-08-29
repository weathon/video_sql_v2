import argparse
import asyncio
import base64
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import duckdb
import wandb
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import BaseModel, RootModel, field_validator, model_validator

from src.preprocess.magic_index_prompts import ACTIVITY, ASR_SYSTEM, ENRICH, ENTITY_FILTER, EVENT_CHAIN, MERGE_SYSTEM, OPENIE_EXAMPLE, OPENIE_EXAMPLE_OUTPUT, OPENIE_SYSTEM, SEMANTIC_EXAMPLE_INPUT, SEMANTIC_EXAMPLE_OUTPUT, SEMANTIC_SYSTEM, TOPIC, TOPIC_SYSTEM, VIDEO_CAPTION, WEEK_EVENT_CHAIN_ROUND2


SEMANTIC_CONSOLIDATION_BATCH_SIZE = 4000


class CombinedOpenIE(BaseModel):
    named_entities: list[str]
    triples: list[tuple[str, str, str]]


class SemanticOutput(BaseModel):
    semantic_triples: list[tuple[str, str, str]]
    episodic_evidence: list[list[int]]

    @model_validator(mode="after")
    def validate_alignment(self):
        if len(self.semantic_triples) != len(self.episodic_evidence):
            raise ValueError("semantic_triples and episodic_evidence must have equal lengths")
        return self


class TripleList(RootModel[list[tuple[str, str, str]]]):
    pass


class EntityList(RootModel[list[str]]):
    pass


class TopicFact(BaseModel):
    time: str
    fact: str

    @field_validator("time")
    @classmethod
    def validate_time(cls, value):
        parse_hms(value)
        return value


class TopicFacts(RootModel[list[TopicFact]]):
    pass


class WeekTopicFact(BaseModel):
    time: str
    fact: str

    @field_validator("time")
    @classmethod
    def validate_time(cls, value):
        if re.fullmatch(r"DAY\d+ \d{2}:\d{2}(?::\d{2})?", value, re.IGNORECASE) is None:
            raise ValueError(f"invalid EgoLife time: {value}")
        return value


class WeekTopicFacts(RootModel[list[WeekTopicFact]]):
    pass


class ActivityItem(BaseModel):
    name: str
    time_range: str
    summary: str
    key_entities: list[str]

    @field_validator("time_range")
    @classmethod
    def validate_time_range(cls, value):
        parse_time_range(value)
        return value


class LocatedActivityItem(ActivityItem):
    location: str


class Activities(RootModel[list[LocatedActivityItem]]):
    pass


class WeekActivities(RootModel[list[ActivityItem]]):
    pass


class EventStep(BaseModel):
    start_time: str
    end_time: str
    description: str

    @model_validator(mode="after")
    def validate_times(self):
        if parse_hms(self.end_time) < parse_hms(self.start_time):
            raise ValueError("event step ends before it starts")
        return self


class EventChainItem(BaseModel):
    name: str
    type: str
    steps: list[EventStep]
    key_entities: list[str]


class EventChains(RootModel[list[EventChainItem]]):
    pass


class WeekEventStep(EventStep):
    day: str

    @field_validator("day")
    @classmethod
    def validate_day(cls, value):
        if re.fullmatch(r"DAY\d+", value, re.IGNORECASE) is None:
            raise ValueError(f"invalid EgoLife day: {value}")
        return value


class WeekEventChainItem(BaseModel):
    name: str
    steps: list[WeekEventStep]
    key_entities: list[str]


class WeekEventChains(RootModel[list[WeekEventChainItem]]):
    pass


class EnrichedDescription(BaseModel):
    description: str


class SampleInputError(RuntimeError):
    pass


def parse_hms(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid HH:MM:SS value: {value}")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_time_range(value: str) -> tuple[float, float]:
    parts = value.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid time range: {value}")
    start, end = parse_hms(parts[0]), parse_hms(parts[1])
    if end < start:
        raise ValueError(f"time range ends before it starts: {value}")
    return start, end


def parse_week_timestamp(value: str) -> tuple[str, float]:
    day, time_value = value.split(" ", 1)
    if time_value.count(":") == 1:
        time_value += ":00"
    return day.lower(), parse_hms(time_value)


def parse_egolife_filename_clock(path: Path) -> float:
    value = path.stem.rsplit("_", 1)[1]
    if re.fullmatch(r"\d{8}", value) is None:
        raise ValueError(f"invalid EgoLife filename clock: {path}")
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:6]) + int(value[6:]) / 100


def week_clock_to_timeline(timeline: dict, day: str, clock_sec: float) -> float:
    for segment in timeline["clock_segments"][day.lower()]:
        if segment["clock_start_sec"] <= clock_sec <= segment["clock_end_sec"]:
            return segment["timeline_start_sec"] + clock_sec - segment["clock_start_sec"]
    raise ValueError(f"EgoLife time is outside real video footage: {day.upper()} {format_hms(clock_sec)}")


def format_hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def stage_path(root: Path, stage: str, key: str) -> Path:
    return root / "shards" / stage / f"{safe_key(key)}.json"


def append_failure(root: Path, stage: str, key: str, error: str, fill_round: bool) -> None:
    path = root / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "key": key, "fill_round": fill_round, "error": error}, ensure_ascii=False) + "\n")


def prepare_video_data_url(clip_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(clip_path), "-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast", "-an", "-y", temp_file.name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SampleInputError(f"ffmpeg failed clip={clip_path} error={result.stderr.strip()}")
        data = Path(temp_file.name).read_bytes()
        if not data:
            raise SampleInputError(f"ffmpeg wrote an empty clip: {clip_path}")
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


class LocalModel:
    def __init__(self, base_url: str, model: str, max_retries: int, root: Path):
        self.client = AsyncOpenAI(api_key="dummy", base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.root = root

    async def generate(self, stage: str, key: str, messages: list[dict], trace_messages: list[dict], response_format=None, result_validator=None):
        attempt_messages = list(messages)
        attempt_trace_messages = list(trace_messages)
        connection_errors = 0
        for attempt in range(1, self.max_retries + 1):
            trace_path = self.root / "traces" / stage / safe_key(key) / f"attempt-{attempt:02d}.json"
            try:
                if response_format is None:
                    response = await self.client.chat.completions.create(model=self.model, messages=attempt_messages, reasoning_effort="low", extra_body={"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}})
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        correction = [{"role": "assistant", "content": content or ""}, {"role": "user", "content": "The previous response was empty. Produce the requested nonempty result."}]
                        write_json(trace_path, {"request": attempt_trace_messages, "response": response.model_dump(mode="json"), "error": "empty response"})
                        attempt_messages = attempt_messages + correction
                        attempt_trace_messages = attempt_trace_messages + correction
                        continue
                    write_json(trace_path, {"request": attempt_trace_messages, "response": response.model_dump(mode="json")})
                    return content.strip()
                response = await self.client.chat.completions.parse(model=self.model, messages=attempt_messages, response_format=response_format, reasoning_effort="low", extra_body={"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}})
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    correction = [{"role": "assistant", "content": response.choices[0].message.content or ""}, {"role": "user", "content": "The previous response did not satisfy the required schema. Return a valid structured response."}]
                    write_json(trace_path, {"request": attempt_trace_messages, "response": response.model_dump(mode="json"), "error": "parsed response is empty"})
                    attempt_messages = attempt_messages + correction
                    attempt_trace_messages = attempt_trace_messages + correction
                    continue
                if result_validator is not None:
                    try:
                        result_validator(parsed)
                    except ValueError as error:
                        correction = [
                            {"role": "assistant", "content": response.choices[0].message.content or ""},
                            {"role": "user", "content": f"The previous response failed validation: {error}. Return a response that satisfies the required schema."},
                        ]
                        write_json(trace_path, {"request": attempt_trace_messages, "response": response.model_dump(mode="json"), "error_type": type(error).__name__, "error": str(error)})
                        attempt_messages = attempt_messages + correction
                        attempt_trace_messages = attempt_trace_messages + correction
                        continue
                write_json(trace_path, {"request": attempt_trace_messages, "response": response.model_dump(mode="json")})
                return parsed
            except APIConnectionError as error:
                connection_errors += 1
                write_json(trace_path, {"request": attempt_trace_messages, "error_type": type(error).__name__, "error": str(error)})
            except APIError as error:
                write_json(trace_path, {"request": attempt_trace_messages, "error_type": type(error).__name__, "status_code": error.status_code, "error": str(error)})
                if error.status_code is not None and 400 <= error.status_code < 500 and error.status_code not in (408, 409, 429):
                    raise
            except ValueError as error:
                correction = [{"role": "user", "content": f"The previous response failed validation: {error}. Return a response that satisfies the required schema."}]
                write_json(trace_path, {"request": attempt_trace_messages, "error_type": type(error).__name__, "error": str(error)})
                attempt_messages = attempt_messages + correction
                attempt_trace_messages = attempt_trace_messages + correction
        if connection_errors == self.max_retries:
            raise RuntimeError(f"model service unavailable for all {self.max_retries} attempts stage={stage} key={key}")
        return None


async def run_jobs(stage: str, jobs: list[dict], concurrency: int, root: Path, worker) -> int:
    pending = [job for job in jobs if not stage_path(root, stage, job["key"]).is_file()]
    if not pending:
        print(f"stage_complete stage={stage} existing={len(jobs)} missing=0")
        wandb.log({"index/stage": stage, "index/completed": len(jobs), "index/total": len(jobs), "index/missing": 0})
        return 0
    for fill_round in (False, True):
        current = [job for job in pending if not stage_path(root, stage, job["key"]).is_file()]
        for start in range(0, len(current), concurrency):
            batch = current[start:start + concurrency]
            await asyncio.gather(*(worker(job, fill_round) for job in batch))
            completed = min(start + len(batch), len(current))
            if completed % 250 == 0 or completed == len(current):
                print(f"stage_progress stage={stage} fill_round={fill_round} completed={completed} total={len(current)}")
                wandb.log({"index/stage": stage, "index/fill_round": int(fill_round), "index/completed": completed, "index/total": len(current)})
        pending = [job for job in jobs if not stage_path(root, stage, job["key"]).is_file()]
        if not pending:
            break
    print(f"stage_complete stage={stage} total={len(jobs)} missing={len(pending)}")
    wandb.log({"index/stage": stage, "index/completed": len(jobs) - len(pending), "index/total": len(jobs), "index/missing": len(pending)})
    if pending:
        raise RuntimeError(f"stage still has missing samples after fill round stage={stage} missing={len(pending)}")
    return len(pending)


def load_catalog(database_path: Path, dataset_video_root: Path, asr_root: Path, clip_dir: Path, root: Path) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    video_rows = connection.execute("SELECT video_id, sequence_index, source_split, filename, duration_sec, global_start, global_end FROM videos ORDER BY sequence_index").fetchall()
    clip_rows = connection.execute("SELECT row_id, video_id, start_sec, end_sec, global_start, global_end FROM video_clips ORDER BY video_id, start_sec").fetchall()
    connection.close()
    sources = {}
    origins = {}
    for video_id, sequence_index, source_split, filename, duration_sec, global_start, global_end in video_rows:
        sources[video_id] = {"video_id": video_id, "sequence_index": int(sequence_index), "source_split": source_split, "filename": filename, "duration_sec": float(duration_sec), "global_start_sec": global_start.total_seconds(), "global_end_sec": global_end.total_seconds()}
        origins[source_split] = min(origins.get(source_split, global_start.total_seconds()), global_start.total_seconds())

    clips_by_source = defaultdict(list)
    clips_by_timeline = defaultdict(list)
    for row_id, source_video_id, old_start, old_end, global_start, global_end in clip_rows:
        source = sources[source_video_id]
        timeline_id = source["source_split"]
        clip_index = int(row_id.rsplit(":", 1)[1])
        clip_path = clip_dir / source_video_id / f"{clip_index:06d}.mp4"
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        clip = {
            "row_id": row_id,
            "video_id": timeline_id,
            "source_video_id": source_video_id,
            "clip_index": clip_index,
            "start_sec": global_start.total_seconds() - origins[timeline_id],
            "end_sec": global_end.total_seconds() - origins[timeline_id],
            "clip_path": str(clip_path),
            "filename": f"{source_video_id}/{clip_index:06d}.mp4",
        }
        clips_by_source[source_video_id].append(clip)
        clips_by_timeline[timeline_id].append(clip)

    catalog = {"videos": [], "clips": []}
    for timeline_id in ("month", "day", "week"):
        clock_segments = {}
        timeline_clips = sorted(clips_by_timeline[timeline_id], key=lambda item: item["start_sec"])
        if not timeline_clips:
            raise RuntimeError(f"timeline has no clips: {timeline_id}")
        raw_segments = []
        timeline_sources = sorted((source for source in sources.values() if source["source_split"] == timeline_id), key=lambda source: source["sequence_index"])
        for source in timeline_sources:
            source_offset = source["global_start_sec"] - origins[timeline_id]
            if timeline_id == "month":
                transcript_path = asr_root / "month" / source["video_id"] / "whisper" / f"{source['video_id']}_whisper.json"
                for segment in read_json(transcript_path):
                    raw_segments.append({"start_sec": source_offset + float(segment["start_sec"]), "end_sec": source_offset + float(segment["end_sec"]), "text": segment["text"], "source_video_id": source["video_id"]})
            elif timeline_id == "day":
                transcript_path = asr_root / "day" / source["video_id"] / "whisper" / f"{source['video_id']}_whisper.json"
                for segment in read_json(transcript_path):
                    raw_segments.append({"start_sec": source_offset + float(segment["start_sec"]), "end_sec": source_offset + float(segment["end_sec"]), "text": segment["text"], "source_video_id": source["video_id"]})
            else:
                source_paths = sorted((dataset_video_root / source["filename"]).glob("*.mp4"))
                source_clips = sorted(clips_by_source[source["video_id"]], key=lambda item: item["start_sec"])
                if len(source_paths) != len(source_clips):
                    raise RuntimeError(f"week source/clip count mismatch video={source['video_id']} sources={len(source_paths)} clips={len(source_clips)}")
                clock_segments[source["video_id"]] = []
                for source_path, clip in zip(source_paths, source_clips, strict=True):
                    clock_start_sec = parse_egolife_filename_clock(source_path)
                    clock_end_sec = clock_start_sec + clip["end_sec"] - clip["start_sec"]
                    clip["week_day"] = source["video_id"]
                    clip["local_clock_start_sec"] = clock_start_sec
                    clock_segments[source["video_id"]].append({
                        "clock_start_sec": clock_start_sec,
                        "clock_end_sec": clock_end_sec,
                        "timeline_start_sec": clip["start_sec"],
                        "timeline_end_sec": clip["end_sec"],
                    })
                    transcript_path = asr_root / "week" / source["video_id"] / source_path.stem / "whisper" / f"{source_path.stem}_whisper.json"
                    for segment in read_json(transcript_path):
                        raw_segments.append({"start_sec": clip["start_sec"] + float(segment["start_sec"]), "end_sec": clip["start_sec"] + float(segment["end_sec"]), "text": segment["text"], "source_video_id": source["video_id"]})
        raw_segments.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
        timeline = {
            "video_id": timeline_id,
            "sequence_index": len(catalog["videos"]),
            "source_split": timeline_id,
            "filename": timeline_id,
            "duration_sec": max(clip["end_sec"] for clip in timeline_clips),
            "source_offsets": {
                source["video_id"]: source["global_start_sec"] - origins[timeline_id]
                for source in timeline_sources
            },
            "source_durations": {
                source["video_id"]: source["duration_sec"]
                for source in timeline_sources
            },
            "clock_segments": clock_segments,
            "clips": timeline_clips,
        }
        catalog["videos"].append(timeline)
        catalog["clips"].extend(timeline_clips)
        write_json(root / "raw_asr" / f"{timeline_id}.json", raw_segments)
    write_json(root / "catalog.json", catalog)
    return catalog


def add_aligned_transcripts(catalog: dict, root: Path) -> None:
    for timeline in catalog["videos"]:
        segments = read_json(root / "raw_asr" / f"{timeline['video_id']}.json")
        segment_index = 0
        for clip in timeline["clips"]:
            texts = []
            while segment_index < len(segments) and (segments[segment_index]["start_sec"] + segments[segment_index]["end_sec"]) / 2 < clip["start_sec"]:
                segment_index += 1
            scan = segment_index
            while scan < len(segments):
                midpoint = (segments[scan]["start_sec"] + segments[scan]["end_sec"]) / 2
                if midpoint >= clip["end_sec"]:
                    break
                if midpoint >= clip["start_sec"] and segments[scan]["text"].strip():
                    texts.append(segments[scan]["text"].strip())
                scan += 1
            clip["asr_text"] = " ".join(texts)


async def run_caption_stages(catalog: dict, model: LocalModel, concurrency: int, root: Path) -> dict[str, list[dict]]:
    vlm_jobs = [{"key": clip["row_id"], "clip": clip} for clip in catalog["clips"]]

    async def caption_video(job, fill_round):
        clip = job["clip"]
        try:
            data_url = await asyncio.to_thread(prepare_video_data_url, Path(clip["clip_path"]))
            messages = [{"role": "user", "content": [{"type": "text", "text": VIDEO_CAPTION}, {"type": "video_url", "video_url": {"url": data_url}}]}]
            text = await model.generate("vlm", job["key"], messages, messages)
        except SampleInputError as error:
            append_failure(root, "vlm", job["key"], str(error), fill_round)
            return
        if text is None:
            append_failure(root, "vlm", job["key"], "ten model attempts failed", fill_round)
            return
        output = {"row_id": f"{clip['row_id']}:captions:vlm", "video_id": clip["video_id"], "source_video_id": clip["source_video_id"], "source": "vlm", "granularity_sec": 30, "start_sec": clip["start_sec"], "end_sec": clip["end_sec"], "text": text}
        if clip["video_id"] == "week":
            output["week_day"] = clip["week_day"]
            output["local_clock_start_sec"] = clip["local_clock_start_sec"]
        write_json(stage_path(root, "vlm", job["key"]), output)

    await run_jobs("vlm", vlm_jobs, concurrency, root, caption_video)

    asr_jobs = [{"key": clip["row_id"], "clip": clip} for clip in catalog["clips"]]

    async def rewrite_asr(job, fill_round):
        clip = job["clip"]
        if not clip["asr_text"]:
            output = {"row_id": f"{clip['row_id']}:captions:asr", "video_id": clip["video_id"], "source_video_id": clip["source_video_id"], "source": "asr", "granularity_sec": 30, "start_sec": clip["start_sec"], "end_sec": clip["end_sec"], "text": "", "raw_text": ""}
            if clip["video_id"] == "week":
                output["week_day"] = clip["week_day"]
                output["local_clock_start_sec"] = clip["local_clock_start_sec"]
            write_json(stage_path(root, "asr_rewrite", job["key"]), output)
            return
        if clip["video_id"] == "week":
            local_end_sec = clip["local_clock_start_sec"] + clip["end_sec"] - clip["start_sec"]
            prompt = json.dumps([{
                "start": clip["local_clock_start_sec"],
                "end": local_end_sec,
                "text": clip["asr_text"],
                "type": "transcript",
            }], ensure_ascii=False, indent=2)
        else:
            prompt = f"Raw transcript [{clip['start_sec']:.3f}s - {clip['end_sec']:.3f}s]:\n{clip['asr_text']}"
        messages = [{"role": "system", "content": ASR_SYSTEM[clip["video_id"]]}, {"role": "user", "content": prompt}]
        text = await model.generate("asr_rewrite", job["key"], messages, messages)
        if text is None:
            append_failure(root, "asr_rewrite", job["key"], "ten model attempts failed", fill_round)
            return
        output = {"row_id": f"{clip['row_id']}:captions:asr", "video_id": clip["video_id"], "source_video_id": clip["source_video_id"], "source": "asr", "granularity_sec": 30, "start_sec": clip["start_sec"], "end_sec": clip["end_sec"], "text": text, "raw_text": clip["asr_text"]}
        if clip["video_id"] == "week":
            output["week_day"] = clip["week_day"]
            output["local_clock_start_sec"] = clip["local_clock_start_sec"]
        write_json(stage_path(root, "asr_rewrite", job["key"]), output)

    await run_jobs("asr_rewrite", asr_jobs, concurrency, root, rewrite_asr)

    merge_jobs = [{"key": clip["row_id"], "clip": clip} for clip in catalog["clips"]]

    async def merge_caption(job, fill_round):
        clip = job["clip"]
        vlm_path = stage_path(root, "vlm", job["key"])
        asr_path = stage_path(root, "asr_rewrite", job["key"])
        if not vlm_path.is_file() or not asr_path.is_file():
            append_failure(root, "merge", job["key"], "required VLM or ASR sample is missing", fill_round)
            return
        visual_text = read_json(vlm_path)["text"]
        asr_text = read_json(asr_path)["text"]
        if not asr_text:
            text = visual_text
        else:
            if clip["video_id"] == "week":
                local_end_sec = clip["local_clock_start_sec"] + clip["end_sec"] - clip["start_sec"]
                prompt = json.dumps([
                    {
                        "start": clip["local_clock_start_sec"],
                        "end": local_end_sec,
                        "text": visual_text,
                        "type": "caption",
                    },
                    {
                        "start": clip["local_clock_start_sec"],
                        "end": local_end_sec,
                        "text": asr_text,
                        "type": "transcript",
                    },
                ], ensure_ascii=False, indent=2)
            else:
                prompt = f"Visual: {visual_text}\nTranscript: {asr_text}"
            messages = [{"role": "system", "content": MERGE_SYSTEM[clip["video_id"]]}, {"role": "user", "content": prompt}]
            text = await model.generate("merge", job["key"], messages, messages)
            if text is None:
                append_failure(root, "merge", job["key"], "ten model attempts failed", fill_round)
                return
        output = {"row_id": f"{clip['row_id']}:captions:merged:30", "video_id": clip["video_id"], "source_video_id": clip["source_video_id"], "source": "merged", "granularity_sec": 30, "start_sec": clip["start_sec"], "end_sec": clip["end_sec"], "text": text}
        if clip["video_id"] == "week":
            output["week_day"] = clip["week_day"]
            output["local_clock_start_sec"] = clip["local_clock_start_sec"]
        write_json(stage_path(root, "merge", job["key"]), output)

    await run_jobs("merge", merge_jobs, concurrency, root, merge_caption)

    levels = {"merge": [{"key": clip["row_id"], "video_id": clip["video_id"], "source_video_id": clip["source_video_id"], "start_sec": clip["start_sec"], "end_sec": clip["end_sec"]} for clip in catalog["clips"]]}
    for source_stage, target_stage, granularity, group_size in (("merge", "aggregate_180", 180, 6), ("aggregate_180", "aggregate_600", 600, 3), ("aggregate_600", "aggregate_3600", 3600, 6)):
        jobs = []
        for timeline in catalog["videos"]:
            children = [
                item
                for item in levels[source_stage]
                if item["video_id"] == timeline["video_id"]
            ]
            children.sort(key=lambda item: item["start_sec"])
            if timeline["video_id"] == "week":
                children_by_day = defaultdict(list)
                for child in children:
                    child_row = read_json(stage_path(root, source_stage, child["key"]))
                    children_by_day[child_row["week_day"]].append(child)
                groups = []
                source_ids = sorted(timeline["source_offsets"], key=lambda source_video_id: timeline["source_offsets"][source_video_id])
                for source_video_id in source_ids:
                    day_children = children_by_day[source_video_id]
                    groups.extend(day_children[start:start + group_size] for start in range(0, len(day_children), group_size))
            else:
                groups = [children[start:start + group_size] for start in range(0, len(children), group_size)]
            group_index = 0
            for group in groups:
                jobs.append({
                    "key": f"{timeline['video_id']}:{granularity}:{group_index}",
                    "video_id": timeline["video_id"],
                    "start_sec": group[0]["start_sec"],
                    "end_sec": group[-1]["end_sec"],
                    "children": group,
                    "source_stage": source_stage,
                    "granularity_sec": granularity,
                })
                group_index += 1

        async def aggregate(job, fill_round, stage=target_stage):
            child_rows = []
            for child in job["children"]:
                child_path = stage_path(root, job["source_stage"], child["key"])
                if not child_path.is_file():
                    append_failure(root, stage, job["key"], f"required child is missing: {child['key']}", fill_round)
                    return
                child_rows.append(read_json(child_path))
            if job["video_id"] == "week":
                segments_text = "\n".join(f"[{row['week_day'].upper()} {format_hms(row['local_clock_start_sec'])}]: {row['text']}" for row in child_rows)
            else:
                segments_text = "\n".join(f"[{row['start_sec']:.3f}s - {row['end_sec']:.3f}s]: {row['text']}" for row in child_rows)
            prompt = f"Summarize the following video segment descriptions into a coherent 3-5 sentence summary. Preserve key events, people, and locations:\n\n{segments_text}\n\nSummary:"
            messages = [{"role": "user", "content": prompt}]
            text = await model.generate(stage, job["key"], messages, messages)
            if text is None:
                append_failure(root, stage, job["key"], "ten model attempts failed", fill_round)
                return
            output = {"row_id": f"{job['key']}:captions:merged", "video_id": job["video_id"], "source": "merged", "granularity_sec": job["granularity_sec"], "start_sec": job["start_sec"], "end_sec": job["end_sec"], "text": text}
            if job["video_id"] == "week":
                output["week_day"] = child_rows[0]["week_day"]
                output["local_clock_start_sec"] = child_rows[0]["local_clock_start_sec"]
            write_json(stage_path(root, stage, job["key"]), output)

        await run_jobs(target_stage, jobs, concurrency, root, aggregate)
        levels[target_stage] = [{"key": job["key"], "video_id": job["video_id"], "start_sec": job["start_sec"], "end_sec": job["end_sec"]} for job in jobs]
    return levels


async def run_memory_stages(catalog: dict, levels: dict[str, list[dict]], model: LocalModel, concurrency: int, root: Path) -> dict:
    openie_jobs = [{"key": item["key"], "item": item} for item in levels["merge"] if stage_path(root, "merge", item["key"]).is_file()]

    async def extract_openie(job, fill_round):
        caption = read_json(stage_path(root, "merge", job["key"]))
        messages = [{"role": "system", "content": OPENIE_SYSTEM}, {"role": "user", "content": OPENIE_EXAMPLE}, {"role": "assistant", "content": OPENIE_EXAMPLE_OUTPUT}, {"role": "user", "content": caption["text"]}]
        result = await model.generate("openie", job["key"], messages, messages, CombinedOpenIE)
        if result is None:
            append_failure(root, "openie", job["key"], "ten model attempts failed", fill_round)
            return
        triples = []
        seen = set()
        for triple in result.triples:
            if triple not in seen:
                seen.add(triple)
                triples.append(list(triple))
        chunk_id = "chunk-" + hashlib.md5(caption["text"].encode()).hexdigest()
        write_json(stage_path(root, "openie", job["key"]), {"video_id": caption["video_id"], "caption_row_id": caption["row_id"], "chunk_id": chunk_id, "start_sec": caption["start_sec"], "end_sec": caption["end_sec"], "named_entities": result.named_entities, "triples": triples})

    await run_jobs("openie", openie_jobs, concurrency, root, extract_openie)
    semantic_jobs = [{"key": job["key"]} for job in openie_jobs if stage_path(root, "openie", job["key"]).is_file()]

    async def extract_semantic(job, fill_round):
        openie = read_json(stage_path(root, "openie", job["key"]))
        triples_text = "\n".join(f"{index}. {triple}" for index, triple in enumerate(openie["triples"]))
        messages = [
            {"role": "system", "content": SEMANTIC_SYSTEM},
            {"role": "user", "content": SEMANTIC_EXAMPLE_INPUT},
            {"role": "assistant", "content": SEMANTIC_EXAMPLE_OUTPUT},
            {"role": "user", "content": f"Episodic triples:\n{triples_text}"},
        ]
        result = await model.generate("semantic", job["key"], messages, messages, SemanticOutput)
        if result is None:
            append_failure(root, "semantic", job["key"], "ten model attempts failed", fill_round)
            return
        write_json(stage_path(root, "semantic", job["key"]), {"video_id": openie["video_id"], "chunk_id": openie["chunk_id"], "semantic_triples": [list(triple) for triple in result.semantic_triples], "episodic_evidence": result.episodic_evidence})

    await run_jobs("semantic", semantic_jobs, concurrency, root, extract_semantic)
    consolidation_jobs = []
    for timeline in catalog["videos"]:
        groups = defaultdict(list)
        for job in semantic_jobs:
            path = stage_path(root, "semantic", job["key"])
            if not path.is_file():
                continue
            result = read_json(path)
            if result["video_id"] != timeline["video_id"]:
                continue
            for triple in result["semantic_triples"]:
                subject = triple[0].strip().lower().replace("'", "").replace("-", " ")
                groups[subject].append(triple)
        for subject, triples in groups.items():
            consolidation_jobs.append({"key": f"{timeline['video_id']}:{hashlib.md5(subject.encode()).hexdigest()}", "video_id": timeline["video_id"], "subject": subject, "triples": triples})

    async def consolidate(job, fill_round):
        if len(job["triples"]) == 1:
            triples = job["triples"]
        else:
            current_triples = job["triples"]
            round_index = 0
            while True:
                batches = [current_triples[start:start + SEMANTIC_CONSOLIDATION_BATCH_SIZE] for start in range(0, len(current_triples), SEMANTIC_CONSOLIDATION_BATCH_SIZE)]
                next_triples = []
                for batch_index, batch in enumerate(batches):
                    batch_key = f"{job['key']}:round:{round_index}:batch:{batch_index}"
                    batch_path = stage_path(root, "semantic_consolidation_batches", batch_key)
                    if batch_path.is_file():
                        batch_triples = read_json(batch_path)["consolidated_semantic_triples"]
                    else:
                        formatted = "\n".join(f"{index}. [{triple[0]}, {triple[1]}, {triple[2]}]" for index, triple in enumerate(batch))
                        prompt = f"Below are semantic triples about the same subject. Merge duplicates and remove redundant ones. Keep distinct facts.\n\n{formatted}\n\nReturn only a JSON array of merged triples, each as [subject, predicate, object]."
                        messages = [{"role": "user", "content": prompt}]
                        result = await model.generate("semantic_consolidation", batch_key, messages, messages, TripleList)
                        if result is None or not result.root:
                            append_failure(root, "semantic_consolidation", batch_key, "ten model attempts failed or returned no triples", fill_round)
                            return
                        batch_triples = [list(triple) for triple in result.root]
                        write_json(batch_path, {"video_id": job["video_id"], "subject": job["subject"], "round_index": round_index, "batch_index": batch_index, "input_count": len(batch), "consolidated_semantic_triples": batch_triples})
                    next_triples.extend(batch_triples)
                if len(batches) == 1:
                    triples = next_triples
                    break
                if len(next_triples) >= len(current_triples):
                    raise RuntimeError(f"semantic consolidation did not reduce triples key={job['key']} round={round_index} input={len(current_triples)} output={len(next_triples)}")
                current_triples = next_triples
                round_index += 1
        write_json(stage_path(root, "semantic_consolidation", job["key"]), {"video_id": job["video_id"], "subject": job["subject"], "consolidated_semantic_triples": triples})

    await run_jobs("semantic_consolidation", consolidation_jobs, concurrency, root, consolidate)
    return {"openie_jobs": openie_jobs, "semantic_jobs": semantic_jobs, "consolidation_jobs": consolidation_jobs}


def entity_candidates(catalog: dict, levels: dict[str, list[dict]], memory: dict, root: Path) -> dict[str, dict[str, list[int]]]:
    skip_words = {"ishowspeed", "speed", "streamer", "man", "person", "people", "fan", "fans", "he", "she", "they", "i", "we", "someone", "everyone", "narrator", "video", "stream", "chat", "viewer", "viewers", "true", "false", "yes", "no", "something", "anything", "nothing", "time", "moment", "scene", "clip", "frame"}
    captions = defaultdict(list)
    for item in levels["merge"]:
        path = stage_path(root, "merge", item["key"])
        if path.is_file():
            captions[item["video_id"]].append(read_json(path))
    output = {}
    for timeline in catalog["videos"]:
        timeline_skip_words = skip_words
        if timeline["video_id"] == "week":
            timeline_skip_words = skip_words | {
                "me", "my", "us", "shure", "katrina", "tasha", "lucia",
                "alice", "nicous", "choiszt", "violet", "jack", "liu ting",
                "huixin", "annie", "zitang", "it", "things", "stuff",
            }
        entities = set()
        for job in memory["semantic_jobs"]:
            path = stage_path(root, "semantic", job["key"])
            if not path.is_file():
                continue
            result = read_json(path)
            if result["video_id"] != timeline["video_id"]:
                continue
            for triple in result["semantic_triples"]:
                for entity in (triple[0].lower().strip(), triple[2].lower().strip()):
                    if len(entity) >= 2 and entity not in timeline_skip_words:
                        entities.add(entity)
        mentions = {}
        for entity in entities:
            pattern = re.compile(r"\b" + re.escape(entity) + r"s?\b", re.IGNORECASE)
            indices = [index for index, caption in enumerate(captions[timeline["video_id"]]) if pattern.search(caption["text"])]
            if len(indices) >= 2:
                mentions[entity] = indices
        output[timeline["video_id"]] = mentions
    return output


async def run_topic_stages(catalog: dict, levels: dict[str, list[dict]], memory: dict, model: LocalModel, concurrency: int, root: Path) -> dict:
    candidates = entity_candidates(catalog, levels, memory, root)
    filter_jobs = []
    for video_id, entities in candidates.items():
        if video_id == "week":
            continue
        ordered = sorted(entities)
        for batch_index, start in enumerate(range(0, len(ordered), 100)):
            batch = ordered[start:start + 100]
            filter_jobs.append({"key": f"{video_id}:{batch_index}", "video_id": video_id, "entities": batch, "prompt": ENTITY_FILTER.format(entity_list="\n".join(f"- {entity} ({len(entities[entity])} mentions)" for entity in batch))})

    async def filter_entities(job, fill_round):
        messages = [{"role": "user", "content": job["prompt"]}]
        result = await model.generate("entity_filter", job["key"], messages, messages, EntityList)
        if result is None:
            append_failure(root, "entity_filter", job["key"], "ten model attempts failed", fill_round)
            return
        allowed = set(job["entities"])
        kept = []
        for entity in result.root:
            normalized = entity.lower()
            if normalized in allowed and normalized not in kept:
                kept.append(normalized)
        write_json(stage_path(root, "entity_filter", job["key"]), {"video_id": job["video_id"], "entities": kept})

    await run_jobs("entity_filter", filter_jobs, concurrency, root, filter_entities)
    kept_by_video = defaultdict(list)
    kept_by_video["week"] = sorted(candidates["week"])
    for job in filter_jobs:
        path = stage_path(root, "entity_filter", job["key"])
        if path.is_file():
            kept_by_video[job["video_id"]].extend(read_json(path)["entities"])

    captions = defaultdict(list)
    for item in levels["merge"]:
        path = stage_path(root, "merge", item["key"])
        if path.is_file():
            captions[item["video_id"]].append(read_json(path))
    timeline_by_id = {timeline["video_id"]: timeline for timeline in catalog["videos"]}
    topic_jobs = []
    for video_id, entities in kept_by_video.items():
        for entity in entities:
            pattern = re.compile(r"\b" + re.escape(entity) + r"s?\b", re.IGNORECASE)
            lines = []
            for caption in captions[video_id]:
                if not pattern.search(caption["text"]):
                    continue
                if video_id == "week":
                    timestamp = f"{caption['week_day'].upper()} {format_hms(caption['local_clock_start_sec'])}"
                else:
                    timestamp = format_hms(caption["start_sec"])
                lines.append(f"[{timestamp}] {caption['text']}")
            if not lines:
                continue
            entity_key = hashlib.md5(entity.encode()).hexdigest()
            for chunk_index, start in enumerate(range(0, len(lines), 50)):
                topic_jobs.append({"key": f"{video_id}:{entity_key}:{chunk_index}", "video_id": video_id, "entity": entity, "lines": lines[start:start + 50]})

    async def extract_topic(job, fill_round):
        prompt = TOPIC[job["video_id"]].format(entity=job["entity"], episodes_text="\n\n".join(job["lines"]))
        messages = [{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": prompt}]
        response_format = WeekTopicFacts if job["video_id"] == "week" else TopicFacts
        if job["video_id"] == "week":
            def validate_topic_times(parsed):
                for fact in parsed.root:
                    day, clock_sec = parse_week_timestamp(fact.time)
                    week_clock_to_timeline(timeline_by_id["week"], day, clock_sec)
            result_validator = validate_topic_times
        else:
            result_validator = None
        result = await model.generate("topic", job["key"], messages, messages, response_format, result_validator)
        if result is None:
            append_failure(root, "topic", job["key"], "ten model attempts failed", fill_round)
            return
        facts = []
        for fact in result.root:
            dumped = fact.model_dump(mode="json")
            if job["video_id"] == "week":
                day, clock_sec = parse_week_timestamp(dumped["time"])
                dumped["local_time"] = dumped["time"]
                dumped["time"] = format_hms(week_clock_to_timeline(timeline_by_id["week"], day, clock_sec))
                dumped["day"] = day.upper()
            facts.append(dumped)
        write_json(stage_path(root, "topic", job["key"]), {"video_id": job["video_id"], "entity": job["entity"], "facts": facts})

    await run_jobs("topic", topic_jobs, concurrency, root, extract_topic)
    return {"candidates": candidates, "filter_jobs": filter_jobs, "topic_jobs": topic_jobs, "kept_by_video": dict(kept_by_video)}


async def run_event_stages(catalog: dict, levels: dict[str, list[dict]], model: LocalModel, concurrency: int, root: Path) -> dict:
    timeline_by_id = {timeline["video_id"]: timeline for timeline in catalog["videos"]}
    activity_jobs = []
    for timeline in catalog["videos"]:
        summaries = [
            item
            for item in levels["aggregate_600"]
            if item["video_id"] == timeline["video_id"]
            and stage_path(root, "aggregate_600", item["key"]).is_file()
        ]
        summaries.sort(key=lambda item: item["start_sec"])
        if timeline["video_id"] == "week":
            assigned = 0
            summary_days = {
                item["key"]: read_json(stage_path(root, "aggregate_600", item["key"]))["week_day"]
                for item in summaries
            }
            source_ids = sorted(timeline["source_offsets"], key=lambda source_video_id: timeline["source_offsets"][source_video_id])
            for source_video_id in source_ids:
                source_summaries = [
                    item
                    for item in summaries
                    if summary_days[item["key"]] == source_video_id
                ]
                if source_summaries:
                    activity_jobs.append({
                        "key": f"week:{source_video_id}",
                        "video_id": "week",
                        "source_video_id": source_video_id,
                        "summaries": source_summaries,
                    })
                    assigned += len(source_summaries)
            if assigned != len(summaries):
                raise RuntimeError(f"week 10min summaries outside source ranges assigned={assigned} total={len(summaries)}")
        else:
            for start in range(0, len(summaries), 5):
                activity_jobs.append({
                    "key": f"{timeline['video_id']}:{start // 5}",
                    "video_id": timeline["video_id"],
                    "summaries": summaries[start:start + 5],
                })

    async def extract_activities(job, fill_round):
        rows = [read_json(stage_path(root, "aggregate_600", item["key"])) for item in job["summaries"]]
        if job["video_id"] == "week":
            segments = "\n\n".join(f"[{format_hms(row['local_clock_start_sec'])}] {row['text']}" for row in rows)
            prompt = ACTIVITY["week"].format(date=job["source_video_id"].upper(), segments=segments)
            messages = [{"role": "user", "content": prompt}]
            response_format = WeekActivities
        else:
            segments = "\n\n".join(f"[{format_hms(row['start_sec'])}] {row['text']}" for row in rows)
            prompt = ACTIVITY[job["video_id"]].format(segments=segments)
            messages = [{"role": "system", "content": "You are a precise video analyzer. Return only the requested structured result."}, {"role": "user", "content": prompt}]
            response_format = Activities
        if job["video_id"] == "week":
            def validate_activity_times(parsed):
                for activity in parsed.root:
                    clock_start, clock_end = parse_time_range(activity.time_range)
                    week_clock_to_timeline(timeline_by_id["week"], job["source_video_id"], clock_start)
                    week_clock_to_timeline(timeline_by_id["week"], job["source_video_id"], clock_end)
            result_validator = validate_activity_times
        else:
            result_validator = None
        result = await model.generate("event_activities", job["key"], messages, messages, response_format, result_validator)
        if result is None:
            append_failure(root, "event_activities", job["key"], "ten model attempts failed", fill_round)
            return
        activities = []
        for activity in result.root:
            dumped = activity.model_dump(mode="json")
            if job["video_id"] == "week":
                clock_start, clock_end = parse_time_range(dumped["time_range"])
                dumped["local_time_range"] = dumped["time_range"]
                timeline_start = week_clock_to_timeline(timeline_by_id["week"], job["source_video_id"], clock_start)
                timeline_end = week_clock_to_timeline(timeline_by_id["week"], job["source_video_id"], clock_end)
                dumped["time_range"] = f"{format_hms(timeline_start)}-{format_hms(timeline_end)}"
                dumped["day"] = job["source_video_id"].upper()
                dumped["location"] = None
            activities.append(dumped)
        write_json(stage_path(root, "event_activities", job["key"]), {"video_id": job["video_id"], "activities": activities})

    await run_jobs("event_activities", activity_jobs, concurrency, root, extract_activities)
    chain_jobs = []
    for timeline in catalog["videos"]:
        activities = []
        for job in activity_jobs:
            path = stage_path(root, "event_activities", job["key"])
            if job["video_id"] == timeline["video_id"] and path.is_file():
                activities.extend(read_json(path)["activities"])
        activities.sort(key=lambda item: parse_time_range(item["time_range"])[0])
        if timeline["video_id"] == "week":
            chain_jobs.append({"key": "week:0", "video_id": "week", "activities": activities, "stage": "event_chains"})
        else:
            for chunk_index, start in enumerate(range(0, len(activities), 50)):
                chain_jobs.append({"key": f"{timeline['video_id']}:{chunk_index}", "video_id": timeline["video_id"], "activities": activities[start:start + 50], "stage": "event_chains"})

    async def extract_chains(job, fill_round):
        activities_text = ""
        current_day = None
        for activity in job["activities"]:
            if job["video_id"] == "week":
                day = activity["day"]
                if day != current_day:
                    activities_text += f"\n=== {day} ===\n"
                    current_day = day
                time_range = activity["local_time_range"]
            else:
                time_range = activity["time_range"]
            activities_text += f"- [{time_range}] {activity['name']}: {activity['summary']}\n"
            if activity["key_entities"]:
                activities_text += f"  Entities: {', '.join(activity['key_entities'])}\n"
        prompt = EVENT_CHAIN[job["video_id"]].format(activities=activities_text)
        if job["video_id"] == "week":
            messages = [{"role": "user", "content": prompt}]
            response_format = WeekEventChains
        else:
            messages = [{"role": "system", "content": "You are a precise video analyzer. Return only the requested structured result."}, {"role": "user", "content": prompt}]
            response_format = EventChains
        if job["video_id"] == "week":
            def validate_chain_times(parsed):
                for chain in parsed.root:
                    for step in chain.steps:
                        week_clock_to_timeline(timeline_by_id["week"], step.day, parse_hms(step.start_time))
                        week_clock_to_timeline(timeline_by_id["week"], step.day, parse_hms(step.end_time))
            result_validator = validate_chain_times
        else:
            result_validator = None
        result = await model.generate("event_chains", job["key"], messages, messages, response_format, result_validator)
        if result is None:
            append_failure(root, "event_chains", job["key"], "ten model attempts failed", fill_round)
            return
        chains = []
        for chain in result.root:
            dumped = chain.model_dump(mode="json")
            if job["video_id"] == "week":
                dumped["type"] = None
                for step in dumped["steps"]:
                    step["local_start_time"] = step["start_time"]
                    step["local_end_time"] = step["end_time"]
                    step["start_time"] = format_hms(week_clock_to_timeline(timeline_by_id["week"], step["day"], parse_hms(step["start_time"])))
                    step["end_time"] = format_hms(week_clock_to_timeline(timeline_by_id["week"], step["day"], parse_hms(step["end_time"])))
            chains.append(dumped)
        write_json(stage_path(root, "event_chains", job["key"]), {"video_id": job["video_id"], "chains": chains})

    await run_jobs("event_chains", chain_jobs, concurrency, root, extract_chains)
    round2_jobs = []
    week_job = next(job for job in chain_jobs if job["video_id"] == "week")
    week_path = stage_path(root, "event_chains", week_job["key"])
    if week_path.is_file() and read_json(week_path)["chains"]:
        round2_jobs.append({"key": "week:0", "video_id": "week", "activities": week_job["activities"], "stage": "event_chains_round2"})

    async def extract_round2(job, fill_round):
        first_round = read_json(stage_path(root, "event_chains", job["key"]))["chains"]
        found_names = "\n".join(f"- {chain['name']}" for chain in first_round)
        daily_text = ""
        current_day = None
        for activity in job["activities"]:
            day = activity["day"]
            if day != current_day:
                daily_text += f"\n=== {day} ===\n"
                current_day = day
            daily_text += f"- [{activity['local_time_range']}] {activity['name']}: {activity['summary']}\n"
            if activity["key_entities"]:
                daily_text += f"  Entities: {', '.join(activity['key_entities'])}\n"
        prompt = WEEK_EVENT_CHAIN_ROUND2.format(found_names=found_names, daily_text=daily_text)
        messages = [{"role": "user", "content": prompt}]
        def validate_round2_times(parsed):
            for chain in parsed.root:
                for step in chain.steps:
                    week_clock_to_timeline(timeline_by_id["week"], step.day, parse_hms(step.start_time))
                    week_clock_to_timeline(timeline_by_id["week"], step.day, parse_hms(step.end_time))
        result = await model.generate("event_chains_round2", job["key"], messages, messages, WeekEventChains, validate_round2_times)
        if result is None:
            append_failure(root, "event_chains_round2", job["key"], "ten model attempts failed", fill_round)
            return
        chains = []
        for chain in result.root:
            dumped = chain.model_dump(mode="json")
            dumped["type"] = None
            for step in dumped["steps"]:
                step["local_start_time"] = step["start_time"]
                step["local_end_time"] = step["end_time"]
                step["start_time"] = format_hms(week_clock_to_timeline(timeline_by_id["week"], step["day"], parse_hms(step["start_time"])))
                step["end_time"] = format_hms(week_clock_to_timeline(timeline_by_id["week"], step["day"], parse_hms(step["end_time"])))
            chains.append(dumped)
        write_json(stage_path(root, "event_chains_round2", job["key"]), {"video_id": "week", "chains": chains})

    await run_jobs("event_chains_round2", round2_jobs, concurrency, root, extract_round2)
    merged_chains = defaultdict(list)
    for timeline in catalog["videos"]:
        seen = set()
        for job in chain_jobs + round2_jobs:
            path = stage_path(root, job["stage"], job["key"])
            if job["video_id"] != timeline["video_id"] or not path.is_file():
                continue
            for chain in read_json(path)["chains"]:
                name = chain["name"].lower()
                if name not in seen:
                    seen.add(name)
                    merged_chains[timeline["video_id"]].append(chain)

    captions = defaultdict(list)
    for item in levels["merge"]:
        path = stage_path(root, "merge", item["key"])
        if path.is_file():
            captions[item["video_id"]].append(read_json(path))
    enrich_jobs = []
    for video_id, chains in merged_chains.items():
        for chain_index, chain in enumerate(chains):
            for step_index, step in enumerate(chain["steps"]):
                start_sec, end_sec = parse_hms(step["start_time"]), parse_hms(step["end_time"])
                matched = [caption for caption in captions[video_id] if start_sec <= caption["start_sec"] <= end_sec]
                enrich_jobs.append({"key": f"{video_id}:{chain_index}:{step_index}", "video_id": video_id, "chain_index": chain_index, "step_index": step_index, "name": chain["name"], "step": step, "captions": matched})

    async def enrich_step(job, fill_round):
        if not job["captions"]:
            write_json(stage_path(root, "event_enrich", job["key"]), {"video_id": job["video_id"], "chain_index": job["chain_index"], "step_index": job["step_index"], "description": job["step"]["description"], "caption_count": 0})
            return
        if job["video_id"] == "week":
            day = job["step"]["day"]
            captions_text = "\n\n".join(f"[{format_hms(caption['local_clock_start_sec'])}] {caption['text']}" for caption in job["captions"])
            start_time = job["step"]["local_start_time"]
            end_time = job["step"]["local_end_time"]
        else:
            day = ""
            captions_text = "\n\n".join(f"[{format_hms(caption['start_sec'])}] {caption['text']}" for caption in job["captions"])
            start_time = job["step"]["start_time"]
            end_time = job["step"]["end_time"]
        prompt = ENRICH[job["video_id"]].format(name=job["name"], day=day, start_time=start_time, end_time=end_time, captions=captions_text)
        messages = [{"role": "user", "content": prompt}]
        result = await model.generate("event_enrich", job["key"], messages, messages, EnrichedDescription)
        if result is None:
            append_failure(root, "event_enrich", job["key"], "ten model attempts failed", fill_round)
            return
        write_json(stage_path(root, "event_enrich", job["key"]), {"video_id": job["video_id"], "chain_index": job["chain_index"], "step_index": job["step_index"], "description": result.description, "caption_count": len(job["captions"])})

    await run_jobs("event_enrich", enrich_jobs, concurrency, root, enrich_step)
    return {"activity_jobs": activity_jobs, "chain_jobs": chain_jobs + round2_jobs, "merged_chains": dict(merged_chains), "enrich_jobs": enrich_jobs}


def assemble_artifacts(catalog: dict, levels: dict[str, list[dict]], memory: dict, topics: dict, events: dict, root: Path) -> None:
    for timeline in catalog["videos"]:
        video_id = timeline["video_id"]
        video_root = root / "mmlifelong" / video_id
        write_json(video_root / "whisper" / f"{video_id}_whisper.json", read_json(root / "raw_asr" / f"{video_id}.json"))
        for source, stage in (("vlm", "vlm"), ("asr", "asr_rewrite")):
            rows = []
            for clip in timeline["clips"]:
                path = stage_path(root, stage, clip["row_id"])
                if path.is_file():
                    rows.append(read_json(path))
            rows.sort(key=lambda row: row["start_sec"])
            write_json(video_root / "captions" / f"{video_id}_{source}_30sec.json", rows)
        for granularity, stage in ((30, "merge"), (180, "aggregate_180"), (600, "aggregate_600"), (3600, "aggregate_3600")):
            rows = []
            for item in levels[stage]:
                path = stage_path(root, stage, item["key"])
                if item["video_id"] == video_id and path.is_file():
                    rows.append(read_json(path))
            rows.sort(key=lambda row: row["start_sec"])
            name = {30: "30sec", 180: "3min", 600: "10min", 3600: "1h"}[granularity]
            write_json(video_root / "merge" / "captions" / f"{video_id}_{name}.json", rows)

        ner_results = {}
        triple_results = {}
        for job in memory["openie_jobs"]:
            path = stage_path(root, "openie", job["key"])
            if not path.is_file():
                continue
            result = read_json(path)
            if result["video_id"] == video_id:
                ner_results[result["chunk_id"]] = result["named_entities"]
                triple_results[result["chunk_id"]] = result["triples"]
        write_json(video_root / "merge" / "episodic_memory" / "openie_results.json", {"ner_results": ner_results, "triple_results": triple_results})

        semantic_results = {}
        for job in memory["semantic_jobs"]:
            path = stage_path(root, "semantic", job["key"])
            if not path.is_file():
                continue
            result = read_json(path)
            if result["video_id"] == video_id:
                semantic_results[result["chunk_id"]] = {"semantic_triples": result["semantic_triples"], "episodic_evidence": result["episodic_evidence"]}
        write_json(video_root / "merge" / "semantic_memory" / "semantic_extraction_results.json", semantic_results)
        consolidated = []
        for job in memory["consolidation_jobs"]:
            path = stage_path(root, "semantic_consolidation", job["key"])
            if job["video_id"] == video_id and path.is_file():
                consolidated.extend(read_json(path)["consolidated_semantic_triples"])
        consolidation_output = {"0": {"consolidated_semantic_triples": consolidated}} if consolidated else {}
        write_json(video_root / "merge" / "semantic_memory" / "semantic_consolidation_results.json", consolidation_output)

        whitelist = sorted(set(topics["kept_by_video"].get(video_id, [])))
        write_json(video_root / "merge" / "entity_whitelist.json", whitelist)
        topic_output = {}
        for entity in whitelist:
            entity_jobs = [job for job in topics["topic_jobs"] if job["video_id"] == video_id and job["entity"] == entity]
            facts = []
            complete = True
            for job in entity_jobs:
                path = stage_path(root, "topic", job["key"])
                if not path.is_file():
                    complete = False
                    break
                facts.extend(read_json(path)["facts"])
            if complete:
                topic_output[entity] = {"entity": entity, "has_lifecycle": bool(facts), "facts": facts}
        write_json(video_root / "merge" / "topic_chains.json", topic_output)

        activities = []
        for job in events["activity_jobs"]:
            path = stage_path(root, "event_activities", job["key"])
            if job["video_id"] == video_id and path.is_file():
                activities.extend(read_json(path)["activities"])
        activities.sort(key=lambda item: parse_time_range(item["time_range"])[0])
        write_json(video_root / "merge" / "storyline_step1_activities.json", activities)
        chains = events["merged_chains"].get(video_id, [])
        write_json(video_root / "merge" / "storyline_step2_chains.json", chains)
        enriched = []
        for chain_index, chain in enumerate(chains):
            copied = json.loads(json.dumps(chain))
            complete = True
            for step_index, step in enumerate(copied["steps"]):
                key = f"{video_id}:{chain_index}:{step_index}"
                path = stage_path(root, "event_enrich", key)
                if not path.is_file():
                    complete = False
                    break
                step["description"] = read_json(path)["description"]
            if complete:
                enriched.append(copied)
        write_json(video_root / "merge" / "storyline_step3_enriched.json", enriched)
        print(f"artifact_assembled video_id={video_id} captions={len(read_json(video_root / 'merge' / 'captions' / f'{video_id}_30sec.json'))} openie={len(triple_results)} semantic={len(semantic_results)} topics={len(topic_output)} events={len(enriched)}")


async def run(args) -> None:
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.artifact_root / "full.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    catalog = load_catalog(args.database, args.dataset_video_root, args.asr_root, args.clip_dir, args.artifact_root)
    add_aligned_transcripts(catalog, args.artifact_root)
    wandb_run = wandb.init(
        project="video-sql-v2",
        name="magic-index-mmlifelong-full",
        config={
            "source": "MAGIC-video@0f2273f7f07360cc58588399553b5a6ea758b084",
            "dataset": "MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443",
            "model": args.model,
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "timeline_definition": "multiple source files sharing one continuous timeline are one logical video; independent timelines remain separate",
            "captions_only": args.captions_only,
            "logical_videos": len(catalog["videos"]),
            "clips": len(catalog["clips"]),
        },
    )
    model = LocalModel(args.base_url, args.model, args.max_retries, args.artifact_root)
    try:
        levels = await run_caption_stages(catalog, model, args.concurrency, args.artifact_root)
        if args.captions_only:
            return
        memory = await run_memory_stages(catalog, levels, model, args.concurrency, args.artifact_root)
        topics = await run_topic_stages(catalog, levels, memory, model, args.concurrency, args.artifact_root)
        events = await run_event_stages(catalog, levels, model, args.concurrency, args.artifact_root)
        assemble_artifacts(catalog, levels, memory, topics, events, args.artifact_root)
    finally:
        await model.client.close()
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-video-root", type=Path, required=True)
    parser.add_argument("--asr-root", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--captions-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_retries != 10:
        raise ValueError("max retries must be 10")
    if arguments.concurrency < 1:
        raise ValueError("concurrency must be positive")
    asyncio.run(run(arguments))
