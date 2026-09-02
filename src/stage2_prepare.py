import argparse
import asyncio
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb
from faster_whisper import WhisperModel
from openai import APIError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, ValidationError, field_validator


WORKERS = 32
WHISPER_MODEL = "large-v3-turbo"
QWEN_MODEL = "Inferact/Qwen3.8-27B-NVFP4"


class SourceVideo(BaseModel):
    source_video_id: str
    video_id: str
    source_group_id: str
    filename: str
    duration_s: float
    timeline_start_s: float
    timeline_end_s: float
    video_codec: str
    audio_codec: str


class SourceGroup(BaseModel):
    source_group_id: str
    video_id: str
    name: str
    timeline_start_s: float
    timeline_end_s: float


class Chunk(BaseModel):
    row_id: str
    visual_row_id: str
    video_id: str
    source_video_id: str
    source_group_id: str
    source_filename: str
    source_start_s: float
    source_end_s: float
    start_s: float
    end_s: float
    clip_filename: str
    official_text: str | None
    official_artifact: str | None


class DenseCaption(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_complete_paragraph(cls, text: str) -> str:
        text = text.strip()
        sentence_endings = re.findall(r"[.!?。！？]", text)
        if len(sentence_endings) < 2 or re.search(r"[.!?。！？]$", text) is None:
            raise ValueError("Dense caption must contain at least two sentence endings and end with sentence punctuation")
        return text


class DenseOutputError(RuntimeError):
    pass


def probe_media(path: Path, require_audio: bool) -> tuple[float, str, str]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name", "-of", "json", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(json.dumps({"command": "ffprobe", "path": str(path), "stderr": result.stderr}))
        result.check_returncode()
    payload = json.loads(result.stdout)
    duration_s = float(payload["format"]["duration"])
    video_codecs = [stream["codec_name"] for stream in payload["streams"] if stream["codec_type"] == "video"]
    audio_codecs = [stream["codec_name"] for stream in payload["streams"] if stream["codec_type"] == "audio"]
    if not video_codecs or (require_audio and not audio_codecs):
        raise RuntimeError(f"Missing video or audio stream in {path}")
    return duration_s, video_codecs[0], audio_codecs[0] if audio_codecs else ""


def load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def annotation_offsets(annotation_paths: list[Path], clue_key: str) -> dict[str, float]:
    offsets: dict[str, list[float]] = {}
    for annotation_path in annotation_paths:
        for question in json.loads(annotation_path.read_text()):
            local_intervals = [(block["video_id"], interval) for block in question[clue_key] for interval in block["intervals"]]
            if len(local_intervals) != len(question["total_intervals"]):
                raise RuntimeError(f"Clue/total interval mismatch in {annotation_path}, question {question['index']}")
            for (video_id, local_interval), total_interval in zip(local_intervals, question["total_intervals"]):
                offsets.setdefault(str(video_id), []).append(float(total_interval[0]) - float(local_interval[0]))
    result = {}
    for video_id, values in offsets.items():
        if max(values) - min(values) > 0.001:
            raise RuntimeError(f"Inconsistent logical offset for {video_id}: min={min(values)}, max={max(values)}")
        result[video_id] = values[0]
    return result


def build_inventory(dataset_root: Path, artifact_root: Path, run_root: Path, clip_root: Path) -> dict[str, int | float]:
    day_paths = [dataset_root / "videos" / "day" / "0.mp4"]
    month_paths = sorted((dataset_root / "videos" / "month").glob("*.mp4"), key=lambda path: int(path.stem))
    week_paths = sorted((dataset_root / "videos" / "week").glob("day*/*.mp4"), key=lambda path: (int(path.parent.name.removeprefix("day")), path.name))
    all_paths = day_paths + month_paths + week_paths
    if len(day_paths) != 1 or len(month_paths) != 23 or len(week_paths) != 6266:
        raise RuntimeError(f"Unexpected source counts: day={len(day_paths)}, month={len(month_paths)}, week={len(week_paths)}")
    for path in all_paths:
        if not path.is_file():
            raise RuntimeError(f"Missing source video {path}")
    media = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(probe_media, path, True): path for path in all_paths}
        for completed, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            media[path] = future.result()
            if completed % 500 == 0 or completed == len(futures):
                print(json.dumps({"stage": "inventory", "completed": completed, "total": len(futures)}))

    month_offsets = annotation_offsets([dataset_root / "month" / "train.json", dataset_root / "month" / "val.json"], "clue_interval")
    week_offsets = annotation_offsets([dataset_root / "week" / "test.json"], "clue_intervals")
    if sorted(month_offsets, key=int) != [str(index) for index in range(1, 24) if index != 2]:
        raise RuntimeError(f"Unexpected Month annotation source IDs: {sorted(month_offsets, key=int)}")
    if sorted(week_offsets) != [f"day{index}" for index in range(1, 8)]:
        raise RuntimeError(f"Unexpected Week annotation source IDs: {sorted(week_offsets)}")
    if abs(media[month_paths[0]][0] + media[month_paths[1]][0] - month_offsets["3"]) > 0.05:
        raise RuntimeError(f"Month logical video 1 physical parts do not end at official video 3 offset: parts={media[month_paths[0]][0] + media[month_paths[1]][0]}, video3_offset={month_offsets['3']}")

    sources = []
    groups = []
    day_duration, day_video_codec, day_audio_codec = media[day_paths[0]]
    sources.append(SourceVideo(source_video_id="day:0", video_id="day", source_group_id="day:0", filename=str(day_paths[0].relative_to(dataset_root)), duration_s=day_duration, timeline_start_s=0.0, timeline_end_s=day_duration, video_codec=day_video_codec, audio_codec=day_audio_codec))
    groups.append(SourceGroup(source_group_id="day:0", video_id="day", name="0", timeline_start_s=0.0, timeline_end_s=day_duration))
    for path in month_paths:
        duration_s, video_codec, audio_codec = media[path]
        source_name = path.stem
        logical_source_name = "1" if source_name == "2" else source_name
        start_s = media[month_paths[0]][0] if source_name == "2" else month_offsets[source_name]
        sources.append(SourceVideo(source_video_id=f"month:{source_name}", video_id="month", source_group_id=f"month:{logical_source_name}", filename=str(path.relative_to(dataset_root)), duration_s=duration_s, timeline_start_s=start_s, timeline_end_s=start_s + duration_s, video_codec=video_codec, audio_codec=audio_codec))
    month_sources_by_group = {}
    for source in sources:
        if source.video_id == "month":
            month_sources_by_group.setdefault(source.source_group_id, []).append(source)
    for source_group_id, group_sources in month_sources_by_group.items():
        groups.append(SourceGroup(source_group_id=source_group_id, video_id="month", name=source_group_id.removeprefix("month:"), timeline_start_s=min(source.timeline_start_s for source in group_sources), timeline_end_s=max(source.timeline_end_s for source in group_sources)))
    week_by_day = {f"day{index}": [] for index in range(1, 8)}
    for path in week_paths:
        week_by_day[path.parent.name].append(path)
    for day_name, paths in week_by_day.items():
        day_start_s = week_offsets[day_name]
        source_offset_s = 0.0
        for path in paths:
            duration_s, video_codec, audio_codec = media[path]
            source_video_id = f"week:{day_name}:{path.stem}"
            sources.append(SourceVideo(source_video_id=source_video_id, video_id="week", source_group_id=f"week:{day_name}", filename=str(path.relative_to(dataset_root)), duration_s=duration_s, timeline_start_s=day_start_s + source_offset_s, timeline_end_s=day_start_s + source_offset_s + duration_s, video_codec=video_codec, audio_codec=audio_codec))
            source_offset_s += duration_s
        groups.append(SourceGroup(source_group_id=f"week:{day_name}", video_id="week", name=day_name, timeline_start_s=day_start_s, timeline_end_s=day_start_s + source_offset_s))

    official_month = {}
    for source_name in sorted((path.name for path in (artifact_root / "mmlifelong").iterdir() if path.is_dir()), key=int):
        caption_path = artifact_root / "mmlifelong" / source_name / "merge" / "captions" / f"{source_name}_30sec.json"
        official_month[source_name] = (caption_path, json.loads(caption_path.read_text()))
    official_week_path = artifact_root / "egolife" / "A1_JAKE" / "captions" / "A1_JAKE_30sec.json"
    official_week = {Path(entry["video_path"]).name: entry for entry in json.loads(official_week_path.read_text())}

    chunks = []
    video_indexes = {"day": 0, "week": 0, "month": 0}
    for source in sources:
        source_path = dataset_root / source.filename
        if source.video_id == "week":
            intervals = [(0.0, source.duration_s)]
        else:
            intervals = [(index * 30.0, min((index + 1) * 30.0, source.duration_s)) for index in range(math.ceil(source.duration_s / 30.0))]
        for source_index, (source_start_s, source_end_s) in enumerate(intervals):
            video_index = video_indexes[source.video_id]
            row_id = f"{source.video_id}:episodes:30sec:{video_index:06d}"
            visual_row_id = f"{source.video_id}:visual_clips:{video_index:06d}"
            official_text = None
            official_artifact = None
            if source.video_id == "month" and source.source_group_id.removeprefix("month:") in official_month:
                caption_path, entries = official_month[source.source_group_id.removeprefix("month:")]
                if len(entries) != len(intervals):
                    raise RuntimeError(f"Official Month caption count mismatch for {source.source_group_id}: captions={len(entries)}, chunks={len(intervals)}")
                entry = entries[source_index]
                if abs(float(entry["start_sec"]) - source_start_s) > 0.001 or abs(float(entry["end_sec"]) - source_end_s) > 0.05:
                    raise RuntimeError(f"Official Month caption boundary mismatch for {source.source_video_id} chunk {source_index}")
                official_text = str(entry["text"]).strip()
                official_artifact = str(caption_path)
            elif source.video_id == "week" and source_path.name in official_week:
                official_text = str(official_week[source_path.name]["text"]).strip()
                official_artifact = str(official_week_path)
            if official_text == "":
                raise RuntimeError(f"Empty official dense caption for {source.source_video_id} chunk {source_index}")
            chunks.append(Chunk(row_id=row_id, visual_row_id=visual_row_id, video_id=source.video_id, source_video_id=source.source_video_id, source_group_id=source.source_group_id, source_filename=source.filename, source_start_s=source_start_s, source_end_s=source_end_s, start_s=source.timeline_start_s + source_start_s, end_s=source.timeline_start_s + source_end_s, clip_filename=str((clip_root.resolve() / source.video_id / f"{video_index:06d}.mp4").relative_to(Path.cwd())), official_text=official_text, official_artifact=official_artifact))
            video_indexes[source.video_id] += 1

    write_jsonl(run_root / "source_videos.jsonl", sources)
    write_jsonl(run_root / "source_groups.jsonl", groups)
    write_jsonl(run_root / "chunks.jsonl", chunks)
    total_duration_s = sum(source.duration_s for source in sources)
    if abs(total_duration_s - 652293.329832) > 0.001:
        raise RuntimeError(f"Full source duration mismatch: {total_duration_s}")
    return {"sources": len(sources), "groups": len(groups), "chunks": len(chunks), "official_dense": sum(chunk.official_text is not None for chunk in chunks), "missing_dense": sum(chunk.official_text is None for chunk in chunks), "duration_s": total_duration_s}


def prepare_clips(dataset_root: Path, run_root: Path) -> dict[str, int]:
    chunks = load_jsonl(run_root / "chunks.jsonl", Chunk)
    def encode_clip(chunk: Chunk) -> None:
        output_path = Path.cwd() / chunk.clip_filename
        if output_path.exists():
            duration_s, _, _ = probe_media(output_path, False)
            if abs(duration_s - (chunk.source_end_s - chunk.source_start_s)) <= 0.2:
                return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = dataset_root / chunk.source_filename
        result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(chunk.source_start_s), "-i", str(source_path), "-t", str(chunk.source_end_s - chunk.source_start_s), "-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast", "-an", "-y", str(output_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(json.dumps({"command": "ffmpeg", "row_id": chunk.row_id, "source": str(source_path), "start_s": chunk.source_start_s, "end_s": chunk.source_end_s, "stderr": result.stderr}))
            result.check_returncode()
        duration_s, _, _ = probe_media(output_path, False)
        if abs(duration_s - (chunk.source_end_s - chunk.source_start_s)) > 0.2:
            raise RuntimeError(f"Encoded clip duration mismatch for {chunk.row_id}: expected={chunk.source_end_s - chunk.source_start_s}, actual={duration_s}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(encode_clip, chunk): chunk.row_id for chunk in chunks}
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 500 == 0 or completed == len(futures):
                print(json.dumps({"stage": "clips", "completed": completed, "total": len(futures)}))
    return {"clips": len(chunks)}


def transcribe_missing_sources(dataset_root: Path, run_root: Path) -> dict[str, int]:
    sources = {source.source_video_id: source for source in load_jsonl(run_root / "source_videos.jsonl", SourceVideo)}
    chunks = load_jsonl(run_root / "chunks.jsonl", Chunk)
    missing_source_ids = sorted({chunk.source_video_id for chunk in chunks if chunk.official_text is None})
    output_dir = run_root / "whisper"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16", num_workers=WORKERS)
    def transcribe(source_video_id: str) -> None:
        output_path = output_dir / f"{source_video_id.replace(':', '__')}.json"
        if output_path.exists():
            payload = json.loads(output_path.read_text())
            if payload["source_video_id"] != source_video_id or payload["model"] != WHISPER_MODEL:
                raise RuntimeError(f"Whisper checkpoint mismatch in {output_path}")
            return
        source = sources[source_video_id]
        source_path = dataset_root / source.filename
        last_error = None
        for attempt in range(1, 6):
            trace_path = run_root / "traces" / "whisper" / source_video_id.replace(":", "__") / f"attempt-{attempt}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            request = {"source_video_id": source_video_id, "source_path": str(source_path), "model": WHISPER_MODEL, "device": "cuda", "compute_type": "float16", "beam_size": 5, "vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 500}}
            try:
                segments, info = model.transcribe(str(source_path), beam_size=5, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
                rows = [{"start_sec": round(segment.start, 3), "end_sec": round(segment.end, 3), "text": segment.text.strip()} for segment in segments]
            except RuntimeError as error:
                last_error = error
                trace_path.write_text(json.dumps({"request": request, "error": str(error)}, ensure_ascii=False, indent=2))
                print(json.dumps({"stage": "whisper", "source_video_id": source_video_id, "attempt": attempt, "error": str(error)}))
                continue
            payload = {"source_video_id": source_video_id, "model": WHISPER_MODEL, "language": info.language, "language_probability": info.language_probability, "segments": rows}
            trace_path.write_text(json.dumps({"request": request, "response": payload}, ensure_ascii=False, indent=2))
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        raise RuntimeError(f"Whisper failed after 5 attempts for {source_video_id}: {last_error}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(transcribe, source_video_id): source_video_id for source_video_id in missing_source_ids}
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            print(json.dumps({"stage": "whisper", "completed": completed, "total": len(futures)}))
    return {"transcribed_sources": len(missing_source_ids)}


async def generate_dense(run_root: Path, base_url: str) -> dict[str, int]:
    chunks = load_jsonl(run_root / "chunks.jsonl", Chunk)
    dense_dir = run_root / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    missing_chunks = []
    retry_attempts = {}
    repair_messages = {}
    for chunk in chunks:
        output_path = dense_dir / f"{chunk.row_id.replace(':', '__')}.json"
        if output_path.exists():
            payload = json.loads(output_path.read_text())
            if payload["row_id"] != chunk.row_id or not str(payload["text"]).strip():
                raise RuntimeError(f"Dense checkpoint mismatch in {output_path}")
            if payload["kind"] == "official":
                continue
            if payload["kind"] != "generated":
                raise RuntimeError(f"Unknown dense checkpoint kind in {output_path}")
            try:
                DenseCaption(text=payload["text"])
            except ValidationError as error:
                trace_dir = run_root / "traces" / "dense" / chunk.row_id.replace(":", "__")
                attempt_numbers = sorted(int(path.stem.removeprefix("attempt-")) for path in trace_dir.glob("attempt-*.json"))
                if not attempt_numbers or attempt_numbers != list(range(1, max(attempt_numbers) + 1)):
                    raise RuntimeError(f"Dense trace attempts are incomplete for {chunk.row_id}")
                retry_attempts[chunk.row_id] = max(attempt_numbers) + 1
                repair_messages[chunk.row_id] = f"Your structured response was rejected by validation: {error}. Return the complete corrected DenseCaption object."
                if retry_attempts[chunk.row_id] > 5:
                    raise RuntimeError(f"Dense caption failed after 5 attempts for {chunk.row_id}: {error}")
            else:
                continue
        if chunk.official_text is not None:
            output_path.write_text(json.dumps({"row_id": chunk.row_id, "kind": "official", "artifact": chunk.official_artifact, "text": chunk.official_text}, ensure_ascii=False, indent=2))
        else:
            missing_chunks.append(chunk)

    whisper_cache = {}
    for chunk in missing_chunks:
        if chunk.source_video_id not in whisper_cache:
            whisper_path = run_root / "whisper" / f"{chunk.source_video_id.replace(':', '__')}.json"
            whisper_payload = json.loads(whisper_path.read_text())
            if whisper_payload["source_video_id"] != chunk.source_video_id or whisper_payload["model"] != WHISPER_MODEL:
                raise RuntimeError(f"Whisper checkpoint mismatch in {whisper_path}")
            whisper_cache[chunk.source_video_id] = whisper_payload["segments"]
    client = AsyncOpenAI(base_url=base_url.rstrip("/") + "/v1", max_retries=0, default_headers={"Authorization": ""}, _enforce_credentials=False)
    semaphore = asyncio.Semaphore(WORKERS)
    system_prompt = "You are an expert video captioner. Describe the supplied video clip in one factual 2-4 sentence paragraph. Treat the visual stream as primary evidence and integrate the aligned Whisper transcript naturally. Mention concrete actions, people, objects, locations, visible text, interface state, and spoken content when present. Do not claim anything not supported by the video or transcript."
    async def generate_one(chunk: Chunk) -> None:
        output_path = dense_dir / f"{chunk.row_id.replace(':', '__')}.json"
        transcript_parts = []
        for segment in whisper_cache[chunk.source_video_id]:
            midpoint_s = (float(segment["start_sec"]) + float(segment["end_sec"])) / 2.0
            if chunk.source_start_s <= midpoint_s < chunk.source_end_s and str(segment["text"]).strip():
                transcript_parts.append(str(segment["text"]).strip())
        transcript = " ".join(transcript_parts)
        clip_path = (Path.cwd() / chunk.clip_filename).resolve()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "video_url", "video_url": {"url": clip_path.as_uri()}}, {"type": "text", "text": f"Whisper transcript for this exact clip:\n{transcript}"}]}]
        last_error = None
        repair_message = repair_messages[chunk.row_id] if chunk.row_id in repair_messages else None
        first_attempt = retry_attempts[chunk.row_id] if chunk.row_id in retry_attempts else 1
        for attempt in range(first_attempt, 6):
            attempt_messages = messages if repair_message is None else messages + [{"role": "user", "content": repair_message}]
            trace_path = run_root / "traces" / "dense" / chunk.row_id.replace(":", "__") / f"attempt-{attempt}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            request = {"model": QWEN_MODEL, "reasoning_effort": "low", "messages": attempt_messages, "response_format": DenseCaption.model_json_schema()}
            try:
                async with semaphore:
                    response = await client.beta.chat.completions.parse(model=QWEN_MODEL, messages=attempt_messages, response_format=DenseCaption, reasoning_effort="low")
                parsed = response.choices[0].message.parsed
                if parsed is None or not parsed.text.strip():
                    raise DenseOutputError(f"Missing parsed dense caption: {response.choices[0].message.model_dump(mode='json')}")
            except (APIError, LengthFinishReasonError, ValidationError, DenseOutputError) as error:
                last_error = error
                if isinstance(error, (ValidationError, DenseOutputError)):
                    repair_message = f"Your structured response was rejected by validation: {error}. Return the complete corrected DenseCaption object."
                trace_path.write_text(json.dumps({"request": request, "error": str(error)}, ensure_ascii=False, indent=2))
                print(json.dumps({"stage": "dense", "row_id": chunk.row_id, "attempt": attempt, "error": str(error)}))
                continue
            response_payload = response.model_dump(mode="json")
            trace_path.write_text(json.dumps({"request": request, "response": response_payload}, ensure_ascii=False, indent=2))
            output_path.write_text(json.dumps({"row_id": chunk.row_id, "kind": "generated", "source_video_id": chunk.source_video_id, "source_start_s": chunk.source_start_s, "source_end_s": chunk.source_end_s, "clip_filename": chunk.clip_filename, "whisper_text": transcript, "text": parsed.text.strip()}, ensure_ascii=False, indent=2))
            return
        raise RuntimeError(f"Dense caption failed after 5 attempts for {chunk.row_id}: {last_error}")
    tasks = [asyncio.create_task(generate_one(chunk)) for chunk in missing_chunks]
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        await task
        if completed % 100 == 0 or completed == len(tasks):
            print(json.dumps({"stage": "dense", "completed": completed, "total": len(tasks)}))
    await client.close()
    return {"official_dense": sum(chunk.official_text is not None for chunk in chunks), "generated_dense": len(missing_chunks), "dense_total": len(chunks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inventory", "clips", "whisper", "dense"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.clip_root.resolve() != (Path.cwd() / "data" / "pilot" / "clips" / "stage2").resolve():
        raise RuntimeError("Stage 2 clip root must be data/pilot/clips/stage2")
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name, job_type=f"stage2-{args.stage}", config={"stage": args.stage, "workers": WORKERS, "whisper_model": WHISPER_MODEL, "qwen_model": QWEN_MODEL, "dataset_root": str(args.dataset_root), "artifact_root": str(args.artifact_root), "run_root": str(args.run_root), "clip_root": str(args.clip_root)})
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    if args.stage == "inventory":
        metrics = build_inventory(args.dataset_root, args.artifact_root, args.run_root, args.clip_root)
    elif args.stage == "clips":
        metrics = prepare_clips(args.dataset_root, args.run_root)
    elif args.stage == "whisper":
        metrics = transcribe_missing_sources(args.dataset_root, args.run_root)
    else:
        if args.base_url is None:
            raise RuntimeError("--base-url is required for dense")
        metrics = asyncio.run(generate_dense(args.run_root, args.base_url))
    run.log(metrics)
    run.finish()
    print(json.dumps({"stage": args.stage, "status": "completed", **metrics}))
