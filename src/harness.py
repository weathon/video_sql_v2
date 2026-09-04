from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import yaml
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner, function_tool
from agents.tool import ToolOutputImage, ToolOutputText
from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel


MAX_AGENT_TURNS = 100
SQL_TIMEOUT_S = 30
SQL_ROW_LIMIT = 100
SQL_CELL_LIMIT = 500
SQL_QA_CHAR_LIMIT = 50_000
CONTEXT_LIMIT = 262_144
COMPACTION_THRESHOLD = 50_000

SEARCH_TABLES: dict[str, tuple[str, str, str]] = {
    "episodes": ("text", "start_s", "end_s"),
    "entities": ("name", "first_seen_s", "last_seen_s"),
    "semantic_triples": ("text", "snapshot_s", "snapshot_s"),
    "topic_facts": ("fact", "time_s", "time_s"),
    "event_steps": ("description", "start_s", "end_s"),
    "object_states": ("text", "start_s", "end_s"),
    "visual_clips": ("text", "start_s", "end_s"),
}


class Question(BaseModel):
    timeline: str = "month"
    source_split: str = ""
    index: int | str
    question: str
    answer: str


class JudgeScore(BaseModel):
    reasoning: str
    score: int


class AgentAnswer(BaseModel):
    status: str = "answered"
    answer: str
    evidence_times: list[str] = []


class AgentReflection(BaseModel):
    reflection: str
    meta_reflection: str


class CompactionLimit(RuntimeError):
    def __init__(self, handoff: str, agent: Agent[Any], result: Any):
        super().__init__("main agent reached the third compaction")
        self.handoff = handoff
        self.agent = agent
        self.result = result


class AgentRaised(RuntimeError):
    pass


@dataclass
class HarnessConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    stage2_root: Path = Path("data/pilot/stage2")
    embedding_base_url: str = "http://127.0.0.1:8013/v1"
    embedding_model: str = "qwen3-vl-embedding-8b"
    main_model: str = "gpt-5.6-luna"
    main_base_url: str | None = None
    reasoning_effort: str = "medium"
    openai_api_key_env: str = "OPENAI_API_KEY"
    attempts: int = 3


@dataclass
class HarnessContext:
    config: HarnessConfig
    index: Stage2Index
    run_dir: Path
    skills_dir: Path
    notes: dict[str, list[str]] = field(default_factory=dict)


def load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return sanitize_trace(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return jsonable(value.__dict__)
    return value


def sanitize_trace(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_trace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_trace(v) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image") or value.startswith("/9j/") or value.startswith("iVBOR"):
            return f"<image-bytes:{len(value)}>"
        if len(value) > 200_000:
            return value[:200_000] + f"<truncated:{len(value)}>"
    return value


def save_response_trace(path: Path, response: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_trace(jsonable(response)), ensure_ascii=False, indent=2))


def response_message_text(model_response: Any) -> str:
    output = getattr(model_response, "output", None)
    if output is None:
        response = getattr(model_response, "response", None)
        output = getattr(response, "output", []) if response is not None else []
    parts: list[str] = []
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            continue
        if item_type == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        elif hasattr(item, "model_dump"):
            parts.append(json.dumps(item.model_dump(mode="json", exclude_none=True), sort_keys=True))
        else:
            parts.append(str(item))
    return "".join(parts)


def has_three_repeated_messages(raw_responses: list[Any]) -> bool:
    last = None
    count = 0
    for response in raw_responses:
        text = response_message_text(response)
        if not text:
            last = None
            count = 0
            continue
        if text == last:
            count += 1
        else:
            last = text
            count = 1
        if count >= 3:
            return True
    return False


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    parts.append(value)
    return "".join(parts)


class Stage2Index:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.stage2_root = (config.repo_root / config.stage2_root).resolve()
        self.index_path = self.stage2_root / "index.duckdb"
        self.embedding_dir = self.stage2_root / "embedding_index"
        self.row_ids: dict[str, np.ndarray] = {}
        self.vectors: dict[str, np.ndarray] = {}
        for table in SEARCH_TABLES:
            row_path = self.embedding_dir / f"{table}.row_ids.npy"
            vector_path = self.embedding_dir / f"{table}.vectors.npy"
            if row_path.exists() and vector_path.exists():
                self.row_ids[table] = np.load(row_path, mmap_mode="r", allow_pickle=False)
                self.vectors[table] = np.load(vector_path, mmap_mode="r", allow_pickle=False)
                if self.vectors[table].shape != (len(self.row_ids[table]), 4096):
                    raise RuntimeError(f"bad embedding shape for {table}: {self.vectors[table].shape}")

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.index_path), read_only=True)

    def schema_text(self) -> str:
        with self.connect() as con:
            tables = [row[0] for row in con.execute("show tables").fetchall()]
            chunks = []
            for table in tables:
                rows = con.execute(f"describe {table}").fetchall()
                cols = ", ".join(f"{name} {typ}" for name, typ, *_ in rows)
                chunks.append(f"{table}: {cols}")
            return "\n".join(chunks)

    def run_sql(self, sql: str, max_rows: int = SQL_ROW_LIMIT) -> dict[str, Any]:
        stripped = sql.strip()
        lowered = stripped.lower()
        if not lowered.startswith(("select", "with", "show", "describe")):
            return {"error": "only read-only select/with/show/describe sql is allowed"}
        for token in ("insert", "update", "delete", "create", "drop", "alter", "copy", "attach", "detach", "install", "load"):
            if re.search(rf"\b{token}\b", lowered):
                return {"error": f"blocked non-read-only token: {token}"}
        con = self.connect()
        def query() -> tuple[list[str], list[tuple[Any, ...]]]:
            result = con.execute(stripped)
            names = [desc[0] for desc in result.description]
            raw_rows = result.fetchmany(max_rows + 1)
            return names, raw_rows
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(query)
            try:
                names, raw_rows = future.result(timeout=SQL_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                con.interrupt()
                try:
                    future.result()
                except duckdb.InterruptException:
                    pass
                return {"error": f"SQL exceeded {SQL_TIMEOUT_S} seconds"}
            finally:
                con.close()
        rows = []
        for raw_row in raw_rows[:max_rows]:
            row = {}
            for name, cell in zip(names, raw_row, strict=True):
                text = str(cell)
                row[name] = {"value": text[:SQL_CELL_LIMIT], "length": len(text)}
            rows.append(row)
        return {"columns": names, "rows": rows, "more": len(raw_rows) > max_rows}

    def sql_text_for_qa(self, sql: str) -> str:
        stripped = sql.strip()
        lowered = stripped.lower()
        if not lowered.startswith(("select", "with", "show", "describe")):
            return json.dumps({"error": "only read-only select/with/show/describe sql is allowed"}, ensure_ascii=False)
        for token in ("insert", "update", "delete", "create", "drop", "alter", "copy", "attach", "detach", "install", "load"):
            if re.search(rf"\b{token}\b", lowered):
                return json.dumps({"error": f"blocked non-read-only token: {token}"}, ensure_ascii=False)
        con = self.connect()
        def query() -> tuple[list[str], list[str], bool]:
            result = con.execute(stripped)
            names = [desc[0] for desc in result.description]
            chunks = [json.dumps(names, ensure_ascii=False)]
            total = len(chunks[0])
            truncated = False
            while True:
                raw_row = result.fetchone()
                if raw_row is None:
                    break
                row = {}
                for name, cell in zip(names, raw_row, strict=True):
                    row[name] = str(cell)
                line = json.dumps(row, ensure_ascii=False)
                total += len(line) + 1
                if total > SQL_QA_CHAR_LIMIT:
                    truncated = True
                    break
                chunks.append(line)
            return names, chunks, truncated
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(query)
            try:
                _names, chunks, truncated = future.result(timeout=SQL_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                con.interrupt()
                try:
                    future.result()
                except duckdb.InterruptException:
                    pass
                return json.dumps({"error": f"SQL exceeded {SQL_TIMEOUT_S} seconds"}, ensure_ascii=False)
            finally:
                con.close()
        if truncated:
            chunks.append("<truncated>")
        return "\n".join(chunks)

    async def embed_text(self, query: str) -> np.ndarray:
        payload = {"model": self.config.embedding_model, "input": [query]}
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.config.embedding_base_url}/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()["data"]
        vector = np.asarray(data[0]["embedding"], dtype=np.float32)
        if vector.shape != (4096,) or not np.isfinite(vector).all():
            raise RuntimeError(f"bad query embedding shape/content: {vector.shape}")
        return vector

    async def semantic_search(self, query: str, table: str, n: int, rank_start: int, rank_end: int | None, min_score: float | None, max_score: float | None, video_id: str | None) -> dict[str, Any]:
        if table not in self.vectors:
            return {"error": f"table has no loaded embedding index: {table}"}
        query_vector = await self.embed_text(query)
        scores = self.vectors[table] @ query_vector
        positions = np.argsort(-scores)
        if video_id:
            with self.connect() as con:
                allowed_rows = con.execute(f"select row_id from {table} where video_id = ?", [video_id]).fetchall()
            allowed = {row[0] for row in allowed_rows}
            positions = np.asarray([pos for pos in positions if str(self.row_ids[table][pos]) in allowed], dtype=np.int64)
        end = rank_end if rank_end is not None else rank_start + n
        hits = []
        text_col, start_col, end_col = SEARCH_TABLES[table]
        with self.connect() as con:
            for pos in positions[rank_start:end]:
                score = float(scores[pos])
                if min_score is not None and score < min_score:
                    continue
                if max_score is not None and score > max_score:
                    continue
                row_id = str(self.row_ids[table][pos])
                row = con.execute(f"select row_id, video_id, {start_col}, {end_col}, {text_col} from {table} where row_id = ?", [row_id]).fetchone()
                if row is None:
                    raise RuntimeError(f"embedding row_id missing from duckdb: {table} {row_id}")
                hits.append({"table": table, "row_id": row[0], "video_id": row[1], "start_s": row[2], "end_s": row[3], "text": str(row[4])[:500], "score": score})
        return {"hits": hits, "rank_start": rank_start, "rank_end": end}

    def visual_rows(self, video_id: str, start_s: float, end_s: float) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "select row_id, video_id, start_s, end_s, source_start_s, source_end_s, filename, text from visual_clips where video_id = ? and end_s > ? and start_s < ? order by start_s",
                [video_id, start_s, end_s],
            ).fetchall()
        return [{"row_id": r[0], "video_id": r[1], "start_s": r[2], "end_s": r[3], "source_start_s": r[4], "source_end_s": r[5], "filename": r[6], "text": r[7]} for r in rows]


def openai_client(config: HarnessConfig) -> AsyncOpenAI:
    load_dotenv(config.repo_root)
    base_url = config.main_base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get(config.openai_api_key_env)
    if api_key is None and base_url and "127.0.0.1" in base_url:
        api_key = "local-vllm"
    if api_key is None:
        for key in ("LAB_OPENAI_API_KEY", "OPENAI_LAB_API_KEY", "PERSONAL_OPENAI_API_KEY", "OPENAI_PERSONAL_API_KEY"):
            if os.environ.get(key):
                api_key = os.environ[key]
                break
    if api_key is None:
        api_key = os.environ[config.openai_api_key_env]
    return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=3)


def responses_model(config: HarnessConfig) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(config.main_model, openai_client(config))


def run_config(config: HarnessConfig) -> RunConfig:
    return RunConfig(reasoning_item_id_policy="preserve", model_settings=ModelSettings(reasoning={"effort": config.reasoning_effort}, preserve_raw_usage=True), workflow_name="video-sql-v2")


def skill_inventory(skills_dir: Path, include_meta: bool = False) -> list[dict[str, str]]:
    skills = []
    for path in sorted(skills_dir.glob("*.md")):
        if path.name == "META-SKILL.md" and not include_meta:
            continue
        text = path.read_text()
        meta = {}
        if text.startswith("---"):
            _, frontmatter, _body = text.split("---", 2)
            meta = yaml.safe_load(frontmatter) or {}
        name = str(meta.get("name", path.stem))
        description = str(meta.get("description", ""))
        skills.append({"name": name, "description": description, "file": path.name})
    return skills


def read_skill_file(skills_dir: Path, name: str) -> str:
    for item in skill_inventory(skills_dir, include_meta=True):
        if item["name"] == name or item["file"] == name or Path(item["file"]).stem == name:
            return (skills_dir / item["file"]).read_text()
    raise FileNotFoundError(f"unknown skill: {name}")


def main_instructions(context: HarnessContext) -> str:
    skills = "\n".join(f"- {item['name']}: {item['description']}" for item in skill_inventory(context.skills_dir))
    return "\n".join([
        "You answer questions about long egocentric video memories by querying the Stage 2 DuckDB database, semantic embedding index, and visual clips.",
        "Use tools as needed. Submit the final answer with submit_answer.",
        "Do not expose host absolute paths. Use evidence from SQL rows, semantic hits, or visual checks.",
        "Available skills:",
        skills,
        "Stage 2 schema:",
        context.index.schema_text(),
    ])


def clip_instructions() -> str:
    return "Use sample_video, notes, and submit_answer to answer the clip question. Return unknown if the clip does not show the answer."


def reflection_instructions(context: HarnessContext) -> str:
    meta_skill = read_skill_file(context.skills_dir, "META-SKILL")
    return main_instructions(context) + "\n\nReflection guideline from META-SKILL.md:\n" + meta_skill + "\nSubmit with submit_reflection."


def make_tools(context: HarnessContext, include_submit: str) -> list[Any]:
    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    async def run_sql(sql: str) -> str:
        """Execute a read-only DuckDB SQL query and return at most 100 rows."""
        result = await asyncio.to_thread(context.index.run_sql, sql)
        return json.dumps(result, ensure_ascii=False)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    async def sql_qa(sql: str, question: str) -> str:
        """Execute SQL, then ask a single LLM call to answer the question from the full bounded result text."""
        table_text = await asyncio.to_thread(context.index.sql_text_for_qa, sql)
        response = await openai_client(context.config).responses.create(
            model=context.config.main_model,
            reasoning={"effort": context.config.reasoning_effort},
            input="Answer the question using only this SQL result.\nQuestion:\n" + question + "\nSQL result:\n" + table_text,
        )
        save_response_trace(context.run_dir / "traces" / f"sql-qa-{uuid.uuid4().hex}.json", response)
        return extract_output_text(response)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    async def semantic_search(query: str, n: int, table: str, rank_start: int = 0, rank_end: int | None = None, min_score: float | None = None, max_score: float | None = None, video_id: str | None = None) -> str:
        """Search one embedding-indexed table by Qwen3-VL embedding similarity."""
        result = await context.index.semantic_search(query, table, n, rank_start, rank_end, min_score, max_score, video_id)
        return json.dumps(result, ensure_ascii=False)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    async def sample_video(video_id: str, start_s: float, end_s: float, fps: float, include_captions: bool, crop_x0: int = 0, crop_y0: int = 0, crop_x1: int = 1000, crop_y1: int = 1000) -> list[ToolOutputText | ToolOutputImage]:
        """Return 4x3 contact-sheet image filenames covering the requested interval."""
        if end_s <= start_s:
            raise RuntimeError("end_s must be greater than start_s")
        rows = context.index.visual_rows(video_id, start_s, end_s)
        if not rows:
            raise RuntimeError("no visual clips overlap requested interval")
        media_dir = context.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        duration = end_s - start_s
        frame_count = max(1, math.ceil(duration * fps))
        for i in range(frame_count):
            t = start_s + (i + 0.5) / fps
            matches = [row for row in rows if row["start_s"] <= t < row["end_s"]]
            if not matches:
                raise RuntimeError(f"no visual clip covers sampled timestamp {t}")
            clip = matches[0]
            source_path = context.config.repo_root / clip["filename"]
            local_t = max(0.0, t - float(clip["start_s"]))
            frame_path = media_dir / f"frame-{uuid.uuid4().hex}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{local_t:.3f}", "-i", str(source_path), "-frames:v", "1", "-q:v", "2", str(frame_path)], check=True)
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                left = int(width * crop_x0 / 1000)
                top = int(height * crop_y0 / 1000)
                right = int(width * crop_x1 / 1000)
                bottom = int(height * crop_y1 / 1000)
                image = image.crop((left, top, right, bottom))
                image.save(frame_path, quality=90)
            frame_paths.append(frame_path)
        pages = []
        for page_start in range(0, len(frame_paths), 12):
            page_frames = frame_paths[page_start:page_start + 12]
            images = [Image.open(path).convert("RGB") for path in page_frames]
            thumb_w = max(image.width for image in images)
            thumb_h = max(image.height for image in images)
            sheet = Image.new("RGB", (thumb_w * 4, thumb_h * 3), "white")
            draw = ImageDraw.Draw(sheet)
            for offset, image in enumerate(images):
                image.thumbnail((thumb_w, thumb_h))
                x = offset % 4 * thumb_w
                y = offset // 4 * thumb_h
                sheet.paste(image, (x, y))
                draw.text((x + 8, y + 8), f"{start_s + (page_start + offset + 0.5) / fps:.1f}s", fill=(255, 255, 0))
                image.close()
            sheet_path = media_dir / f"contact-{uuid.uuid4().hex}.jpg"
            sheet.save(sheet_path, quality=90)
            pages.append(sheet_path.name)
            for image in images:
                image.close()
        captions = [{"start_s": row["start_s"], "end_s": row["end_s"], "text": row["text"]} for row in rows] if include_captions else []
        outputs: list[ToolOutputText | ToolOutputImage] = [ToolOutputText(text=json.dumps({"contact_sheets": pages, "captions": captions}, ensure_ascii=False))]
        for page in pages:
            page_path = media_dir / page
            encoded = base64.b64encode(page_path.read_bytes()).decode("ascii")
            outputs.append(ToolOutputImage(image_url="data:image/jpeg;base64," + encoded, detail="high"))
        return outputs

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    async def clip_qa(video_id: str, start_s: float, end_s: float, question: str) -> list[ToolOutputText | ToolOutputImage]:
        """Run an inner visual agent on a clip no longer than 20 minutes."""
        if end_s - start_s > 1200:
            return [ToolOutputText(text=json.dumps({"status": "unknown", "answer": "clip longer than 20 minutes", "evidence_times": []}, ensure_ascii=False))]
        answer = await run_clip_qa(context, video_id, start_s, end_s, question)
        media_dir = context.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        screenshots = []
        outputs: list[ToolOutputText | ToolOutputImage] = []
        for item in answer.evidence_times:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", str(item))
            if match is None:
                raise RuntimeError(f"cannot parse evidence timestamp: {item}")
            timestamp_s = float(match.group(0))
            rows = context.index.visual_rows(video_id, timestamp_s, timestamp_s + 0.001)
            if not rows:
                raise RuntimeError(f"no visual clip for evidence timestamp {timestamp_s}")
            clip = rows[0]
            source_path = context.config.repo_root / clip["filename"]
            local_t = max(0.0, timestamp_s - float(clip["start_s"]))
            screenshot_path = media_dir / f"evidence-{uuid.uuid4().hex}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{local_t:.3f}", "-i", str(source_path), "-frames:v", "1", "-q:v", "2", str(screenshot_path)], check=True)
            screenshots.append(screenshot_path.name)
            encoded = base64.b64encode(screenshot_path.read_bytes()).decode("ascii")
            outputs.append(ToolOutputImage(image_url="data:image/jpeg;base64," + encoded, detail="high"))
        payload = answer.model_dump()
        payload["evidence_screenshots"] = screenshots
        outputs.insert(0, ToolOutputText(text=json.dumps(payload, ensure_ascii=False)))
        return outputs

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    def add_note(key: str, note: str) -> str:
        """Save a note for this rollout."""
        context.notes.setdefault(key, []).append(note)
        return json.dumps({"ok": True}, ensure_ascii=False)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    def read_notes(key: str) -> str:
        """Read notes saved in this rollout."""
        return json.dumps({"notes": context.notes.get(key, [])}, ensure_ascii=False)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    def read_skill(name: str) -> str:
        """Read one skill Markdown file by skill name."""
        return read_skill_file(context.skills_dir, name)

    @function_tool(failure_error_function=lambda _ctx, exc: json.dumps({"error": str(exc)}, ensure_ascii=False))
    def agent_raise(reason: str) -> str:
        """End this rollout as an agent-raised failure."""
        return json.dumps({"status": "raised", "reason": reason}, ensure_ascii=False)

    tools = [run_sql, sql_qa, semantic_search, clip_qa, add_note, read_notes, read_skill, agent_raise]
    if include_submit == "answer":
        @function_tool
        def submit_answer(status: str, answer: str, evidence_times: list[str]) -> str:
            """Submit the final answer and stop the agent loop."""
            return json.dumps({"status": status, "answer": answer, "evidence_times": evidence_times}, ensure_ascii=False)
        tools.append(submit_answer)
    elif include_submit == "reflection":
        @function_tool
        def submit_reflection(reflection: str, meta_reflection: str) -> str:
            """Submit reflection and meta-reflection and stop the reflection loop."""
            return json.dumps({"reflection": reflection, "meta_reflection": meta_reflection}, ensure_ascii=False)
        tools.append(submit_reflection)
    return tools


def make_clip_tools(context: HarnessContext) -> list[Any]:
    tools = [tool for tool in make_tools(context, "answer") if tool.name in {"sample_video", "add_note", "read_notes", "submit_answer", "agent_raise"}]
    return tools


async def run_agent_stream(agent: Agent[HarnessContext], input_payload: str | list[dict[str, Any]], context: HarnessContext, trace_path: Path, compact_main: bool = False, force_submit_tool: str | None = None, allow_force: bool = True):
    result = Runner.run_streamed(agent, input_payload, context=context, max_turns=MAX_AGENT_TURNS, run_config=run_config(context.config))
    events = []
    compacted = False
    forced = False
    try:
        async for event in result.stream_events():
            events.append(jsonable(event))
            if compact_main and getattr(event, "type", None) == "raw_response_event":
                response = getattr(getattr(event, "data", None), "response", None)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
                output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                if input_tokens is not None and output_tokens is not None:
                    remaining = CONTEXT_LIMIT - input_tokens - output_tokens
                    if remaining <= COMPACTION_THRESHOLD and not compacted:
                        result.cancel(mode="after_turn")
                        compacted = True
            if force_submit_tool is not None and not compact_main and getattr(event, "type", None) == "raw_response_event":
                response = getattr(getattr(event, "data", None), "response", None)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
                output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                if input_tokens is not None and output_tokens is not None:
                    remaining = CONTEXT_LIMIT - input_tokens - output_tokens
                    if remaining <= COMPACTION_THRESHOLD and not forced:
                        result.cancel(mode="after_turn")
                        forced = True
    finally:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({"events": sanitize_trace(events), "raw_responses": jsonable(result.raw_responses), "new_items": jsonable(result.new_items), "final_output": jsonable(result.final_output)}, ensure_ascii=False, indent=2))
    if has_three_repeated_messages(result.raw_responses):
        raise RuntimeError("three repeated non-reasoning model messages")
    if forced and allow_force:
        saved_tools = agent.tools
        saved_behavior = agent.tool_use_behavior
        agent.tools = [tool for tool in saved_tools if getattr(tool, "name", "") in {force_submit_tool, "agent_raise"}]
        agent.tool_use_behavior = {"stop_at_tool_names": [force_submit_tool, "agent_raise"]}
        forced_result, _forced_compacted = await run_agent_stream(
            agent,
            result.to_input_list(mode="preserve_all") + [{"role": "user", "content": "Context is nearly full. Do not call exploratory tools. Submit the final output now."}],
            context,
            trace_path.with_name(trace_path.stem + "-forced" + trace_path.suffix),
            force_submit_tool=force_submit_tool,
            allow_force=False,
        )
        agent.tools = saved_tools
        agent.tool_use_behavior = saved_behavior
        return forced_result, False
    return result, compacted


async def run_clip_qa(context: HarnessContext, video_id: str, start_s: float, end_s: float, question: str) -> AgentAnswer:
    agent = Agent(
        name="clip_qa",
        instructions=clip_instructions(),
        model=responses_model(context.config),
        model_settings=ModelSettings(reasoning={"effort": context.config.reasoning_effort}, preserve_raw_usage=True),
        tools=make_clip_tools(context),
        tool_use_behavior={"stop_at_tool_names": ["submit_answer", "agent_raise"]},
    )
    prompt = f"video_id={video_id}\nstart_s={start_s}\nend_s={end_s}\nquestion={question}"
    trace_path = context.run_dir / "traces" / f"clip-{uuid.uuid4().hex}.json"
    result, _compacted = await run_agent_stream(agent, prompt, context, trace_path, force_submit_tool="submit_answer")
    if '"status": "raised"' in str(result.final_output):
        raise AgentRaised(str(result.final_output))
    answer = AgentAnswer.model_validate_json(result.final_output)
    if answer.status == "answered" and not answer.evidence_times:
        raise RuntimeError("clip_qa answered without evidence_times")
    return answer


async def answer_question(context: HarnessContext, question: Question, trace_name: str) -> tuple[AgentAnswer, Any, Agent[HarnessContext]]:
    agent = Agent(
        name="retrieval_agent",
        instructions=main_instructions(context),
        model=responses_model(context.config),
        model_settings=ModelSettings(reasoning={"effort": context.config.reasoning_effort}, preserve_raw_usage=True),
        tools=make_tools(context, "answer"),
        tool_use_behavior={"stop_at_tool_names": ["submit_answer", "agent_raise"]},
    )
    prompt = f"timeline: {question.timeline}\nquestion: {question.question}"
    current_input: str | list[dict[str, Any]] = prompt
    for compaction_index in range(3):
        agent.tools = make_tools(context, "answer")
        agent.tool_use_behavior = {"stop_at_tool_names": ["submit_answer", "agent_raise"]}
        result, compacted = await run_agent_stream(agent, current_input, context, context.run_dir / "traces" / f"{trace_name}-answer-segment-{compaction_index:02d}.json", compact_main=True)
        if not compacted:
            if '"status": "raised"' in str(result.final_output):
                raise AgentRaised(str(result.final_output))
            return AgentAnswer.model_validate_json(result.final_output), result, agent
        history = result.to_input_list(mode="preserve_all")
        saved_tools = agent.tools
        saved_behavior = agent.tool_use_behavior
        agent.tools = []
        agent.tool_use_behavior = "run_llm_again"
        compact_result, _ = await run_agent_stream(agent, history + [{"role": "user", "content": "Stop using tools. Summarize the current question, evidence found, failed paths, unresolved needs, and next best retrieval step as a trusted handoff for the same agent."}], context, context.run_dir / "traces" / f"{trace_name}-compaction-{compaction_index:02d}.json")
        agent.tools = saved_tools
        agent.tool_use_behavior = saved_behavior
        handoff = str(compact_result.final_output)
        if compaction_index == 2:
            raise CompactionLimit(handoff, agent, compact_result)
        current_input = "Trusted handoff from previous context:\n" + handoff + "\n\nContinue answering.\n" + prompt
    raise RuntimeError("unreachable compaction loop")


async def reflect_question(context: HarnessContext, agent: Agent[HarnessContext], answer_result: Any, question: Question, candidate: AgentAnswer, score: int, trace_name: str) -> AgentReflection:
    agent.instructions = reflection_instructions(context)
    agent.tools = make_tools(context, "reflection")
    agent.tool_use_behavior = {"stop_at_tool_names": ["submit_reflection", "agent_raise"]}
    input_items = answer_result.to_input_list(mode="preserve_all")
    input_items.append({"role": "user", "content": "Correct answer: " + question.answer + "\nYour candidate answer: " + candidate.answer + f"\nJudge score out of 5: {score}\nReflect on why the answer was right or wrong, then extract reusable reflection guidance."})
    result, _compacted = await run_agent_stream(agent, input_items, context, context.run_dir / "traces" / f"{trace_name}-reflection.json", force_submit_tool="submit_reflection")
    if '"status": "raised"' in str(result.final_output):
        raise AgentRaised(str(result.final_output))
    return AgentReflection.model_validate_json(result.final_output)


async def judge_answer(config: HarnessConfig, reference_answer: str, candidate_answer: str, trace_path: Path | None = None) -> int:
    client = openai_client(config)
    parsed = await client.responses.parse(
        model=config.main_model,
        reasoning={"effort": config.reasoning_effort},
        input="Reason briefly, then score semantic equivalence from 0 to 5. Ignore extra video IDs, timestamps, and row IDs.\nReference answer:\n" + reference_answer + "\nCandidate answer:\n" + candidate_answer,
        text_format=JudgeScore,
    )
    if trace_path is not None:
        save_response_trace(trace_path, parsed)
    score = parsed.output_parsed.score
    if score < 0 or score > 5:
        raise RuntimeError(f"judge score out of range: {score}")
    return score


async def summarize_reflections(config: HarnessConfig, payload: str, trace_path: Path | None = None) -> str:
    response = await openai_client(config).responses.create(
        model=config.main_model,
        reasoning={"effort": config.reasoning_effort},
        input="Summarize these rollout reflections into concise updater instructions. Keep them abstract and remove dataset-specific sensitive details.\n" + payload,
    )
    if trace_path is not None:
        save_response_trace(trace_path, response)
    return extract_output_text(response)
