import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import duckdb
import httpx
import numpy as np
from PIL import Image, ImageDraw
from torchcodec.decoders import VideoDecoder

from src.preprocess.normalize import normalize_text


SEARCH_TABLES = {
    "asr_segments": ("text", "video_id", "start_sec", "end_sec"),
    "captions": ("text", "video_id", "start_sec", "end_sec"),
    "entities": ("entity", "video_id", "NULL", "NULL"),
    "episodic_triples": ("text", "video_id", "start_sec", "end_sec"),
    "semantic_triples": ("text", "video_id", "NULL", "NULL"),
    "activities": ("text", "video_id", "start_sec", "end_sec"),
    "event_chains": ("text", "video_id", "NULL", "NULL"),
    "event_steps": ("text", "video_id", "start_sec", "end_sec"),
    "topic_chains": ("entity", "video_id", "NULL", "NULL"),
    "topic_facts": ("text", "video_id", "start_sec", "start_sec"),
    "video_clips": ("''", "video_id", "start_sec", "end_sec"),
}


class AgentRaised(RuntimeError):
    pass


class Notes:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[str] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, text: str) -> dict:
        self.entries.append(text)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "entry_count": len(self.entries)}

    def read(self) -> dict:
        return {"entries": list(self.entries)}


class VideoSQLHarness:
    def __init__(self, database_path: Path, embedding_endpoint: str, embedding_model: str, clip_dir: Path, media_dir: Path, notes_path: Path):
        self.database_path = database_path
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.clip_dir = clip_dir
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.notes = Notes(notes_path)

    def run_sql(self, sql: str) -> dict:
        connection = duckdb.connect(str(self.database_path), read_only=True)
        executor = ThreadPoolExecutor(max_workers=1)

        def execute_query():
            print(f"run_sql sql={sql}")
            result = connection.execute(sql)
            columns = [column[0] for column in result.description]
            return columns, result.fetchmany(101)

        future = executor.submit(execute_query)
        try:
            columns, fetched = future.result(timeout=30)
        except TimeoutError:
            connection.interrupt()
            raise TimeoutError("SQL exceeded the 30 second limit")
        except (duckdb.ParserException, duckdb.BinderException, duckdb.CatalogException, duckdb.ConversionException, duckdb.InvalidInputException) as error:
            return {"status": "query_error", "error": f"{type(error).__name__}: {error}"}
        finally:
            executor.shutdown(wait=True)
            connection.close()
        rows = []
        for source_row in fetched[:100]:
            row = []
            for cell in source_row:
                rendered = str(cell)
                if len(rendered) > 500:
                    row.append({"text": rendered[:500], "original_characters": len(rendered), "truncated": True})
                elif cell is not None and not isinstance(cell, (str, int, float, bool)):
                    row.append(rendered)
                else:
                    row.append(cell)
            rows.append(row)
        return {"status": "ok", "columns": columns, "rows": rows, "hidden_more": len(fetched) == 101}

    def semantic_search(self, query: str, n: int, table: str) -> dict:
        if table not in SEARCH_TABLES:
            raise ValueError(f"table must be one of {sorted(SEARCH_TABLES)}")
        print(f"semantic_search table={table} n={n} query={query}")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": [{"type": "text", "text": normalize_text(query)}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]
        response = httpx.post(
            self.embedding_endpoint,
            json={"model": self.embedding_model, "messages": messages, "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True},
            timeout=1800,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        text_column, video_column, start_column, end_column = SEARCH_TABLES[table]
        connection = duckdb.connect(str(self.database_path), read_only=True)
        rows = connection.execute(
            f"""
            SELECT s.row_id, s.{video_column}, {start_column}, {end_column}, {text_column},
                   list_cosine_similarity(e.vector, ?::FLOAT[]) AS score
            FROM {table} s JOIN embeddings e
              ON e.source_table = ? AND e.row_id = s.row_id
            ORDER BY score DESC
            LIMIT ?
            """,
            [vector, table, n],
        ).fetchall()
        connection.close()
        return {"table": table, "results": [{"row_id": row_id, "video_id": video_id, "start_sec": start, "end_sec": end, "text": text[:500], "score": score} for row_id, video_id, start, end, text, score in rows]}

    def add_note(self, text: str) -> dict:
        return self.notes.add(text)

    def read_notes(self) -> dict:
        return self.notes.read()

    def agent_raise(self, reason: str):
        raise AgentRaised(reason)

    def sample_video(self, video_id: str, start_sec: float, end_sec: float, fps: float, x1: int, x2: int, y1: int, y2: int, return_caption: bool) -> dict:
        if not 0 <= x1 < x2 <= 1000 or not 0 <= y1 < y2 <= 1000:
            raise ValueError("crop coordinates must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000")
        if start_sec < 0 or end_sec <= start_sec or fps <= 0:
            raise ValueError("start_sec, end_sec, and fps define an invalid sampling interval")
        connection = duckdb.connect(str(self.database_path), read_only=True)
        duration = connection.execute("SELECT duration_sec FROM videos WHERE video_id = ?", [video_id]).fetchone()[0]
        if end_sec > duration:
            connection.close()
            raise ValueError(f"end_sec {end_sec} exceeds video duration {duration}")
        captions = []
        if return_caption:
            captions = connection.execute("SELECT start_sec, end_sec, text FROM captions WHERE video_id = ? AND source = 'merged' AND granularity_sec = 30 AND end_sec > ? AND start_sec < ? ORDER BY start_sec", [video_id, start_sec, end_sec]).fetchall()
        clips = connection.execute("SELECT row_id, start_sec, end_sec FROM video_clips WHERE video_id = ? AND end_sec > ? AND start_sec < ? ORDER BY start_sec", [video_id, start_sec, end_sec]).fetchall()
        connection.close()

        timestamps = np.arange(start_sec, end_sec, 1.0 / fps).tolist()
        frames: list[tuple[float, Image.Image] | None] = [None] * len(timestamps)
        for row_id, clip_start, clip_end in clips:
            selected = [(index, timestamp) for index, timestamp in enumerate(timestamps) if clip_start <= timestamp < clip_end]
            if not selected:
                continue
            clip_index = int(row_id.rsplit(":", 1)[1])
            clip_path = self.clip_dir / video_id / f"{clip_index:06d}.mp4"
            if not clip_path.is_file():
                raise FileNotFoundError(clip_path)
            decoder = VideoDecoder(clip_path, dimension_order="NHWC", num_ffmpeg_threads=4, seek_mode="approximate")
            clip_begin = decoder.metadata.begin_stream_seconds
            if clip_begin is None:
                raise RuntimeError(f"cached clip has no start time row_id={row_id} path={clip_path}")
            batch = decoder.get_frames_played_at([clip_begin + timestamp - clip_start for _, timestamp in selected]).data.cpu().numpy()
            for (frame_index, timestamp), frame in zip(selected, batch, strict=True):
                height, width = frame.shape[:2]
                crop = frame[y1 * height // 1000:y2 * height // 1000, x1 * width // 1000:x2 * width // 1000]
                frames[frame_index] = (timestamp, Image.fromarray(crop).resize((480, 270)))
        if any(frame is None for frame in frames):
            raise RuntimeError(f"cached clips did not cover requested interval video={video_id} start={start_sec} end={end_sec}")

        pages = []
        for page_index, page_start in enumerate(range(0, len(frames), 12)):
            page_frames = frames[page_start:page_start + 12]
            canvas = Image.new("RGB", (1920, 900), "black")
            draw = ImageDraw.Draw(canvas)
            for index, (timestamp, image) in enumerate(page_frames):
                column, row = index % 4, index // 4
                canvas.paste(image, (column * 480, row * 300))
                draw.text((column * 480 + 8, row * 300 + 274), f"{timestamp:.3f}s", fill="white")
            path = self.media_dir / f"sample-{video_id}-{start_sec:.3f}-{end_sec:.3f}-p{page_index}.jpg"
            canvas.save(path)
            pages.append(str(path))
        return {"contact_sheets": pages, "frame_timestamps": timestamps, "captions": [{"start_sec": start, "end_sec": end, "text": text} for start, end, text in captions]}

    def evidence_screenshots(self, video_id: str, timestamps: list[float]) -> list[str]:
        connection = duckdb.connect(str(self.database_path), read_only=True)
        clips = connection.execute("SELECT row_id, start_sec, end_sec FROM video_clips WHERE video_id = ? ORDER BY start_sec", [video_id]).fetchall()
        connection.close()
        frames = [None] * len(timestamps)
        for row_id, clip_start, clip_end in clips:
            selected = [(index, timestamp) for index, timestamp in enumerate(timestamps) if clip_start <= timestamp < clip_end]
            if not selected:
                continue
            clip_index = int(row_id.rsplit(":", 1)[1])
            clip_path = self.clip_dir / video_id / f"{clip_index:06d}.mp4"
            if not clip_path.is_file():
                raise FileNotFoundError(clip_path)
            decoder = VideoDecoder(clip_path, dimension_order="NHWC", num_ffmpeg_threads=4, seek_mode="approximate")
            clip_begin = decoder.metadata.begin_stream_seconds
            if clip_begin is None:
                raise RuntimeError(f"cached clip has no start time row_id={row_id} path={clip_path}")
            batch = decoder.get_frames_played_at([clip_begin + timestamp - clip_start for _, timestamp in selected]).data.cpu().numpy()
            for (frame_index, _), frame in zip(selected, batch, strict=True):
                frames[frame_index] = frame
        if any(frame is None for frame in frames):
            raise RuntimeError(f"cached clips did not cover all evidence timestamps video={video_id} timestamps={timestamps}")
        paths = []
        for index, (timestamp, frame) in enumerate(zip(timestamps, frames, strict=True)):
            path = self.media_dir / f"evidence-{video_id}-{timestamp:.3f}-{index}.jpg"
            Image.fromarray(frame).save(path)
            paths.append(str(path))
        return paths
