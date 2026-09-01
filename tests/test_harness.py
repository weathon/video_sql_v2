import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
from agents import RunContextWrapper
from PIL import Image

from src.harness import ClipContext, RolloutContext, clip_qa, run_sql, sample_video, semantic_search, write_trace


def test_write_trace_persists_input_without_image_bytes(tmp_path: Path) -> None:
    result = SimpleNamespace(raw_responses=[], new_items=[], input=[{"role": "user", "content": [{"type": "input_text", "text": "caption"}, {"type": "input_image", "image_url": "data:image/jpeg;base64,secret"}]}], is_complete=True, current_turn=1, context_wrapper=SimpleNamespace(usage={}), final_output="done")
    trace_path = tmp_path / "trace.json"
    write_trace(result, [], trace_path)
    payload = json.loads(trace_path.read_text())
    assert payload["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "caption"}, {"type": "input_image", "image_url": "<image_data_removed>"}]}]


async def test_run_sql_limits_cells_and_blocks_external_access(tmp_path: Path) -> None:
    database_path = tmp_path / "tool.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE TABLE items AS SELECT range AS item_id, repeat('x', 600) AS text FROM range(101)")
    connection.close()
    runtime = SimpleNamespace(database_path=database_path)
    context = RunContextWrapper(RolloutContext(runtime=runtime, attempt_dir=tmp_path, clip_max_turns=10))
    result = json.loads(await run_sql.__wrapped__(context, "SELECT * FROM items ORDER BY item_id"))
    assert len(result["rows"]) == 100
    assert result["has_more"] is True
    assert result["rows"][0][1] == {"text": "x" * 500, "length": 600}
    external = json.loads(await run_sql.__wrapped__(context, "SELECT * FROM read_csv_auto('/etc/passwd')"))
    assert "error" in external
    write = json.loads(await run_sql.__wrapped__(context, "DELETE FROM items"))
    assert "error" in write
    verification = duckdb.connect(str(database_path), read_only=True).execute("SELECT count(*) FROM items").fetchone()[0]
    assert verification == 101


async def test_model_correctable_tool_rejections(tmp_path: Path) -> None:
    runtime = SimpleNamespace(indexes={})
    rollout = RolloutContext(runtime=runtime, attempt_dir=tmp_path, clip_max_turns=10)
    invalid_table = json.loads(await semantic_search.__wrapped__(RunContextWrapper(rollout), "test", 5, "missing"))
    assert "Allowed tables" in invalid_table["error"]
    clip = ClipContext(rollout=rollout, video_id="4", start_s=0.0, end_s=10.0, call_id="validation")
    invalid_interval = await sample_video.__wrapped__(RunContextWrapper(clip), -1.0, 2.0, 1.0, 0, 0, 1000, 1000, True)
    assert "Sampling interval" in invalid_interval[0].text
    invalid_fps = await sample_video.__wrapped__(RunContextWrapper(clip), 0.0, 2.0, 0.0, 0, 0, 1000, 1000, True)
    assert "fps must be positive" in invalid_fps[0].text
    invalid_crop = await sample_video.__wrapped__(RunContextWrapper(clip), 0.0, 2.0, 1.0, 100, 0, 100, 1000, True)
    assert "Crop coordinates" in invalid_crop[0].text
    too_long = await clip_qa.__wrapped__(RunContextWrapper(rollout), "4", 0.0, 1201.0, "question")
    assert "longer than 20 minutes" in too_long[0].text


async def test_concurrent_sample_video_outputs_are_distinct_and_decodable(tmp_path: Path) -> None:
    runtime = SimpleNamespace(database_path=Path("data/pilot/pilot.duckdb"), clip_root=Path("data/pilot/clips"))
    rollout = RolloutContext(runtime=runtime, attempt_dir=tmp_path, clip_max_turns=10)
    first = ClipContext(rollout=rollout, video_id="4", start_s=0.0, end_s=2.0, call_id="first-call")
    second = ClipContext(rollout=rollout, video_id="4", start_s=0.0, end_s=2.0, call_id="second-call")
    first_output, second_output = await asyncio.gather(sample_video.__wrapped__(RunContextWrapper(first), 0.0, 2.0, 0.5, 0, 0, 1000, 1000, True), sample_video.__wrapped__(RunContextWrapper(second), 0.0, 2.0, 1.0, 100, 100, 900, 900, False))
    first_metadata = json.loads(first_output[0].text)
    second_metadata = json.loads(second_output[0].text)
    first_name = first_metadata["contact_sheets"][0]
    second_name = second_metadata["contact_sheets"][0]
    assert first_name != second_name
    assert "/" not in first_name
    assert "/" not in second_name
    assert first_output[1].image_url.startswith("data:image/jpeg;base64,")
    assert second_output[1].image_url.startswith("data:image/jpeg;base64,")
    with Image.open(tmp_path / "media" / first_name) as first_image:
        assert first_image.size == (1920, 900)
    with Image.open(tmp_path / "media" / second_name) as second_image:
        assert second_image.size == (1920, 900)
