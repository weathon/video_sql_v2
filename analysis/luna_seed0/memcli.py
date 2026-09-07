#!/usr/bin/env python
"""CLI for inspecting the Luna seed0 eval run: memory DB, source frames, trace files."""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import uuid

ROOT = "/raid/wg25r/video_sql_v2"
RUN = os.path.join(ROOT, "runs/training_free_luna_seed0_7510e535")
TEST_FILE = os.path.join(ROOT, "data/test-seed-0.json")
RUNTIME = os.path.join(ROOT, "data/runtime")


def out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def clip(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated, total {len(text)} chars>"


def load_segment(path):
    with open(path) as fh:
        return json.load(fh)


def cmd_sql(args):
    import duckdb

    con = duckdb.connect(os.path.join(RUNTIME, args.timeline, "index.duckdb"), read_only=True)
    cur = con.execute(args.sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchmany(args.max_rows)
    has_more = bool(cur.fetchone())
    con.close()
    shaped = []
    for row in rows:
        item = []
        for cell in row:
            if isinstance(cell, str) and len(cell) > args.max_chars:
                item.append({"value": cell[: args.max_chars], "original_length": len(cell)})
            else:
                item.append(cell if isinstance(cell, (str, int, float, bool, type(None))) else str(cell))
        shaped.append(item)
    out({"columns": columns, "rows": shaped, "row_count": len(shaped), "has_more": has_more})


def cmd_schema(args):
    import duckdb

    con = duckdb.connect(os.path.join(RUNTIME, args.timeline, "index.duckdb"), read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    schema = {}
    for table in tables:
        cols = con.execute(f'DESCRIBE "{table}"').fetchall()
        count = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        schema[table] = {"rows": count, "columns": [{"name": c[0], "type": c[1]} for c in cols]}
    con.close()
    out(schema)


def cmd_question(args):
    with open(TEST_FILE) as fh:
        items = json.load(fh)
    match = [q for q in items if q["timeline"] == args.timeline and q["index"] == args.index]
    if not match:
        raise SystemExit(f"no question {args.timeline}-{args.index} in {TEST_FILE}")
    out(match[0])


def cmd_rollouts(args):
    qdir = os.path.join(RUN, "val", f"question-{args.timeline}-{args.index}")
    results = []
    for rdir in sorted(glob.glob(os.path.join(qdir, "rollout-*"))):
        entry = {"rollout_dir": rdir}
        rpath = os.path.join(rdir, "result.json")
        if os.path.exists(rpath):
            with open(rpath) as fh:
                entry["result"] = json.load(fh)
        traces = {}
        for adir in sorted(glob.glob(os.path.join(rdir, "attempt-*"))):
            files = sorted(glob.glob(os.path.join(adir, "traces", "*-segment-*.json")))
            traces[os.path.basename(adir)] = files
            errors = sorted(glob.glob(os.path.join(adir, "*error*.json")))
            if errors:
                entry.setdefault("error_files", []).extend(errors)
        entry["segment_files"] = traces
        results.append(entry)
    out({"question": f"{args.timeline}-{args.index}", "question_dir": qdir, "rollouts": results})


def cmd_trace_list(args):
    seg = load_segment(args.file)
    outputs = {}
    for i, item in enumerate(seg["history"]):
        if item.get("type") == "function_call_output":
            outputs[item["call_id"]] = (i, item["output"])
    steps = []
    for i, item in enumerate(seg["history"]):
        kind = item.get("type")
        if kind == "function_call":
            oidx, oval = outputs.get(item["call_id"], (None, None))
            steps.append(
                {
                    "index": i,
                    "tool": item["name"],
                    "arguments": clip(item["arguments"], args.max_chars),
                    "output_index": oidx,
                    "output": clip(oval, args.max_chars) if oval is not None else None,
                }
            )
        elif kind is None and item.get("role") in ("user", "assistant"):
            steps.append({"index": i, "role": item["role"], "content": clip(item["content"], args.max_chars)})
    out(
        {
            "file": args.file,
            "agent": seg["agent"].get("name"),
            "input": clip(seg["input"], args.max_chars),
            "is_complete": seg["is_complete"],
            "error": seg["error"],
            "final_output": seg["final_output"],
            "usage": [
                {"input_tokens": r["usage"]["input_tokens"], "output_tokens": r["usage"]["output_tokens"]}
                for r in seg["raw_responses"]
            ],
            "steps": steps,
        }
    )


def cmd_trace_item(args):
    seg = load_segment(args.file)
    item = seg["history"][args.index]
    result = {"index": args.index, "item": item}
    if item.get("type") == "function_call":
        for i, other in enumerate(seg["history"]):
            if other.get("type") == "function_call_output" and other["call_id"] == item["call_id"]:
                result["output_index"] = i
                result["output"] = other["output"]
    if item.get("type") == "reasoning":
        result["item"] = {k: v for k, v in item.items() if k != "encrypted_content"}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(text) > args.max_chars:
        text = text[: args.max_chars] + f"\n...<truncated, total {len(text)} chars; raise --max-chars to see more>"
    print(text)


def cmd_copy_frame(args):
    if args.path:
        src = args.path
    else:
        matches = glob.glob(os.path.join(args.rollout_dir, "attempt-*", "media", f"frame-{args.frame_id}.jpg"))
        if not matches:
            raise SystemExit(f"frame {args.frame_id} not found under {args.rollout_dir}")
        src = matches[0]
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, os.path.basename(src))
    shutil.copyfile(src, dst)
    meta_src = src[: -len(".jpg")] + ".json"
    meta = None
    if os.path.exists(meta_src):
        with open(meta_src) as fh:
            meta = json.load(fh)
    out({"image": dst, "source": src, "frame_meta": meta})


def cmd_frame(args):
    with open(os.path.join(RUNTIME, args.timeline, "media-map.json")) as fh:
        sources = json.load(fh)["sources"]
    os.makedirs(args.out_dir, exist_ok=True)
    images = []
    for k in range(args.count):
        t = args.time + k * args.step
        source = next((s for s in sources if s["timeline_start_s"] <= t < s["timeline_end_s"]), None)
        if source is None:
            raise SystemExit(f"timeline time {t} is outside {args.timeline} media map")
        offset = t - source["timeline_start_s"]
        dst = os.path.join(args.out_dir, f"t{t:.3f}-{uuid.uuid4().hex[:8]}.jpg")
        cmd = ["ffmpeg", "-nostdin", "-ss", f"{offset:.3f}", "-i", source["path"], "-frames:v", "1",
               "-vf", "scale='min(854,iw)':-2", "-q:v", "3", "-y", dst]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(dst):
            raise SystemExit(f"ffmpeg failed for t={t}: {proc.stderr[-2000:]}")
        images.append({"time_s": t, "source_video_id": source["source_video_id"], "source_offset_s": offset, "image": dst})
    out({"images": images})


parser = argparse.ArgumentParser(description="Luna seed0 run inspection CLI")
sub = parser.add_subparsers(dest="cmd", required=True)

p = sub.add_parser("sql", help="read-only SELECT against a timeline memory DB")
p.add_argument("--timeline", required=True)
p.add_argument("--sql", required=True)
p.add_argument("--max-rows", type=int, default=50)
p.add_argument("--max-chars", type=int, default=800)
p.set_defaults(func=cmd_sql)

p = sub.add_parser("schema", help="tables, row counts and columns of a timeline DB")
p.add_argument("--timeline", required=True)
p.set_defaults(func=cmd_schema)

p = sub.add_parser("question", help="question record with its metadata from the test file")
p.add_argument("--timeline", required=True)
p.add_argument("--index", type=int, required=True)
p.set_defaults(func=cmd_question)

p = sub.add_parser("rollouts", help="per-rollout results and trace file paths for one question")
p.add_argument("--timeline", required=True)
p.add_argument("--index", type=int, required=True)
p.set_defaults(func=cmd_rollouts)

p = sub.add_parser("trace-list", help="tool-call skeleton of one segment JSON")
p.add_argument("--file", required=True)
p.add_argument("--max-chars", type=int, default=400)
p.set_defaults(func=cmd_trace_list)

p = sub.add_parser("trace-item", help="full input/output of one history item")
p.add_argument("--file", required=True)
p.add_argument("--index", type=int, required=True)
p.add_argument("--max-chars", type=int, default=20000)
p.set_defaults(func=cmd_trace_item)

p = sub.add_parser("copy-frame", help="copy a frame image already produced in a trace to a tmp dir")
p.add_argument("--path")
p.add_argument("--frame-id")
p.add_argument("--rollout-dir")
p.add_argument("--out-dir", default="/tmp/luna-frames")
p.set_defaults(func=cmd_copy_frame)

p = sub.add_parser("frame", help="extract frames from the source video at logical timeline times")
p.add_argument("--timeline", required=True)
p.add_argument("--time", type=float, required=True)
p.add_argument("--count", type=int, default=1)
p.add_argument("--step", type=float, default=1.0)
p.add_argument("--out-dir", default="/tmp/luna-frames")
p.set_defaults(func=cmd_frame)

args = parser.parse_args()
args.func(args)
