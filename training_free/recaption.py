"""Re-caption every MM-Lifelong Month interval from video frames with Qwen3.8-27B-FP8."""

import argparse
import asyncio
import base64
import io
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import wandb
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel

from training_free.normalize import normalize_text


SOURCE = Path('/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong')
DATASET = Path('/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443')
GRANULARITIES = ('30sec', '3min', '10min', '1h')
FRAMES = {'30sec': 32, '3min': 64, '10min': 128, '1h': 256}
LONG_SIDE = 640
EMPTY_COMPLETION_ATTEMPTS = 5
MODEL = 'Qwen/Qwen3.8-27B-FP8'
INSTRUCTIONS = (
    'You are annotating one segment of a long livestream broadcast for a searchable memory index. '
    'Report only what the frames and the transcript support. Do not guess. '
    'scene: the setting, who is present and what they wear, and every action and event in the order it happens. '
    'objects: every distinguishable object, device, screen, food, drink, vehicle, animal and piece of furniture '
    'that is visible, each with its state. State covers on or off, open or closed, full or empty, who holds it, '
    'moving or still, worn or removed, lit or unlit, running or stopped, and a count when there is more than one. '
    'onscreen_text: every legible on-screen or physical text string, transcribed exactly, with where it appears.'
)


class ObjectState(BaseModel):
    name: str
    state: str


class Caption(BaseModel):
    scene: str
    objects: list[ObjectState]
    onscreen_text: list[str]


def read_frames(video_path: str, start_s: float, decode_s: float, fps: float, count: int) -> tuple[list[bytes], list[float]]:
    available = int(decode_s * fps)
    if available < count:
        raise RuntimeError(f'Interval holds {available} frames, fewer than the {count} requested: {video_path} start={start_s} decode={decode_s}')
    indices = [round(value) for value in np.linspace(0, available - 1, count)]
    if len(set(indices)) != count:
        raise RuntimeError(f'Uniform frame indices collide: {video_path} start={start_s} decode={decode_s} available={available} count={count}')
    selection = '+'.join(f'eq(n\\,{index})' for index in indices)
    command = [
        'ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-v', 'info',
        '-ss', f'{start_s:.6f}', '-i', video_path, '-t', f'{decode_s:.6f}', '-map', '0:v:0',
        '-vf', f'select={selection},scale={LONG_SIDE}:{LONG_SIDE}:force_original_aspect_ratio=decrease,showinfo',
        '-filter_threads', '1', '-fps_mode', 'passthrough', '-frames:v', str(count),
        '-threads', '1', '-f', 'image2pipe', '-c:v', 'png', 'pipe:1',
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log = process.stderr.decode(errors='replace')
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg exit={process.returncode} command={command} log={log[-4000:]}')
    times = [start_s + float(value) for value in re.findall(r'\bpts_time:\s*(-?[\d.]+)', log)]
    jpegs = []
    with av.open(io.BytesIO(process.stdout), format='image2pipe', options={'video_codec': 'png'}) as container:
        for decoded in container.decode(video=0):
            output = io.BytesIO()
            decoded.to_image().convert('RGB').save(output, format='JPEG', quality=90)
            jpegs.append(output.getvalue())
    if len(jpegs) != count or len(times) != count:
        raise RuntimeError(f'ffmpeg returned {len(jpegs)} frames and {len(times)} timestamps, expected {count}: command={command} log={log[-4000:]}')
    return jpegs, times


def index_text(caption: Caption) -> str:
    sections = [caption.scene.strip()]
    if caption.objects:
        sections.append('objects: ' + '; '.join(f'{entry.name.strip()}: {entry.state.strip()}' for entry in caption.objects))
    if caption.onscreen_text:
        sections.append('on-screen text: ' + ' | '.join(value.strip() for value in caption.onscreen_text))
    text = normalize_text('\n'.join(section for section in sections if section))
    if not text:
        raise ValueError(f'Caption produced empty index text: {caption.model_dump_json()}')
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=Path('runs/mmlifelong-recaption'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8244/v1')
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--frame-workers', type=int, default=32)
    parser.add_argument('--granularities', nargs='+', choices=GRANULARITIES, default=list(GRANULARITIES))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--wandb-name', required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


async def run(args) -> None:
    durations = {path.stem: json.loads(path.read_text()) for path in (Path('runs/mmlifelong-month/artifacts/video-metadata')).glob('*.json')}
    transcripts = {}
    items = []
    for granularity in args.granularities:
        for video in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
            video_id = video.name
            if video_id not in transcripts:
                transcripts[video_id] = json.loads((video / f'whisper/{video_id}_whisper.json').read_text())
            records = json.loads((video / f'merge/captions/{video_id}_{granularity}.json').read_text())
            selected = records[:args.limit] if args.limit is not None else records
            for index, record in enumerate(selected):
                items.append({'granularity': granularity, 'video_id': video_id, 'index': index, 'start_sec': float(record['start_sec']), 'end_sec': float(record['end_sec']), 'video_path': record['video_path']})
    caption_root = args.run_root / 'captions'
    (args.run_root / 'logs').mkdir(parents=True, exist_ok=True)
    print(f'RECAPTION items={len(items)} granularities={args.granularities} concurrency={args.concurrency}', flush=True)
    run_handle = wandb.init(entity='3dsmile', project='video-sql-v2', name=args.wandb_name, dir=str(args.run_root), config={'model': MODEL, 'frames': FRAMES, 'long_side': LONG_SIDE, 'reasoning_effort': 'low', 'instructions': INSTRUCTIONS, 'granularities': args.granularities, 'items': len(items), 'base_url': args.base_url, 'concurrency': args.concurrency, 'limit': args.limit})
    counters = {'done': 0, 'reused': 0, 'input_tokens': 0, 'output_tokens': 0}
    lock = threading.Lock()
    started = time.monotonic()
    client = AsyncOpenAI(base_url=args.base_url, api_key='EMPTY', timeout=None)
    semaphore = asyncio.Semaphore(args.concurrency)
    frame_executor = ThreadPoolExecutor(max_workers=args.frame_workers)

    async def one(item: dict) -> None:
        output = caption_root / item['granularity'] / item['video_id'] / f'{item["index"]:06d}.json'
        if output.exists():
            with lock:
                counters['reused'] += 1
                counters['done'] += 1
            return
        async with semaphore:
            count = FRAMES[item['granularity']]
            source = f'runs/mmlifelong-month/artifacts/videos/month/{item["video_id"]}.mp4'
            decode_s = min(item['end_sec'], durations[item['video_id']]['duration']) - item['start_sec']
            jpegs, times = await asyncio.get_running_loop().run_in_executor(frame_executor, read_frames, source, item['start_sec'], decode_s, durations[item['video_id']]['fps'], count)
            spoken = [segment for segment in transcripts[item['video_id']] if item['start_sec'] <= (segment['start_sec'] + segment['end_sec']) / 2 < item['end_sec']]
            transcript = '\n'.join(f'[{segment["start_sec"] - item["start_sec"]:.0f}s] {segment["text"]}' for segment in spoken) if spoken else 'none'
            prompt = f'Segment length: {item["end_sec"] - item["start_sec"]:.2f} seconds. The {count} images are uniformly sampled from it, in order.\nSpeech transcript for this segment:\n{transcript}'
            content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()}} for jpeg in jpegs]
            content.append({'type': 'text', 'text': prompt})
            for attempt in range(EMPTY_COMPLETION_ATTEMPTS):
                completion = await client.beta.chat.completions.parse(model=MODEL, messages=[{'role': 'system', 'content': INSTRUCTIONS}, {'role': 'user', 'content': content}], response_format=Caption, reasoning_effort='low')
                caption = completion.choices[0].message.parsed
                if caption is not None:
                    break
                print(f'RECAPTION_EMPTY granularity={item["granularity"]} video_id={item["video_id"]} index={item["index"]} attempt={attempt} finish_reason={completion.choices[0].finish_reason} output_tokens={completion.usage.completion_tokens}', flush=True)
            if caption is None:
                raise RuntimeError(f'Caption request returned no parsed content in {EMPTY_COMPLETION_ATTEMPTS} attempts: item={item} finish_reason={completion.choices[0].finish_reason} content={completion.choices[0].message.content!r}')
            text = index_text(caption)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({**item, 'decode_seconds': decode_s, 'frame_count': count, 'frame_times': times, 'attempts': attempt + 1, 'transcript_segments': len(spoken), 'transcript': transcript, 'prompt': prompt, 'response_id': completion.id, 'finish_reason': completion.choices[0].finish_reason, 'usage': completion.usage.model_dump(), 'message': completion.choices[0].message.model_dump(), 'parsed': caption.model_dump(), 'text': text}, ensure_ascii=False, indent=2) + '\n')
            with lock:
                counters['done'] += 1
                counters['input_tokens'] += completion.usage.prompt_tokens
                counters['output_tokens'] += completion.usage.completion_tokens
                payload = {'progress/done': counters['done'], 'progress/reused': counters['reused'], 'progress/total': len(items), 'tokens/in': counters['input_tokens'], 'tokens/out': counters['output_tokens'], 'throughput/items_per_second': counters['done'] / (time.monotonic() - started)}
                run_handle.log(payload)
            if counters['done'] % 100 == 0:
                print(f'RECAPTION_PROGRESS done={counters["done"]}/{len(items)} reused={counters["reused"]} in={counters["input_tokens"]} out={counters["output_tokens"]}', flush=True)

    try:
        async with asyncio.TaskGroup() as group:
            for item in items:
                group.create_task(one(item))
    finally:
        frame_executor.shutdown(wait=True, cancel_futures=True)
        await client.close()
    for granularity in args.granularities:
        merged = []
        for video in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
            video_id = video.name
            records = json.loads((video / f'merge/captions/{video_id}_{granularity}.json').read_text())
            selected = records[:args.limit] if args.limit is not None else records
            for index, record in enumerate(selected):
                saved = json.loads((caption_root / granularity / video_id / f'{index:06d}.json').read_text())
                merged.append({'start_sec': float(record['start_sec']), 'end_sec': float(record['end_sec']), 'text': saved['text'], 'video_path': record['video_path']})
        (caption_root / f'mmlifelong_{granularity}.json').write_text(json.dumps(merged, ensure_ascii=False))
        print(f'RECAPTION_MERGED granularity={granularity} count={len(merged)}', flush=True)
    print(f'RECAPTION_COMPLETE items={len(items)} reused={counters["reused"]} in={counters["input_tokens"]} out={counters["output_tokens"]}', flush=True)
    run_handle.finish()


if __name__ == '__main__':
    main()
