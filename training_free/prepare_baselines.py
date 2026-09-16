"""Prepare the shared MAGIC/SQL EgoLife artifacts without regenerating annotations."""

import argparse
import asyncio
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pyarrow as pa

from training_free.index_schema import DDL, EDGE_TABLES, NODE_TABLES, SEARCH_TABLES, fact_interval, timestamp_seconds
from training_free.normalize import normalize_text


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/egolife/A1_JAKE')
VIDEO_SOURCE = Path('/raid/wg25r/egolife/A1_JAKE')
QA_SOURCE = Path('/raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json')
sys.path.insert(0, str(ROOT / 'third_party/magic_video/src'))


def prepare(root: Path) -> None:
    from worldmm.memory.unified.graph import MultimodalGraph
    from worldmm.memory.unified.nodes import make_entity_id

    magic = root / 'magic'
    magic.mkdir(parents=True, exist_ok=True)
    for source in SOURCE.rglob('*.json'):
        target = magic / source.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    caption_paths = {g: str(SOURCE / f'captions/A1_JAKE_{g}.json') for g in ('30sec', '3min', '10min', '1h')}
    captions = json.loads(Path(caption_paths['30sec']).read_text())
    graph = MultimodalGraph()
    graph.add_episode_nodes(caption_paths)
    graph.add_visual_nodes(captions, {})
    graph.add_entity_nodes(str(SOURCE / 'episodic_memory/openie_results_openai_gpt-oss-120b.json'), captions, 'A1_JAKE')
    graph.add_semantic_nodes(str(SOURCE / 'semantic_memory/semantic_consolidation_results_openai_gpt-oss-120b.json'), 'A1_JAKE')
    for node in graph.nodes.values():
        node.text = normalize_text(node.text)
        for field in ('subject', 'predicate', 'object'):
            if field in node.metadata:
                node.metadata[field] = normalize_text(node.metadata[field])
    for granularity in caption_paths:
        records = json.loads(Path(caption_paths[granularity]).read_text())
        for record in records:
            record['text'] = normalize_text(record['text'])
        (magic / f'captions/A1_JAKE_{granularity}.json').write_text(json.dumps(records, ensure_ascii=False))
    semantic_path = magic / 'semantic_memory/semantic_consolidation_results_openai_gpt-oss-120b.json'
    semantic = json.loads(semantic_path.read_text())
    for snapshot in semantic.values():
        snapshot['consolidated_semantic_triples'] = [[normalize_text(part) for part in triple] for triple in snapshot['consolidated_semantic_triples']]
    semantic_path.write_text(json.dumps(semantic, ensure_ascii=False))
    rows = {table: [] for table in DDL}
    for node in graph.nodes.values():
        rows[node.node_type.value].append({'id': node.id, 'node_type': node.node_type.value, 'text': node.text, 'start_ts': node.start_ts, 'end_ts': node.end_ts, 'metadata': json.dumps(node.metadata, ensure_ascii=False)})
    for edge in graph.edges:
        table = edge.edge_type.value
        if table == 'co_clip':
            table += '_' + ('visual_episode' if graph.nodes[edge.src_id].node_type.value == 'visual_clip' else 'episode_visual')
        rows[table].append({'src_id': edge.src_id, 'dst_id': edge.dst_id, 'edge_type': edge.edge_type.value, 'weight': edge.weight})
    topics = json.loads((SOURCE / 'topic_chains/topic_chains.json').read_text())
    for chain_id, chain in topics.items():
        entity_id = make_entity_id(chain['entity'], 'A1_JAKE')
        chain['entity'] = normalize_text(chain['entity'])
        rows['topic_chains'].append({'id': chain_id, 'entity': chain['entity'], 'has_lifecycle': chain['has_lifecycle']})
        if entity_id in graph.nodes:
            rows['topic_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
        for index, fact in enumerate(chain['facts']):
            fact['fact'] = normalize_text(fact['fact'])
            if 'location' in fact:
                fact['location'] = normalize_text(fact['location'])
            rows['topic_facts'].append({'id': f'{chain_id}:{index}', 'chain_id': chain_id, 'time': fact['time'], 'fact': fact['fact'], 'location': fact['location'] if 'location' in fact else None})
    (magic / 'topic_chains/topic_chains.json').write_text(json.dumps(topics, ensure_ascii=False))
    events = json.loads((SOURCE / 'event_chains/step3_enriched_chains.json').read_text())
    for index, chain in enumerate(events):
        chain_id = str(index)
        for name in chain['key_entities']:
            entity_id = make_entity_id(name, 'A1_JAKE')
            if entity_id in graph.nodes:
                rows['event_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
        chain['name'] = normalize_text(chain['name'])
        chain['key_entities'] = [normalize_text(name) for name in chain['key_entities']]
        rows['event_chains'].append({'id': chain_id, 'name': chain['name'], 'key_entities': chain['key_entities']})
        for step_index, step in enumerate(chain['steps']):
            step['description'] = normalize_text(step['description'])
            rows['event_steps'].append({'id': f'{chain_id}:{step_index}', 'chain_id': chain_id, 'step_index': step_index, **step})
    (magic / 'event_chains/step3_enriched_chains.json').write_text(json.dumps(events, ensure_ascii=False))
    database = root / 'index.duckdb'
    with duckdb.connect(str(database)) as connection:
        tables = {row[0] for row in connection.execute('SHOW TABLES').fetchall()}
        if tables and tables != set(DDL):
            raise ValueError(f'Incomplete database schema: {database}: {sorted(tables)}')
        if not tables:
            connection.execute('BEGIN')
            for table, ddl in DDL.items():
                connection.execute(ddl)
                if rows[table]:
                    arrow_rows = pa.Table.from_pylist(rows[table])
                    connection.register('source_rows', arrow_rows)
                    connection.execute(f'INSERT INTO {table} BY NAME SELECT * FROM source_rows')
                    connection.unregister('source_rows')
            connection.execute('COMMIT')
    graph.save(str(root / 'graph-unembedded.pkl'))
    questions = []
    original = json.loads(QA_SOURCE.read_text())
    for item in original:
        question = re.sub(r'\bWhich of the following people\b', 'Who', item['question'], flags=re.I)
        question = re.sub(r'(?:\n|\s)(?:\([Aa]\)|[Aa][.)])\s+.*?(?:\([Bb]\)|[Bb][.)])\s+.*', '', question, flags=re.S).strip()
        query_time = int(item['query_time']['date'].removeprefix('DAY') + item['query_time']['time'].zfill(8))
        reference = item['choice_' + item['answer'].lower()]
        questions.append({'id': item['ID'], 'question': normalize_text(question), 'query_time': query_time, 'reference_answer': normalize_text(reference)})
    (root / 'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2))
    selected = random.Random(0).sample(range(len(questions)), 100)
    (root / 'qa-rule-audit.json').write_text(json.dumps([{'original': original[i], 'converted': questions[i]} for i in selected], ensure_ascii=False, indent=2))
    (root / 'latency-ids.json').write_text(json.dumps(random.Random(0).sample([q['id'] for q in questions], 20)))
    (root / 'node-counts.json').write_text(json.dumps({table: len(records) for table, records in rows.items()}, indent=2))
    print(f'PREPARED database={database} questions={len(questions)} nodes={len(graph.nodes)}', flush=True)


def probe_video(source: Path, root: Path) -> dict:
    target = root / 'videos/A1_JAKE' / source.relative_to(VIDEO_SOURCE)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    output = root / 'video-metadata' / source.relative_to(VIDEO_SOURCE).with_suffix('.json')
    if output.exists():
        return json.loads(output.read_text())
    command = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_streams', '-show_entries', 'stream=avg_frame_rate,duration,start_time,nb_frames,time_base', '-of', 'json', str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print(f'ffprobe input={target} error={result.stderr}', flush=True)
        result.check_returncode()
    payload = json.loads(result.stdout)
    stream = payload['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    fps = numerator / denominator
    day, _, _, time = source.stem.split('_')
    timestamp = int(day.removeprefix('DAY') + time)
    start = timestamp_seconds(timestamp, fps)
    info = {'path': str(target), 'relative_path': str(target.relative_to(root)), 'original_path': 'data/EgoLife/A1_JAKE/' + str(source.relative_to(VIDEO_SOURCE)), 'timeline_start_s': start, 'timeline_end_s': start + float(stream['duration']), 'fps': fps, 'duration': float(stream['duration']), 'total_frames': int(stream['nb_frames']), 'command': command}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def media(root: Path) -> None:
    sources = sorted(VIDEO_SOURCE.glob('DAY*/*.mp4'))
    records = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(probe_video, source, root): source for source in sources}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 100 == 0:
                print(f'MEDIA {len(records)}/{len(sources)}', flush=True)
    records.sort(key=lambda item: item['timeline_start_s'])
    (root / 'media-map.json').write_text(json.dumps({'sources': records}, ensure_ascii=False))
    print(f'MEDIA_COMPLETE sources={len(records)}', flush=True)


async def embed_text(root: Path, endpoint: str) -> None:
    with duckdb.connect(str(root / 'index.duckdb'), read_only=True) as connection:
        texts = []
        seen = set()
        for table, field in SEARCH_TABLES.items():
            for (text,) in connection.execute(f'SELECT {field} FROM {table} ORDER BY id').fetchall():
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
    directory = root / 'embedding-shards/text'
    directory.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=None) as client:
        for offset in range(0, len(texts), 512):
            batch = texts[offset:offset + 512]
            path = directory / f'{offset:09d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if saved['texts'].tolist() != batch:
                        raise ValueError(f'Text shard inputs changed: {path}')
                continue
            print(f'EMBED_TEXT offset={offset} count={len(batch)} total={len(texts)}', flush=True)
            response = await client.post(endpoint.rstrip('/') + '/embeddings', json={'model': 'qwen3-vl-embedding-8b', 'input': batch})
            if response.is_error:
                print(f'EMBED_TEXT error offset={offset}: {response.text}', flush=True)
            response.raise_for_status()
            data = sorted(response.json()['data'], key=lambda row: row['index'])
            if [row['index'] for row in data] != list(range(len(batch))):
                raise ValueError(f'Invalid embedding indexes at text offset={offset}')
            vectors = np.asarray([row['embedding'] for row in data], dtype=np.float32)
            if vectors.shape != (len(batch), 4096) or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
                raise ValueError(f'Invalid text embeddings at offset={offset}: shape={vectors.shape}')
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            np.savez(path, texts=np.asarray(batch), vectors=vectors)
    print(f'TEXT_COMPLETE unique_texts={len(texts)}', flush=True)


def cache_caption_clip(caption: dict, source: dict, root: Path, time_format: str = 'egolife') -> dict:
    from worldmm.memory.unified.nodes import caption_to_timestamps, make_visual_id

    start_ts, end_ts = caption_to_timestamps(caption)
    start_s = float(start_ts) if time_format == 'mmlifelong' else timestamp_seconds(start_ts, source['fps'])
    end_s = float(end_ts) if time_format == 'mmlifelong' else timestamp_seconds(end_ts, source['fps'])
    first = max(0, round((start_s - source['timeline_start_s']) * source['fps']))
    stop = min(source['total_frames'], round((end_s - source['timeline_start_s']) * source['fps']))
    if first >= stop:
        raise ValueError(f'Caption has no frames in its associated video: {caption}')
    relative = Path(source['relative_path']).relative_to('videos')
    if time_format == 'mmlifelong':
        relative = Path(source['video_id']) / f'{start_s}-{end_s}.mp4'
    clip = root / 'videos/caption-clips' / relative
    metadata_path = root / 'clip-metadata' / relative.with_suffix('.json')
    if metadata_path.exists():
        info = json.loads(metadata_path.read_text())
        clip.stat()
        if info['caption_start_ts'] != start_ts or info['caption_end_ts'] != end_ts:
            raise ValueError(f'Cached caption interval changed: {metadata_path}')
        return info
    clip.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if first == 0 and stop == source['total_frames']:
        if not clip.exists():
            os.link(source['path'], clip)
        command = None
    else:
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-v', 'error', '-threads', '1', '-i', source['path'], '-map', '0:v:0', '-an', '-vf', f'trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS', '-filter_threads', '1', '-fps_mode', 'passthrough', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '0', '-threads', '1', str(clip)]
        if time_format == 'mmlifelong':
            command[command.index('-i'):command.index('-i')] = ['-ss', str(first / source['fps'])]
            command[command.index('-vf') + 1] = f'trim=end_frame={stop-first},setpts=PTS-STARTPTS'
            command[-1:-1] = ['-frames:v', str(stop - first)]
        print(f'CACHE_CLIP source={source["path"]} frames={first}:{stop} output={clip}', flush=True)
        metadata_path.with_suffix('.command.json').write_text(json.dumps(command))
        result = subprocess.run(command, capture_output=True, text=True)
        metadata_path.with_suffix('.log').write_text(result.stderr)
        if result.returncode:
            print(f'CACHE_CLIP error source={source["path"]}: {result.stderr}', flush=True)
            result.check_returncode()
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate,nb_frames,duration,width,height,pix_fmt', '-of', 'json', str(clip)], capture_output=True, text=True)
    if probe.returncode:
        print(f'CACHE_CLIP ffprobe error path={clip}: {probe.stderr}', flush=True)
        probe.check_returncode()
    stream = json.loads(probe.stdout)['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    if int(stream['nb_frames']) != stop - first or numerator / denominator != source['fps']:
        raise ValueError(f'Cached clip frame count or fps differs: {clip}: {stream}')
    node_id = f"{source['video_id']}:vis_{start_ts}_{end_ts}" if time_format == 'mmlifelong' else make_visual_id(caption['date'], str(caption['start_time']))
    info = {'node_id': node_id, 'original_path': caption['video_path'], 'path': str(clip), 'relative_path': str(clip.relative_to(root)), 'source_relative_path': source['relative_path'], 'caption_start_ts': start_ts, 'caption_end_ts': end_ts, 'caption_start_s': start_s, 'caption_end_s': end_s, 'source_first_frame': first, 'source_stop_frame': stop, 'timeline_start_s': source['timeline_start_s'] + first / source['fps'], 'timeline_end_s': source['timeline_start_s'] + stop / source['fps'], 'fps': source['fps'], 'total_frames': stop - first, 'stream': stream, 'command': command}
    if time_format == 'mmlifelong':
        info['video_id'] = source['video_id']
        info['clip_key'] = caption['clip_key']
    metadata_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


async def embed_video(root: Path, endpoint: str, time_format: str = 'egolife') -> None:
    sources = {row['original_path']: row for row in json.loads((root / 'media-map.json').read_text())['sources']}
    caption_file = 'mmlifelong_30sec.json' if time_format == 'mmlifelong' else 'A1_JAKE_30sec.json'
    captions = json.loads((root / 'magic/captions' / caption_file).read_text())
    paths = {row['clip_key'] if time_format == 'mmlifelong' else row['video_path']: row for row in captions}
    directory = root / 'embedding-shards/video-grouped'
    directory.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(64)
    clip_executor = ThreadPoolExecutor(max_workers=10)
    clips = {}
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=64)) as client:
        async def one(index: int, path: str) -> None:
            async with semaphore:
                info = await asyncio.get_running_loop().run_in_executor(clip_executor, cache_caption_clip, paths[path], sources[paths[path]['video_path']], root, time_format)
                clips[path] = info
                shard_directory = directory / f'{index // 1000:03d}'
                shard_directory.mkdir(exist_ok=True)
                output = shard_directory / f'{index:06d}.npz'
                if output.exists():
                    with np.load(output, allow_pickle=False) as saved:
                        if str(saved['video_path']) != path or str(saved['sampling']) != 'qwen3_vl_uniform_caption':
                            raise ValueError(f'Video shard ID or sampling mismatch: {output}')
                    return
                request = {'model': 'qwen3-vl-embedding-8b', 'messages': [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': Path(info['path']).as_uri()}}]}], 'media_io_kwargs': {'video': {'video_backend': 'qwen3_vl_uniform', 'backend': 'torchcodec', 'trace_path': info['path'], 'num_frames': -1, 'num_ffmpeg_threads': 4, 'seek_mode': 'exact'}}, 'mm_processor_kwargs': {'do_sample_frames': False}}
                (shard_directory / f'{index:06d}-request.json').write_text(json.dumps(request))
                response = await client.post(endpoint.rstrip('/') + '/embeddings', json=request)
                if response.is_error:
                    print(f'EMBED_VIDEO error path={path}: {response.text}', flush=True)
                response.raise_for_status()
                payload = response.json()
                (shard_directory / f'{index:06d}-response.json').write_text(json.dumps(payload))
                vector = np.asarray(payload['data'][0]['embedding'], dtype=np.float32)
                if vector.shape != (4096,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError(f'Invalid video embedding: {path}')
                vector /= np.linalg.norm(vector)
                np.savez(output, video_path=np.asarray(path), sampling=np.asarray('qwen3_vl_uniform_caption'), vector=vector)
                print(f'EMBED_VIDEO index={index} sampling=qwen3_vl_uniform_caption saved={output.name}', flush=True)
        try:
            async with asyncio.TaskGroup() as group:
                for index, path in enumerate(paths):
                    group.create_task(one(index, path))
        finally:
            clip_executor.shutdown(wait=True, cancel_futures=True)
    (root / 'clip-map.json').write_text(json.dumps(clips, ensure_ascii=False))
    print(f'VIDEO_COMPLETE videos={len(paths)}', flush=True)


def assemble(root: Path) -> None:
    from worldmm.memory.unified.graph import MultimodalGraph

    vectors = {'': np.zeros(4096, dtype=np.float32)}
    for path in sorted((root / 'embedding-shards/text').glob('*.npz')):
        with np.load(path, allow_pickle=False) as shard:
            for text, vector in zip(shard['texts'], shard['vectors'], strict=True):
                vectors[str(text)] = vector
    visual = {}
    for path in sorted((root / 'embedding-shards/video-grouped').glob('*/*.npz')):
        with np.load(path, allow_pickle=False) as shard:
            visual[str(shard['video_path'])] = shard['vector']
    graph = MultimodalGraph.load(str(root / 'graph-unembedded.pkl'))
    for node in graph.nodes.values():
        node.text_embedding = vectors[node.text]
        if node.node_type.value == 'visual_clip':
            key = node.metadata['clip_key'] if 'clip_key' in node.metadata else node.metadata['video_path']
            node.visual_embedding = visual[key]
    graph.save(str(root / 'magic/graph.pkl'))
    with (root / 'magic/visual_embeddings.pkl').open('wb') as output:
        pickle.dump(visual, output)
    directory = root / 'embeddings'
    directory.mkdir(exist_ok=True)
    with duckdb.connect(str(root / 'index.duckdb'), read_only=True) as connection:
        for table, field in SEARCH_TABLES.items():
            records = connection.execute(f'SELECT id, {field} FROM {table} ORDER BY id').fetchall()
            np.save(directory / f'{table}.row_ids.npy', np.asarray([row[0] for row in records]))
            matrix = np.lib.format.open_memmap(directory / f'{table}.vectors.npy', mode='w+', dtype=np.float32, shape=(len(records), 4096))
            for index, (_, text) in enumerate(records):
                matrix[index] = vectors[text]
            matrix.flush()
            if table == 'visual_clip':
                matrix = np.lib.format.open_memmap(directory / f'{table}.video_vectors.npy', mode='w+', dtype=np.float32, shape=(len(records), 4096))
                for index, (node_id, _) in enumerate(records):
                    matrix[index] = graph.nodes[node_id].visual_embedding
                matrix.flush()
    print('ASSEMBLED graph, visual pickle and searchable matrices', flush=True)


def databases(root: Path) -> None:
    questions = json.loads((root / 'questions.json').read_text())
    directory = root / 'questions'
    directory.mkdir(exist_ok=True)
    with duckdb.connect(str(root / 'index.duckdb'), read_only=True) as full:
        facts = full.execute('SELECT id, time FROM topic_facts').fetchall()
        intervals = [(row_id, *interval) for row_id, time in facts if (interval := fact_interval(time)) is not None]
    for question in questions:
        path = directory / (question['id'] + '.duckdb')
        with duckdb.connect(str(path)) as connection:
            tables = {row[0] for row in connection.execute('SHOW TABLES').fetchall()}
            if tables == set(DDL):
                print(f'DATABASE_REUSE question={question["id"]} path={path.name}', flush=True)
                continue
            if tables:
                raise ValueError(f'Incomplete database schema: {path}: {sorted(tables)}')
            source_path = str(root / 'index.duckdb').replace("'", "''")
            connection.execute(f"ATTACH '{source_path}' AS complete (READ_ONLY)")
            connection.execute('BEGIN')
            for table, ddl in DDL.items():
                connection.execute(ddl)
            cutoff = question['query_time']
            for table in NODE_TABLES:
                connection.execute(f'INSERT INTO {table} SELECT * FROM complete.{table} WHERE start_ts > 0 AND end_ts <= ?', [cutoff])
            allowed_facts = pa.table({'id': [row_id for row_id, start, end in intervals if start > 0 and end <= cutoff]}, schema=pa.schema([('id', pa.string())]))
            connection.register('allowed_facts', allowed_facts)
            connection.execute('INSERT INTO topic_chains SELECT c.* FROM complete.topic_chains c WHERE c.id IN (SELECT f.chain_id FROM complete.topic_facts f JOIN allowed_facts a ON f.id = a.id)')
            connection.execute('INSERT INTO topic_facts SELECT f.* FROM complete.topic_facts f JOIN allowed_facts a ON f.id = a.id')
            step_filter = "CAST(replace(day, 'DAY', '') || replace(end_time, ':', '') || '00' AS BIGINT) <= ?"
            connection.execute(f'INSERT INTO event_chains SELECT c.* FROM complete.event_chains c WHERE c.id IN (SELECT chain_id FROM complete.event_steps WHERE {step_filter})', [cutoff])
            connection.execute(f'INSERT INTO event_steps SELECT * FROM complete.event_steps WHERE {step_filter}', [cutoff])
            for prefix in ('topic', 'event'):
                connection.execute(f'INSERT INTO {prefix}_entities SELECT r.* FROM complete.{prefix}_entities r JOIN {prefix}_chains c ON r.chain_id=c.id JOIN entity e ON r.entity_id=e.id')
            for table, (src, dst) in EDGE_TABLES.items():
                connection.execute(f'INSERT INTO {table} SELECT e.* FROM complete.{table} e JOIN {src} s ON e.src_id=s.id JOIN {dst} d ON e.dst_id=d.id')
            connection.execute('COMMIT')
        print(f'DATABASE question={question["id"]} cutoff={cutoff} path={path.name}', flush=True)
    print(f'DATABASES_COMPLETE count={len(questions)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'media', 'embed-text', 'embed-video', 'assemble', 'databases'])
    parser.add_argument('--root', type=Path, default=Path('/raid/videosql-video-artifacts'))
    parser.add_argument('--embedding-url', default='http://127.0.0.1:8114/v1')
    parser.add_argument('--time-format', choices=('egolife', 'mmlifelong'), default='egolife')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        if args.time_format == 'mmlifelong':
            from training_free.mmlifelong_data import prepare as prepare_mm
            prepare_mm(args.root)
        else:
            prepare(args.root)
    elif args.stage == 'media':
        if args.time_format == 'mmlifelong':
            from training_free.mmlifelong_data import media as media_mm
            media_mm(args.root)
        else:
            media(args.root)
    elif args.stage == 'embed-text':
        asyncio.run(embed_text(args.root, args.embedding_url))
    elif args.stage == 'embed-video':
        asyncio.run(embed_video(args.root, args.embedding_url, args.time_format))
    elif args.stage == 'assemble':
        assemble(args.root)
    elif args.stage == 'databases':
        if args.time_format == 'mmlifelong':
            print(f'DATABASES_COMPLETE shared_full_database={args.root / "index.duckdb"}', flush=True)
        else:
            databases(args.root)


if __name__ == '__main__':
    main()
