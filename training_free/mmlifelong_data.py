"""MM-Lifelong Month artifacts from the released MAGIC merge annotations."""

import json
import os
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa

from training_free.index_schema import DDL, NODE_TABLES
from training_free.normalize import normalize_text


SOURCE = Path('/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong')
DATASET = Path('/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443')
MM_DDL = dict(DDL)
for table in NODE_TABLES:
    MM_DDL[table] = DDL[table].replace('BIGINT', 'DOUBLE').removesuffix(')') + ', video_id VARCHAR)'
for table in ('topic_chains', 'event_chains'):
    MM_DDL[table] = DDL[table].removesuffix(')') + ', video_id VARCHAR)'
for table in ('topic_facts', 'event_steps'):
    MM_DDL[table] = DDL[table].removesuffix(')') + ', video_id VARCHAR, start_ts DOUBLE, end_ts DOUBLE)'
MM_SCHEMA = '\n'.join(MM_DDL.values()) + '\nAll 14 broadcasts are available to every question. video_id identifies the broadcast. start_ts/end_ts and semantic_hits.start_s/end_s are seconds within that video, including fractional seconds. There is no question cutoff. topic_facts.time and event_steps.start_time/end_time preserve HH:MM:SS strings; event_steps.day is NULL. Pass video_id with the video-local interval to clip_qa.'


def prepare(root: Path) -> None:
    from worldmm.memory.unified.graph import MultimodalGraph
    from worldmm.memory.unified.nodes import make_entity_id

    graph = MultimodalGraph()
    rows = {table: [] for table in MM_DDL}
    all_captions = {g: [] for g in ('30sec', '3min', '10min', '1h')}
    magic = root / 'magic'
    (magic / 'captions').mkdir(parents=True, exist_ok=True)
    for video in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
        video_id = video.name
        source = video / 'merge'
        output = magic / video_id
        output.mkdir(exist_ok=True)
        caption_paths = {g: str(source / f'captions/{video_id}_{g}.json') for g in all_captions}
        captions = json.loads(Path(caption_paths['30sec']).read_text())
        native = MultimodalGraph()
        native.add_episode_nodes(caption_paths)
        native.add_visual_nodes(captions, {})
        native.add_entity_nodes(str(source / 'episodic_memory/openie_results.json'), captions, 'IShowSpeed')
        native.add_semantic_nodes(str(source / 'semantic_memory/semantic_consolidation_results.json'), 'IShowSpeed')
        node_ids = {nid: f'{video_id}:vis_{node.start_ts}_{node.end_ts}' if node.node_type.value == 'visual_clip' else f'{video_id}:{nid}' for nid, node in native.nodes.items()}
        for original_id, node in native.nodes.items():
            node.id = node_ids[original_id]
            node.text = normalize_text(node.text)
            node.metadata['video_id'] = video_id
            for field in ('subject', 'predicate', 'object'):
                if field in node.metadata and isinstance(node.metadata[field], str):
                    node.metadata[field] = normalize_text(node.metadata[field])
            if node.node_type.value == 'visual_clip':
                node.metadata['clip_key'] = f'{video_id}:{float(node.start_ts)}:{float(node.end_ts)}'
            graph.nodes[node.id] = node
            rows[node.node_type.value].append({'id': node.id, 'node_type': node.node_type.value, 'text': node.text, 'start_ts': node.start_ts, 'end_ts': node.end_ts, 'metadata': json.dumps(node.metadata, ensure_ascii=False), 'video_id': video_id})
        for edge in native.edges:
            edge.src_id, edge.dst_id = node_ids[edge.src_id], node_ids[edge.dst_id]
            graph.edges.append(edge)
            table = edge.edge_type.value
            if table == 'co_clip':
                table += '_' + ('visual_episode' if graph.nodes[edge.src_id].node_type.value == 'visual_clip' else 'episode_visual')
            rows[table].append({'src_id': edge.src_id, 'dst_id': edge.dst_id, 'edge_type': edge.edge_type.value, 'weight': edge.weight})
        for granularity, path in caption_paths.items():
            for caption in json.loads(Path(path).read_text()):
                caption['text'] = normalize_text(caption['text'])
                caption['video_id'] = video_id
                caption['clip_key'] = f"{video_id}:{float(caption['start_sec'])}:{float(caption['end_sec'])}"
                all_captions[granularity].append(caption)
        topics = json.loads((source / 'topic_chains.json').read_text())
        for native_id, chain in topics.items():
            chain_id = f'{video_id}:{native_id}'
            entity_id = f"{video_id}:{make_entity_id(chain['entity'], 'IShowSpeed')}"
            chain['entity'] = normalize_text(chain['entity'])
            rows['topic_chains'].append({'id': chain_id, 'entity': chain['entity'], 'has_lifecycle': chain['has_lifecycle'], 'video_id': video_id})
            if entity_id in graph.nodes:
                rows['topic_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
            for index, fact in enumerate(chain['facts']):
                fact['fact'] = normalize_text(fact['fact'])
                seconds = sum(float(part) * scale for part, scale in zip(fact['time'].split(':'), (3600, 60, 1), strict=True))
                rows['topic_facts'].append({'id': f'{chain_id}:{index}', 'chain_id': chain_id, 'time': fact['time'], 'fact': fact['fact'], 'location': None, 'video_id': video_id, 'start_ts': seconds, 'end_ts': seconds})
        (output / 'topic_chains.json').write_text(json.dumps(topics, ensure_ascii=False))
        events = json.loads((source / 'storyline_step3_enriched.json').read_text())
        for index, chain in enumerate(events):
            chain_id = f'{video_id}:{index}'
            for name in chain['key_entities']:
                entity_id = f"{video_id}:{make_entity_id(name, 'IShowSpeed')}"
                if entity_id in graph.nodes:
                    rows['event_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
            chain['name'] = normalize_text(chain['name'])
            chain['key_entities'] = [normalize_text(name) for name in chain['key_entities']]
            rows['event_chains'].append({'id': chain_id, 'name': chain['name'], 'key_entities': chain['key_entities'], 'video_id': video_id})
            for step_index, step in enumerate(chain['steps']):
                step['description'] = normalize_text(step['description'])
                start, end = [sum(float(part) * scale for part, scale in zip(step[field].split(':'), (3600, 60, 1), strict=True)) for field in ('start_time', 'end_time')]
                rows['event_steps'].append({'id': f'{chain_id}:{step_index}', 'chain_id': chain_id, 'step_index': step_index, 'day': None, **step, 'video_id': video_id, 'start_ts': start, 'end_ts': end})
        (output / 'event_chains.json').write_text(json.dumps(events, ensure_ascii=False))
        print(f'PREPARE_VIDEO video_id={video_id} nodes={len(native.nodes)} captions={len(captions)}', flush=True)
    for granularity, captions in all_captions.items():
        (magic / f'captions/mmlifelong_{granularity}.json').write_text(json.dumps(captions, ensure_ascii=False))
    with duckdb.connect(str(root / 'index.duckdb')) as connection:
        tables = {row[0] for row in connection.execute('SHOW TABLES').fetchall()}
        if tables and tables != set(MM_DDL):
            raise ValueError(f'Incomplete MM-Lifelong schema: {sorted(tables)}')
        if not tables:
            connection.execute('BEGIN')
            for table, ddl in MM_DDL.items():
                connection.execute(ddl)
                if rows[table]:
                    connection.register('source_rows', pa.Table.from_pylist(rows[table]))
                    connection.execute(f'INSERT INTO {table} BY NAME SELECT * FROM source_rows')
                    connection.unregister('source_rows')
            connection.execute('COMMIT')
    graph.save(str(root / 'graph-unembedded.pkl'))
    original = json.loads((DATASET / 'month/val.json').read_text())
    questions = [{'id': item['index'], 'question': item['question'], 'query_time': None, 'reference_answer': item['answer']} for item in original]
    (root / 'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2))
    (root / 'node-counts.json').write_text(json.dumps({table: len(records) for table, records in rows.items()}, indent=2))
    print(f'PREPARED questions={len(questions)} nodes={len(graph.nodes)}', flush=True)


def media(root: Path) -> None:
    records = []
    for video in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
        video_id = video.name
        source = DATASET / f'videos/month/{video_id}.mp4'
        target = root / f'videos/month/{video_id}.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source.resolve(), target)
        output = root / f'video-metadata/{video_id}.json'
        if output.exists():
            records.append(json.loads(output.read_text()))
            continue
        command = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate,duration,nb_frames,time_base', '-of', 'json', str(target)]
        print(f'MEDIA_PROBE video_id={video_id} path={target}', flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(f'MEDIA_ERROR {result.stderr}', flush=True)
            result.check_returncode()
        stream = json.loads(result.stdout)['streams'][0]
        numerator, denominator = map(int, stream['r_frame_rate'].split('/'))
        info = {'video_id': video_id, 'path': str(target), 'relative_path': str(target.relative_to(root)), 'original_path': f'data/MM-Lifelong/month/{video_id}.mp4', 'timeline_start_s': 0.0, 'timeline_end_s': float(stream['duration']), 'fps': numerator / denominator, 'duration': float(stream['duration']), 'total_frames': int(stream['nb_frames']), 'command': command}
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        records.append(info)
    (root / 'media-map.json').write_text(json.dumps({'sources': records}, ensure_ascii=False))
    print(f'MEDIA_COMPLETE sources={len(records)}', flush=True)
