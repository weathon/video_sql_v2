"""Prepare the complete Month validation corpus from the released merge artifacts."""

import json
import os
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa

from training_free.index_schema import MM_DDL
from training_free.normalize import normalize_text


SOURCE = Path('/raid/huggingface/hub/datasets--jiazhengli7--magic-video-artifacts/snapshots/8573f317684ce55c0999f86386666d3d3fe76ae3/mmlifelong')
DATASET = Path('/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443')


def prepare(root):
    from worldmm.memory.unified.graph import MultimodalGraph
    from worldmm.memory.unified.nodes import make_entity_id

    rows = {table: [] for table in MM_DDL}
    all_captions = []
    for source in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
        video_id = source.name
        source = source / 'merge'
        target = root / 'magic' / video_id
        (target / 'captions').mkdir(parents=True, exist_ok=True)
        paths = {gran: str(source / 'captions' / f'{video_id}_{gran}.json') for gran in ('30sec', '3min', '10min', '1h')}
        captions = json.loads(Path(paths['30sec']).read_text())
        graph = MultimodalGraph()
        graph.add_episode_nodes(paths)
        graph.add_visual_nodes(captions, {})
        graph.add_entity_nodes(str(source / 'episodic_memory/openie_results.json'), captions, video_id)
        graph.add_semantic_nodes(str(source / 'semantic_memory/semantic_consolidation_results.json'), video_id)
        identities = {}
        for node in graph.nodes.values():
            identities[node.id] = f'{video_id}:vis:{node.start_ts}-{node.end_ts}' if node.node_type.value == 'visual_clip' else f'{video_id}:{node.id}'
        for edge in graph.edges:
            edge.src_id, edge.dst_id = identities[edge.src_id], identities[edge.dst_id]
        nodes = {}
        for node in graph.nodes.values():
            node.id = identities[node.id]
            node.text = normalize_text(node.text)
            node.metadata['video_id'] = video_id
            for field in ('subject', 'predicate', 'object'):
                if field in node.metadata and isinstance(node.metadata[field], str):
                    node.metadata[field] = normalize_text(node.metadata[field])
            if node.node_type.value == 'visual_clip':
                node.metadata['clip_key'] = f"{node.metadata['video_path']}:{node.start_ts}-{node.end_ts}"
            nodes[node.id] = node
            rows[node.node_type.value].append({'id': node.id, 'video_id': video_id, 'node_type': node.node_type.value, 'text': node.text, 'start_ts': node.start_ts, 'end_ts': node.end_ts, 'metadata': json.dumps(node.metadata, ensure_ascii=False)})
        graph.nodes = nodes
        for edge in graph.edges:
            table = edge.edge_type.value
            if table == 'co_clip':
                table += '_' + ('visual_episode' if graph.nodes[edge.src_id].node_type.value == 'visual_clip' else 'episode_visual')
            rows[table].append({'src_id': edge.src_id, 'dst_id': edge.dst_id, 'edge_type': edge.edge_type.value, 'weight': edge.weight})
        for gran, path in paths.items():
            records = json.loads(Path(path).read_text())
            for record in records:
                record['text'] = normalize_text(record['text'])
                record['video_id'] = video_id
            (target / 'captions' / f'{video_id}_{gran}.json').write_text(json.dumps(records, ensure_ascii=False))
            if gran == '30sec':
                all_captions.extend(records)
        topics = json.loads((source / 'topic_chains.json').read_text())
        for key, topic in topics.items():
            chain_id = f'{video_id}:topic:{key}'
            entity_id = f"{video_id}:{make_entity_id(topic['entity'], video_id)}"
            topic['entity'] = normalize_text(topic['entity'])
            rows['topic_chains'].append({'id': chain_id, 'video_id': video_id, 'entity': topic['entity'], 'has_lifecycle': topic['has_lifecycle']})
            if entity_id in nodes:
                rows['topic_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
            for index, fact in enumerate(topic['facts']):
                fact['fact'] = normalize_text(fact['fact'])
                seconds = sum(float(part) * scale for part, scale in zip(fact['time'].split(':'), (3600, 60, 1), strict=True))
                rows['topic_facts'].append({'id': f'{chain_id}:{index}', 'video_id': video_id, 'chain_id': chain_id, 'time': fact['time'], 'fact': fact['fact'], 'location': normalize_text(fact['location']) if 'location' in fact else None, 'start_ts': seconds, 'end_ts': seconds})
        (target / 'topic_chains.json').write_text(json.dumps(topics, ensure_ascii=False))
        events = json.loads((source / 'storyline_step3_enriched.json').read_text())
        for index, event in enumerate(events):
            chain_id = f'{video_id}:event:{index}'
            for name in event['key_entities']:
                entity_id = f'{video_id}:{make_entity_id(name, video_id)}'
                if entity_id in nodes:
                    rows['event_entities'].append({'chain_id': chain_id, 'entity_id': entity_id})
            event['name'] = normalize_text(event['name'])
            event['key_entities'] = [normalize_text(name) for name in event['key_entities']]
            rows['event_chains'].append({'id': chain_id, 'video_id': video_id, 'name': event['name'], 'key_entities': event['key_entities']})
            for step_index, step in enumerate(event['steps']):
                step['description'] = normalize_text(step['description'])
                for clock, numeric in (('start_time', 'start_ts'), ('end_time', 'end_ts')):
                    step[numeric] = sum(float(part) * scale for part, scale in zip(step[clock].split(':'), (3600, 60, 1), strict=True))
                rows['event_steps'].append({'id': f'{chain_id}:{step_index}', 'video_id': video_id, 'chain_id': chain_id, 'step_index': step_index, **step})
        (target / 'event_chains.json').write_text(json.dumps(events, ensure_ascii=False))
        graph.save(str(target / 'graph-unembedded.pkl'))
        print(f'PREPARE video={video_id} nodes={len(nodes)}', flush=True)
    with duckdb.connect(str(root / 'index.duckdb')) as connection:
        if connection.execute('SHOW TABLES').fetchall():
            raise ValueError('Preparation database already exists; use its saved artifacts or a new root')
        for table, ddl in MM_DDL.items():
            connection.execute(ddl)
            if rows[table]:
                connection.register('source_rows', pa.Table.from_pylist(rows[table]))
                connection.execute(f'INSERT INTO {table} BY NAME SELECT * FROM source_rows')
                connection.unregister('source_rows')
    questions = [{'id': str(item['index']), 'question': item['question'], 'query_time': None, 'reference_answer': item['answer']} for item in json.loads((DATASET / 'month/val.json').read_text())]
    (root / 'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2))
    (root / 'magic/captions.json').write_text(json.dumps(all_captions, ensure_ascii=False))
    (root / 'node-counts.json').write_text(json.dumps({table: len(records) for table, records in rows.items()}, indent=2))
    print(f'PREPARED questions={len(questions)} clips={len(all_captions)}', flush=True)


def media(root):
    records = []
    for source in sorted(SOURCE.iterdir(), key=lambda path: int(path.name)):
        video_id = source.name
        video = DATASET / 'videos/month' / f'{video_id}.mp4'
        target = root / 'videos/month' / video.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(video.resolve(), target)
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate,duration,start_time,nb_frames,time_base', '-of', 'json', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(f'PROBE input={video} error={result.stderr}', flush=True)
            result.check_returncode()
        stream = json.loads(result.stdout)['streams'][0]
        numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
        records.append({'video_id': video_id, 'path': str(target), 'relative_path': str(target.relative_to(root)), 'original_path': f'data/MM-Lifelong/month/{video_id}.mp4', 'timeline_start_s': 0.0, 'timeline_end_s': float(stream['duration']), 'fps': numerator / denominator, 'duration': float(stream['duration']), 'total_frames': int(stream['nb_frames']), 'command': command})
        print(f'MEDIA video={video_id} duration={stream["duration"]} frames={stream["nb_frames"]}', flush=True)
    (root / 'media-map.json').write_text(json.dumps({'sources': records}, ensure_ascii=False))
