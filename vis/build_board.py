#!/usr/bin/env python3
"""Export the continuation run for the framework-free, file:// compatible board."""
import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/qwen38_27b_fp8_epd_seed0_continuation'
OUT = ROOT / 'vis/data'
OUT.mkdir(exist_ok=True)
traces = {}
tool_counts = {}


def export_traces(directory):
    records = []
    for path in sorted(directory.glob('**/traces/*.json')):
        trace = json.loads(path.read_text())
        key = path.relative_to(RUN).as_posix()
        target = OUT / (key + '.js')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Raw responses duplicate history; the unmodified source remains linked.
        payload = {k: trace[k] for k in ('input', 'history', 'final_output', 'is_complete')}
        target.write_text('window.BOARD_CACHE[' + json.dumps(key) + ']=' + json.dumps(payload, ensure_ascii=False) + ';\n')
        calls = [e for e in trace['history'] if e.get('type') == 'function_call']
        for call in calls:
            tool_counts[call['name']] = tool_counts.get(call['name'], 0) + 1
        record = {
            'key': key, 'file': path.name, 'group': path.name.rsplit('-segment-', 1)[0],
            'attempt': path.relative_to(directory).parts[0] if path.parent.parent != directory else '',
            'events': len(trace['history']), 'calls': len(calls),
            'tools': sorted({e['name'] for e in calls}), 'complete': trace['is_complete'],
        }
        records.append(record)
        traces[key] = record
    return records


rollouts = []
batches = []
for batch in sorted(RUN.glob('train/epoch-*/batch-*')):
    batch_id = batch.parent.name + '/' + batch.name
    metadata_path = batch / 'batch.json'
    batch_record = {'id': batch_id, 'metadata': json.loads(metadata_path.read_text()) if metadata_path.exists() else None}
    updater = batch / 'updater/result.json'
    batch_record['update'] = json.loads(updater.read_text()) if updater.exists() else None
    batch_record['traces'] = export_traces(batch / 'updater')
    batches.append(batch_record)
    for question in sorted(batch.glob('question-*')):
        summary_path = question / 'summary/result.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        summary_traces = export_traces(question / 'summary')
        for directory in sorted(question.glob('rollout-*')):
            result_path = directory / 'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            records = export_traces(directory)
            answer_paths = sorted(directory.glob('attempt-*/traces/answer-*.json'))
            prompt = json.loads(answer_paths[0].read_text())['input'].removeprefix('Question: ').split('\n\nAvailable skills:')[0] if answer_paths else None
            errors = [{'path': p.relative_to(RUN).as_posix(), 'error': json.loads(p.read_text())} for p in sorted(directory.glob('**/*error.json'))]
            media = [p.relative_to(RUN).as_posix() for p in sorted(directory.glob('**/media/*')) if p.is_file()]
            key = directory.relative_to(RUN).as_posix()
            detail = {'result': result, 'traces': records, 'errors': errors, 'media': media, 'summary': summary, 'summary_traces': summary_traces}
            target = OUT / (key + '.js')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('window.BOARD_CACHE[' + json.dumps(key) + ']=' + json.dumps(detail, ensure_ascii=False) + ';\n')
            rollouts.append({
                'id': key, 'batch': batch_id, 'question_id': int(question.name.split('-')[1]),
                'rollout': int(directory.name.split('-')[1]), 'question': prompt,
                'status': result['status'] if result else 'no-result', 'score': result['score'] if result else None,
                'answer': result['answer'] if result else None,
                'traces': len(records), 'calls': sum(t['calls'] for t in records), 'errors': len(errors),
                'tools': sorted({tool for t in records for tool in t['tools']}),
            })
    print(batch_id, 'exported', flush=True)

versions = []
previous = None
for directory in [*sorted((RUN / 'skill-snapshots').iterdir()), RUN / 'skills']:
    files = {p.name: p.read_text() for p in sorted(directory.glob('*.md'))}
    changes = {}
    if previous is not None:
        for name in sorted(files.keys() | previous.keys()):
            before = previous[name] if name in previous else ''
            after = files[name] if name in files else ''
            if before != after:
                changes[name] = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='previous/' + name, tofile=directory.name + '/' + name))
    versions.append({'id': directory.name, 'files': files, 'changes': changes, 'baseline': previous is None})
    previous = files

config = json.loads((RUN / 'config.json').read_text())
payload = {'run': RUN.name, 'model': config['model'], 'built': datetime.now(timezone.utc).isoformat(),
           'source': '../runs/' + RUN.name + '/', 'rollouts': rollouts, 'batches': batches,
           'versions': versions, 'trace_count': len(traces), 'tools': tool_counts}
(OUT / 'index.js').write_text('window.BOARD_DATA=' + json.dumps(payload, ensure_ascii=False) + ';\n')
print(json.dumps({'rollouts': len(rollouts), 'traces': len(traces), 'versions': len(versions)}, indent=2))
