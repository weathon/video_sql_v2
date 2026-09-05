#!/usr/bin/env python3
"""Make a self-contained Sites source checkout from the exported board."""
import json
import shutil
from pathlib import Path

VIS = Path(__file__).resolve().parent
ROOT = VIS.parent
TARGET = VIS / 'publish'
DIST = TARGET / 'dist'
DIST.mkdir(parents=True, exist_ok=True)
(TARGET / '.openai').mkdir(exist_ok=True)
shutil.copy2(VIS / '.openai/hosting.json', TARGET / '.openai/hosting.json')
for name in ('index.html', 'board.js', 'board.css'):
    shutil.copy2(VIS / name, DIST / name)
shutil.copytree(VIS / 'data', DIST / 'data', dirs_exist_ok=True)
index_path = DIST / 'data/index.js'
index = json.loads(index_path.read_text().removeprefix('window.BOARD_DATA=').removesuffix(';\n'))
index['source'] = 'runs/' + index['run'] + '/'
index_path.write_text('window.BOARD_DATA=' + json.dumps(index, ensure_ascii=False) + ';\n')
source_paths = set()
for row in index['rollouts']:
    detail = json.loads((VIS / 'data' / (row['id'] + '.js')).read_text().split('=', 1)[1].removesuffix(';\n'))
    source_paths.update(detail['media'])
    source_paths.update(t['key'] for t in detail['traces'] + detail['summary_traces'])
    if detail['result'] is not None:
        source_paths.add(row['id'] + '/result.json')
for batch in index['batches']:
    source_paths.update(t['key'] for t in batch['traces'])
run = ROOT / 'runs' / index['run']
total_bytes = 0
for number, relative in enumerate(sorted(source_paths), 1):
    source = run / relative
    target = DIST / index['source'] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    total_bytes += source.stat().st_size
    if number % 5000 == 0:
        print(f'Copied {number}/{len(source_paths)} linked source files', flush=True)
print(json.dumps({'linked_source_files': len(source_paths), 'source_bytes': total_bytes, 'directory': str(TARGET)}))
