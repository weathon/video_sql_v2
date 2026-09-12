"""Upload the finished EgoLife artifacts to the project token owner's private dataset."""

import json
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


ROOT = Path('/raid/videosql-video-artifacts')
REPOSITORY = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'INDEX.md', 'MAGIC_UPSTREAM.txt', 'questions.json', 'qa-rule-audit.json', 'latency-ids.json', 'node-counts.json', 'media-map.json', 'index.duckdb', 'magic/graph.pkl', 'magic/visual_embeddings.pkl']
DIRECTORIES = ['questions', 'embeddings', 'magic', 'videos', 'video-metadata', 'results', 'validation']


def main():
    (ROOT / 'README.md').write_text((REPOSITORY / 'README.md').read_text().replace('(notes/index.md)', '(INDEX.md)').replace('(third_party/magic_video/UPSTREAM)', '(MAGIC_UPSTREAM.txt)'))
    (ROOT / 'INDEX.md').write_text((REPOSITORY / 'notes/index.md').read_text().replace('(../README.md)', '(README.md)').replace('third_party/magic_video/UPSTREAM', 'MAGIC_UPSTREAM.txt'))
    (ROOT / 'MAGIC_UPSTREAM.txt').write_text((REPOSITORY / 'third_party/magic_video/UPSTREAM').read_text())
    for name in FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
    questions = json.loads((ROOT / 'questions.json').read_text())
    for question in questions:
        if not (ROOT / 'questions' / f'{question["id"]}.duckdb').is_file():
            raise FileNotFoundError(f'Missing database for question {question["id"]}')
    files = {name: ROOT / name for name in FILES}
    for directory in DIRECTORIES:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file():
                files[str(path.relative_to(ROOT))] = path
    token = dotenv_values(REPOSITORY / '.env')['HF_TOKEN']
    api = HfApi(token=token)
    owner = api.whoami()['name']
    repo_id = owner + '/egolife-a1-jake-magic-videosql-qwen3vl8b'
    print(f'UPLOAD repo={repo_id} files={len(files)} bytes={sum(path.stat().st_size for path in files.values())}', flush=True)
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type='dataset').private:
        raise RuntimeError(f'Dataset is not private: {repo_id}')
    api.upload_large_folder(repo_id=repo_id, repo_type='dataset', folder_path=ROOT, private=True, allow_patterns=list(files), num_workers=10, print_report_every=3600)
    remote = {item.path: item.size for item in api.list_repo_tree(repo_id, repo_type='dataset', recursive=True) if isinstance(item, RepoFile)}
    for name, path in files.items():
        if remote[name] != path.stat().st_size:
            raise ValueError(f'Uploaded size differs: {name}: local={path.stat().st_size} remote={remote[name]}')
    for name in ('README.md', 'INDEX.md', 'questions.json', 'latency-ids.json', 'media-map.json'):
        downloaded = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=name, token=token, cache_dir='/raid/huggingface')
        if Path(downloaded).read_bytes() != files[name].read_bytes():
            raise ValueError(f'Uploaded content differs: {name}')
    result = {'repo_id': repo_id, 'url': f'https://huggingface.co/datasets/{repo_id}', 'private': api.repo_info(repo_id, repo_type='dataset').private, 'files': len(files), 'bytes': sum(path.stat().st_size for path in files.values())}
    (ROOT / 'upload-result.json').write_text(json.dumps(result, indent=2) + '\n')
    api.upload_file(path_or_fileobj=ROOT / 'upload-result.json', path_in_repo='upload-result.json', repo_id=repo_id, repo_type='dataset')
    print(f'UPLOAD_COMPLETE {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
