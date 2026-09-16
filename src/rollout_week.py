"""Collect the MM-Lifelong Week main paths and immediate fixed-prefix branches."""

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import random
import re
import time
import traceback
import uuid
from pathlib import Path

import httpx
import httpx2
import openai
import wandb
from agents import ModelBehaviorError, OpenAIResponsesModel
from dotenv import dotenv_values
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import ValidationError
from wandb.sdk.lib.runid import generate_id

from src.harness import AgentRaised, InvalidTrace, REQUEST_TRACE, VideoSQLHarness, exception_chain, stripped_json
from training_free.evaluate_mm_official import SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443')
ARTIFACTS = Path('/raid/videosql-video-artifacts')
WEEK_MEDIA_MAP = Path('/raid/wg25r/video_sql_v2/data/runtime/week/media-map.json')
MODEL = 'Qwen/Qwen3.8-27B-FP8'
JUDGE_MODEL = 'gpt-5'
JUDGE_PROMPT = SYSTEM_PROMPT[:SYSTEM_PROMPT.index('\n\nEgoLife metadata rules:')]
ATTEMPTS = 4
LANE_CONCURRENCY = 10


async def record_wire_request(request):
    info = REQUEST_TRACE.get()
    path = Path(info['trace_dir']) / f'wire-{uuid.uuid4().hex}.json'
    record = {'url': str(request.url), 'trace_file': info['trace_file'], 'agent': info['agent'], 'body': stripped_json(json.loads(request.content))}
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')


def retryable(error):
    for item in exception_chain(error):
        if isinstance(item, openai.APIError) and item.code == 'insufficient_quota':
            return False
        if isinstance(item, (FileNotFoundError, PermissionError, KeyError, TypeError, AttributeError, NameError)):
            return False
    for item in exception_chain(error):
        if isinstance(item, (AgentRaised, InvalidTrace, ModelBehaviorError, ValidationError, openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError, openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError, httpx.TransportError, httpx2.TransportError)):
            return True
        if isinstance(item, openai.BadRequestError) and (item.code == 'context_length_exceeded' or any(text in str(item).lower() for text in ('maximum context length', 'context length exceeded', 'context window exceeded'))):
            return True
        if isinstance(item, httpx.HTTPStatusError) and (item.response.status_code == 429 or item.response.status_code >= 500):
            return True
    return False


class RecordedResponsesModel(OpenAIResponsesModel):
    def __init__(self, context):
        super().__init__(model=MODEL, openai_client=context.client)
        self.context = context

    async def stream_response(self, *args, **kwargs):
        info = dict(REQUEST_TRACE.get())
        path = self.context.run_dir / 'traces' / f'model-{self.context.answer_turns:03d}-{uuid.uuid4().hex}.json'
        info['trace_file'] = str(path)
        REQUEST_TRACE.set(info)
        record = {'agent': info['agent'], 'prefix_file': info['snapshot_file'], 'response': None, 'error': None}
        path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'MODEL_START turn={self.context.answer_turns} trace={path}', flush=True)
        try:
            async with self.context.model_limiter:
                async for event in super().stream_response(*args, **kwargs):
                    if event.type in ('response.completed', 'response.incomplete', 'response.failed'):
                        response = event.response
                        record['response'] = stripped_json(response)
                        usage = {'response_id': response.id, 'agent': info['agent'], 'model': response.model, 'usage': None if response.usage is None else response.usage.model_dump(), 'status': event.type}
                        self.context.response_usage[response.id] = usage
                        self.context.response_states[response.id] = info['snapshot_file']
                        if event.type == 'response.completed':
                            self.context.path_responses.append(usage)
                        path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
                    yield event
        except BaseException as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            print(f'MODEL_ERROR trace={path} error={type(error).__name__}: {error}', flush=True)
            raise
        finally:
            path.write_text(json.dumps(record, ensure_ascii=False) + '\n')


class WeekCollector:
    def __init__(self, directory, questions, clients, embedding_client, embedding_url, judge_client, run):
        self.directory = directory
        self.questions = {q['index']: q for q in questions}
        self.clients = clients
        self.embedding_client = embedding_client
        self.embedding_url = embedding_url
        self.judge_client = judge_client
        self.run = run
        self.limiters = [asyncio.Semaphore(LANE_CONCURRENCY), asyncio.Semaphore(LANE_CONCURRENCY)]
        self.queues = [asyncio.PriorityQueue(), asyncio.PriorityQueue()]
        self.sequence = itertools.count()
        self.random = random.Random()
        self.batches = {}
        self.active = [0, 0]
        self.judge_limiter = asyncio.Semaphore(20)

    async def rollout(self, question, lane, name, prefix_file):
        directory = self.directory / 'questions' / str(question['index']) / name
        directory.mkdir(parents=True, exist_ok=True)
        answer_path = directory / 'answer.json'
        if answer_path.exists():
            return json.loads(answer_path.read_text())
        snapshot = None if prefix_file is None else json.loads(Path(prefix_file).read_text())
        for attempt in range(1, ATTEMPTS + 1):
            attempt_dir = directory / f'attempt-{attempt}'
            if attempt_dir.exists():
                error_path = attempt_dir / 'error.json'
                if error_path.exists() and json.loads(error_path.read_text())['retryable']:
                    continue
                archived = directory / f'interrupted-attempt-{attempt}-{uuid.uuid4().hex}'
                attempt_dir.rename(archived)
                print(f'RESUME archived={archived}', flush=True)
            harness = VideoSQLHarness(database=ARTIFACTS / 'index.duckdb', embedding_index=ARTIFACTS / 'embeddings', media_map=ARTIFACTS / 'media-map.json', skill_dir=ROOT / 'training_free/skills-vector', run_dir=attempt_dir, model_name=MODEL, api_key='EMPTY', base_url=str(self.clients[lane].base_url), embedding_url=self.embedding_url, embedding_model='qwen3-vl-embedding-8b', query_time=None, clip_qa_video=True, reasoning_effort=None, semantic_search_mode='vector', client=self.clients[lane], embedding_client=self.embedding_client, model_limiter=self.limiters[lane])
            harness.context.model = RecordedResponsesModel(harness.context)
            print(f'ROLLOUT_START question={question["index"]} trajectory={name} lane={lane} attempt={attempt}', flush=True)
            try:
                if snapshot is None:
                    session = await harness.answer(question['question'])
                else:
                    items = await harness.restore_prefix(snapshot)
                    session = await harness.answer(question['question'], input_items=items, instructions=snapshot['agent']['instructions'], base_instructions=snapshot['base_instructions'], compactions=snapshot['compactions'])
                if session.status != 'answered' or not isinstance(session.answer, str) or not session.answer.strip():
                    raise InvalidTrace(f'Rollout has no final answer: {session}')
            except BaseException as error:
                can_retry = retryable(error)
                record = {'type': type(error).__name__, 'message': str(error), 'retryable': can_retry, 'traceback': traceback.format_exc(), 'chain': [{'type': type(e).__name__, 'message': str(e)} for e in exception_chain(error)]}
                (attempt_dir / 'error.json').write_text(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'ROLLOUT_ERROR question={question["index"]} trajectory={name} attempt={attempt} retryable={can_retry} error={error}', flush=True)
                if not can_retry:
                    raise
                if any(isinstance(e, (openai.APIConnectionError, httpx.ConnectError, httpx2.ConnectError)) for e in exception_chain(error)):
                    response = await self.embedding_client.get(str(self.clients[lane].base_url).removesuffix('/v1/') + '/health', timeout=10)
                    response.raise_for_status()
                continue
            answer = {'question_id': question['index'], 'trajectory': name, 'lane': lane, 'attempt': attempt, 'attempt_dir': str(attempt_dir), 'status': 'answered', 'answer': session.answer, 'prefix_file': prefix_file, 'trace_files': session.trace_files, 'investigation_calls': harness.context.investigation_calls, 'final_calls': harness.context.final_calls}
            answer_path.write_text(json.dumps(answer, ensure_ascii=False) + '\n')
            print(f'ROLLOUT_DONE question={question["index"]} trajectory={name} attempt={attempt}', flush=True)
            return answer
        answer = {'question_id': question['index'], 'trajectory': name, 'lane': lane, 'attempt': ATTEMPTS, 'attempt_dir': str(directory / f'attempt-{ATTEMPTS}'), 'status': 'failed', 'answer': None, 'prefix_file': prefix_file, 'trace_files': sorted(str(p) for p in (directory / f'attempt-{ATTEMPTS}' / 'traces').glob('answer-segment-*.json'))}
        answer_path.write_text(json.dumps(answer, ensure_ascii=False) + '\n')
        return answer

    def select_branches(self, question, answer):
        path = self.directory / 'questions' / str(question['index']) / 'branches.json'
        if path.exists():
            return json.loads(path.read_text())['selected']
        candidates = []
        for trace_file in sorted((Path(answer['attempt_dir']) / 'traces').glob('model-*.json')):
            trace = json.loads(trace_file.read_text())
            response = trace['response']
            if response is None or response['status'] != 'completed':
                continue
            if any(item['type'] == 'function_call' for item in response['output']):
                snapshot = json.loads(Path(trace['prefix_file']).read_text())
                candidates.append({'name': f'branch-{snapshot["answer_turns"]:03d}', 'prefix_file': trace['prefix_file'], 'source_response': response['id'], 'source_trace': str(trace_file)})
        selected = self.random.sample(candidates, min(5, len(candidates)))
        path.write_text(json.dumps({'question_id': question['index'], 'source_attempt': answer['attempt'], 'candidate_count': len(candidates), 'selected': selected}, ensure_ascii=False) + '\n')
        print(f'BRANCHES question={question["index"]} candidates={len(candidates)} selected={[x["name"] for x in selected]}', flush=True)
        return selected

    async def judge(self, question, answer):
        directory = self.directory / 'questions' / str(question['index']) / answer['trajectory']
        result_path = directory / 'result.json'
        if result_path.exists():
            self.update_batch(question)
            return
        if answer['status'] == 'failed':
            result = {**answer, 'score': 0, 'rounded_score': 0.0, 'judge_trace': None}
            result_path.write_text(json.dumps(result, ensure_ascii=False) + '\n')
            self.update_batch(question)
            return
        messages = [{'role': 'system', 'content': JUDGE_PROMPT}, {'role': 'user', 'content': f'Question: {question["question"]}\nGroundtruth answer: {question["answer"]}\nCandidate answer: {answer["answer"]}\nYour response: '}]
        traces = directory / 'judge'
        traces.mkdir(exist_ok=True)
        attempt = len(list(traces.glob('attempt-*.json')))
        async with self.judge_limiter:
            while True:
                attempt += 1
                timeout = 60 * attempt
                path = traces / f'attempt-{attempt:04d}.json'
                record = {'request': {'model': JUDGE_MODEL, 'messages': messages, 'timeout': timeout}, 'response': None, 'error': None}
                path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'JUDGE_START question={question["index"]} trajectory={answer["trajectory"]} attempt={attempt}', flush=True)
                try:
                    raw = await self.judge_client.chat.completions.with_raw_response.create(model=JUDGE_MODEL, messages=messages, timeout=timeout)
                    record['request'] = {**json.loads(raw.http_request.content), 'timeout': timeout}
                    record['response'] = raw.http_response.json()
                    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
                    choice = raw.parse().choices[0]
                    if choice.finish_reason != 'stop' or choice.message.content is None:
                        raise InvalidTrace(f'Judge did not finish with text: {record["response"]}')
                    matches = re.findall(r'^Final Score:\s*([0-5])\s*$', choice.message.content, flags=re.IGNORECASE | re.MULTILINE)
                    if len(matches) != 1:
                        raise InvalidTrace(f'Judge needs one explicit Final Score: {choice.message.content}')
                    score = int(matches[0])
                except BaseException as error:
                    record['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
                    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
                    print(f'JUDGE_ERROR question={question["index"]} trajectory={answer["trajectory"]} attempt={attempt} error={error}', flush=True)
                    if not retryable(error):
                        raise
                    continue
                break
        rounded = 1.0 if score >= 4 else 0.5 if score == 3 else 0.0
        result = {**answer, 'score': score, 'rounded_score': rounded, 'judge_trace': str(path)}
        result_path.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        print(f'JUDGE_DONE question={question["index"]} trajectory={answer["trajectory"]} score={score}', flush=True)
        self.update_batch(question)

    def update_batch(self, question):
        root = self.directory / 'questions' / str(question['index'])
        selected = json.loads((root / 'branches.json').read_text())['selected']
        paths = [root / 'main/result.json', *(root / branch['name'] / 'result.json' for branch in selected)]
        if not all(path.exists() for path in paths):
            return
        results = [json.loads(path.read_text()) for path in paths]
        mean_score = sum(item['rounded_score'] for item in results) / len(results)
        correct = any(item['rounded_score'] == 1 for item in results)
        batch = {'question_id': question['index'], 'question': question['question'], 'reference_answer': question['answer'], 'mean_rounded_score': mean_score, 'at_least_one_correct': correct, 'selected_branches': selected, 'results': results}
        (root / 'batch.json').write_text(json.dumps(batch, ensure_ascii=False) + '\n')
        self.batches[question['index']] = batch
        metrics = {'completed_questions': len(self.batches), 'total_questions': len(self.questions), 'completed_rollouts': sum(len(b['results']) for b in self.batches.values()), 'mean_rounded_score': sum(b['mean_rounded_score'] for b in self.batches.values()) / len(self.batches), 'at_least_one_correct_rate': sum(b['at_least_one_correct'] for b in self.batches.values()) / len(self.batches)}
        (self.directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        self.run.log(metrics)
        print('BATCH_DONE ' + json.dumps({'question_id': question['index'], **metrics}), flush=True)

    async def worker(self, lane, group):
        while True:
            _, _, question_id, name, prefix_file = await self.queues[lane].get()
            try:
                if question_id is None:
                    return
                question = self.questions[question_id]
                self.active[lane] += 1
                self.run.log({f'active/lane_{lane}': self.active[lane]})
                try:
                    while True:
                        answer = await self.rollout(question, lane, name, prefix_file)
                        pending = []
                        if name == 'main':
                            selected = self.select_branches(question, answer)
                            pending = [branch for branch in selected if not (self.directory / 'questions' / str(question_id) / branch['name'] / 'result.json').exists()]
                            for branch in pending[1:]:
                                self.queues[lane].put_nowait((0, next(self.sequence), question_id, branch['name'], branch['prefix_file']))
                        group.create_task(self.judge(question, answer))
                        if not pending:
                            break
                        name, prefix_file = pending[0]['name'], pending[0]['prefix_file']
                        print(f'IMMEDIATE_BRANCH question={question_id} trajectory={name} lane={lane}', flush=True)
                finally:
                    self.active[lane] -= 1
                    self.run.log({f'active/lane_{lane}': self.active[lane]})
            finally:
                self.queues[lane].task_done()


async def main(args):
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    all_questions = json.loads((DATASET / 'week/test.json').read_text())
    questions = [q for q in all_questions if args.ids is None or q['index'] in args.ids]
    if args.ids is not None and set(args.ids) != {q['index'] for q in questions}:
        raise ValueError(f'Unknown Week question IDs: {args.ids}')
    source_rows = json.loads((ARTIFACTS / 'media-map.json').read_text())['sources']
    week_rows = json.loads(WEEK_MEDIA_MAP.read_text())['sources']
    source_names = {Path(row['relative_path']).name: row for row in source_rows}
    week_names = {Path(row['path']).name: row for row in week_rows}
    dataset_names = {path.name for day in (DATASET / 'videos/week').iterdir() for path in day.glob('*.mp4')}
    if source_names.keys() != week_names.keys() or source_names.keys() != dataset_names:
        raise ValueError('Week and EgoLife video inventories differ')
    for row in source_rows:
        if not (ARTIFACTS / row['relative_path']).is_file():
            raise FileNotFoundError(ARTIFACTS / row['relative_path'])
    offsets = {f'day{day}': next(row['timeline_start_s'] for row in week_rows if f'/day{day}/' in row['path']) for day in range(1, 8)}
    intervals = 0
    for question in all_questions:
        local = [(clue['video_id'], start, end) for clue in question['clue_intervals'] for start, end in clue['intervals']]
        for (day, start, end), (total_start, total_end) in zip(local, question['total_intervals'], strict=True):
            if not math.isclose(start + offsets[day], total_start, abs_tol=1e-6) or not math.isclose(end + offsets[day], total_end, abs_tol=1e-6):
                raise ValueError(f'Week timestamp mismatch for question {question["index"]}')
            intervals += 1
    compared = []
    for day in range(1, 8):
        rows = [r for r in source_rows if f'/DAY{day}/' in r['relative_path']]
        row = rows[len(rows) // 2]
        ego_path = ARTIFACTS / row['relative_path']
        week_path = DATASET / 'videos/week' / f'day{day}' / ego_path.name
        with ego_path.open('rb') as ego, week_path.open('rb') as week:
            same = hashlib.file_digest(ego, 'sha256').digest() == hashlib.file_digest(week, 'sha256').digest()
        if not same:
            raise ValueError(f'Week/EgoLife video bytes differ: {ego_path.name}')
        compared.append(ego_path.name)
    audit = {'dataset': str(DATASET / 'week/test.json'), 'week_media_map': str(WEEK_MEDIA_MAP), 'egolife_media_map': str(ARTIFACTS / 'media-map.json'), 'questions': len(all_questions), 'video_files': len(source_names), 'checked_intervals': intervals, 'day_offsets': offsets, 'byte_identical_sample_files': compared}
    (directory / 'data-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('DATA_READY ' + json.dumps(audit), flush=True)
    config = {'dataset': str(DATASET / 'week/test.json'), 'artifact_root': str(ARTIFACTS), 'question_ids': [q['index'] for q in questions], 'base_urls': args.base_urls, 'embedding_url': args.embedding_url, 'model': MODEL, 'judge_model': JUDGE_MODEL, 'lane_concurrency': LANE_CONCURRENCY, 'attempts': ATTEMPTS, 'branches_per_question': 5, 'mean_weighting': 'mean_within_question_then_across_questions', 'query_time': None}
    config_path = directory / 'config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved['config'] != config:
            raise ValueError('Resume settings differ from the saved Week run')
        wandb_id, resume = saved['wandb_id'], 'must'
    else:
        wandb_id, resume = generate_id(), 'never'
        config_path.write_text(json.dumps({'config': config, 'wandb_id': wandb_id}, indent=2) + '\n')
    values = dotenv_values(ROOT / '.env')
    judge_client = AsyncOpenAI(api_key=values['SHARED_OPENAI_KEY'], max_retries=0)
    clients = [AsyncOpenAI(api_key='EMPTY', base_url=url, max_retries=0, timeout=None, http_client=DefaultAsyncHttpxClient(limits=httpx2.Limits(max_connections=1000, max_keepalive_connections=256), event_hooks={'request': [record_wire_request]})) for url in args.base_urls]
    embedding_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=60, read=100000, write=100000, pool=100000), limits=httpx.Limits(max_connections=1000, max_keepalive_connections=256))
    run = None
    try:
        for url, expected in [(args.embedding_url, 'qwen3-vl-embedding-8b'), *((url, MODEL) for url in args.base_urls)]:
            response = await embedding_client.get(url.rstrip('/') + '/models', timeout=10)
            response.raise_for_status()
            if expected not in [item['id'] for item in response.json()['data']]:
                raise RuntimeError(f'Unexpected model at {url}: {response.text}')
        run = wandb.init(entity='3dsmile', project='video-sql-v2', id=wandb_id, name=directory.name, config=config, resume=resume, dir=str(directory))
        collector = WeekCollector(directory, questions, clients, embedding_client, args.embedding_url, judge_client, run)
        lanes = {question['index']: int(position >= len(all_questions) // 2) for position, question in enumerate(all_questions)}
        for question in questions:
            batch_path = directory / 'questions' / str(question['index']) / 'batch.json'
            if batch_path.exists():
                collector.batches[question['index']] = json.loads(batch_path.read_text())
            else:
                collector.queues[lanes[question['index']]].put_nowait((1, next(collector.sequence), question['index'], 'main', None))
        async with asyncio.TaskGroup() as group:
            for lane in range(2):
                for _ in range(LANE_CONCURRENCY):
                    group.create_task(collector.worker(lane, group))
            await asyncio.gather(*(queue.join() for queue in collector.queues))
            for lane in range(2):
                for _ in range(LANE_CONCURRENCY):
                    collector.queues[lane].put_nowait((2, next(collector.sequence), None, None, None))
        if collector.batches.keys() != collector.questions.keys():
            raise RuntimeError('Some Week batches are missing after all workers finished')
        for question in questions:
            collector.update_batch(question)
        summary = json.loads((directory / 'metrics.json').read_text())
        summary['wandb_url'] = run.url
        summary['run_dir'] = str(directory)
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COLLECTION_COMPLETE ' + json.dumps(summary), flush=True)
    finally:
        await asyncio.gather(*(client.close() for client in clients), judge_client.close(), embedding_client.aclose())
        if run is not None:
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/mmlifelong-week/collection')
    parser.add_argument('--base-urls', nargs=2, default=['http://127.0.0.1:8310/v1', 'http://127.0.0.1:8311/v1'])
    parser.add_argument('--embedding-url', default='http://127.0.0.1:8214/v1')
    parser.add_argument('--ids', nargs='+', type=int)
    asyncio.run(main(parser.parse_args()))
