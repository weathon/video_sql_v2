import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src import trace2skill_baseline as baseline


@pytest.mark.parametrize("phase", ["collection", "month"])
def test_main_agents_keep_slots_while_qa_runs_independently(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(baseline, "RUN", tmp_path)
    questions = [{"timeline": "day", "index": i} for i in range(4 if phase == "collection" else 20)]
    if phase == "collection":
        for question in questions:
            path = tmp_path / "collection/day" / f"question-{question['index']}" / "tree.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"nodes": [{"index": i, "parent": None if i == 0 else 0, "expanded": True} for i in range(5)]}))

    async def scenario():
        slots = baseline.MainAgentSlots(Mock())
        metrics = Mock()
        events = []
        started = []
        qa_started = []
        main_waiting = []
        all_qa_started = asyncio.Event()
        release_qa = asyncio.Event()
        all_qa_returned = asyncio.Event()
        release_main = asyncio.Event()

        async def qa_response(kind):
            qa_started.append(kind)
            events.append({"event": "qa_started", "kind": kind, "main_active": slots.active})
            if len(qa_started) == 135:
                all_qa_started.set()
            await release_qa.wait()
            return kind, {"agent": kind}

        context = SimpleNamespace(client=Mock(), model_settings=SimpleNamespace(), clip_response=qa_response, summary_response=qa_response, embedding_client=SimpleNamespace(post=AsyncMock(return_value="embedding")))
        monkeypatch.setattr(baseline, "VideoSQLHarness", Mock(return_value=SimpleNamespace(context=context)))
        harness = baseline.make_harness(tmp_path, tmp_path, metrics, "day")

        async def rollout(question, node, phase, skill_version, metrics):
            identity = [question["index"], node["index"]]
            started.append(identity)
            events.append({"event": "main_started", "identity": identity, "main_active": slots.active})
            responses = await asyncio.gather(*[
                call(kind) for _ in range(3) for call, kind in (
                    (harness.context.clip_response, "clip-qa"),
                    (harness.context.summary_response, "sql-qa"),
                    (harness.context.summary_response, "semantic-qa"),
                )
            ])
            assert [raw for raw, recorded in responses] == ["clip-qa", "sql-qa", "semantic-qa"] * 3
            main_waiting.append(identity)
            if len(main_waiting) == 15:
                all_qa_returned.set()
            await release_main.wait()
            events.append({"event": "main_answered", "identity": identity, "main_active": slots.active})
            return {"timeline": question["timeline"], "question": str(question["index"]), "rollout": node["index"]}

        monkeypatch.setattr(baseline, "rollout", rollout)
        monkeypatch.setattr(baseline, "score_rollout", AsyncMock(side_effect=lambda answer, metrics: answer))
        work = baseline.collect_trees(questions, slots, metrics) if phase == "collection" else baseline.evaluate_month(questions, "initial", slots, metrics)
        task = asyncio.create_task(work)
        await asyncio.wait_for(all_qa_started.wait(), timeout=5)
        assert len(started) == slots.active == slots.peak == 15
        assert len(qa_started) == 135
        assert await asyncio.wait_for(harness.context.embedding_client.post("query"), timeout=5) == "embedding"
        release_qa.set()
        await asyncio.wait_for(all_qa_returned.wait(), timeout=5)
        assert len(started) == slots.active == 15
        assert not task.done()
        release_main.set()
        results = await asyncio.wait_for(task, timeout=5)
        assert len(results) == len(started) == 20
        assert len(qa_started) == metrics.record_response.call_count == 180
        assert slots.active == 0
        assert slots.peak == 15
        assert len({tuple(identity) for identity in started}) == 20
        events.append({"event": "complete", "main_active": slots.active, "main_peak": slots.peak, "answers": len(results), "qa_calls": len(qa_started)})
        (tmp_path / "concurrency-events.json").write_text(json.dumps(events, indent=2) + "\n")

    asyncio.run(scenario())


def test_cancelled_main_agent_and_waiter_release_slots():
    async def scenario():
        slots = baseline.MainAgentSlots(Mock())
        holding = asyncio.Event()
        release = asyncio.Event()
        for _ in range(14):
            await slots.__aenter__()

        async def main_agent():
            async with slots:
                holding.set()
                await release.wait()

        active = asyncio.create_task(main_agent())
        await asyncio.wait_for(holding.wait(), timeout=5)
        waiter = asyncio.create_task(main_agent())
        await asyncio.sleep(0)
        assert slots.active == 15
        assert len(slots.waiters) == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert slots.waiters == []
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert slots.active == 14
        async with slots:
            assert slots.active == 15
        for _ in range(14):
            await slots.__aexit__(None, None, None)
        assert slots.active == 0

    asyncio.run(scenario())
