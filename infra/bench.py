"""
concurrency + long-context benchmark client for an openai-compatible server
(vllm or sglang). streams responses to measure ttft, measures end-to-end latency,
and aggregates prefill/decode throughput across a fixed concurrency level.
no client-side max_tokens is set (per project rule): the model generates to eos;
prompts ask for a brief answer so decode stays short and the stress is on
high-concurrency long-context prefill.
"""
import argparse, asyncio, json, time, statistics
import aiohttp


async def one_request(session, url, model, messages, sem, results, idx):
    payload = {"model": model, "messages": messages, "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0.0}
    async with sem:
        t0 = time.perf_counter()
        ttft = None
        prompt_tok = comp_tok = 0
        ok = True
        err = ""
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    ok = False
                    err = f"http {resp.status}: {(await resp.text())[:200]}"
                else:
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                        if obj.get("usage"):
                            prompt_tok = obj["usage"]["prompt_tokens"]
                            comp_tok = obj["usage"]["completion_tokens"]
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {str(e)[:200]}"
        lat = time.perf_counter() - t0
    results.append({"idx": idx, "ok": ok, "err": err, "ttft": ttft, "lat": lat,
                    "prompt_tok": prompt_tok, "comp_tok": comp_tok})


async def run_level(convs, url, model, concurrency, n_requests):
    sem = asyncio.Semaphore(concurrency)
    results = []
    timeout = aiohttp.ClientTimeout(total=1800)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(
            one_request(session, url, model, convs[i % len(convs)]["messages"], sem, results, i))
            for i in range(n_requests)]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall, concurrency):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    lats = [r["lat"] for r in ok]
    sum_prompt = sum(r["prompt_tok"] for r in ok)
    sum_comp = sum(r["comp_tok"] for r in ok)

    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(p / 100 * len(s)))]

    return {
        "concurrency": concurrency,
        "n_ok": len(ok), "n_fail": len(fail),
        "wall_s": round(wall, 2),
        "req_throughput_per_s": round(len(ok) / wall, 3) if wall else None,
        "prompt_tok_total": sum_prompt, "comp_tok_total": sum_comp,
        "prefill_tok_per_s": round(sum_prompt / wall, 1) if wall else None,
        "decode_tok_per_s": round(sum_comp / wall, 1) if wall else None,
        "total_tok_per_s": round((sum_prompt + sum_comp) / wall, 1) if wall else None,
        "ttft_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
        "ttft_p50": round(pct(ttfts, 50), 2) if ttfts else None,
        "ttft_p99": round(pct(ttfts, 99), 2) if ttfts else None,
        "lat_mean": round(statistics.mean(lats), 2) if lats else None,
        "lat_p50": round(pct(lats, 50), 2) if lats else None,
        "lat_p99": round(pct(lats, 99), 2) if lats else None,
        "mean_prompt_tok_per_req": round(sum_prompt / len(ok)) if ok else None,
        "mean_comp_tok_per_req": round(sum_comp / len(ok)) if ok else None,
        "errors_sample": [r["err"] for r in fail[:5]],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--n-requests", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    convs = [json.loads(l) for l in open(args.convs)]
    all_summaries = []
    for c in args.concurrency:
        print(f"=== concurrency {c}, {args.n_requests} requests ===", flush=True)
        results, wall = await run_level(convs, args.url, args.model, c, args.n_requests)
        s = summarize(results, wall, c)
        all_summaries.append(s)
        print(json.dumps(s, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"wrote summaries to {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
