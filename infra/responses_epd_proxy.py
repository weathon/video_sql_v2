import argparse
import asyncio
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


app = FastAPI()
session: aiohttp.ClientSession


def extract_input_images(request_data: dict) -> list[dict]:
    images = []
    pending = [request_data["input"]]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if item.get("type") == "input_image":
                images.append(item)
            else:
                pending.extend(item.values())
    return images


async def prime_image(image: dict, model: str, request_id: str) -> None:
    encoder_request = {
        "model": model,
        "input": [{"role": "user", "content": [image]}],
        "stream": False,
    }
    async with session.post(
        f"{app.state.encoder_url}/v1/responses",
        json=encoder_request,
        headers={"x-request-id": request_id},
    ) as response:
        body = await response.read()
        if response.status >= 400:
            raise HTTPException(
                status_code=response.status,
                detail=body.decode(errors="replace"),
            )


async def stream_response(response: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.content.iter_chunked(65536):
            yield chunk
    finally:
        response.release()
        app.state.active_requests -= 1


@app.on_event("startup")
async def startup() -> None:
    global session
    app.state.cache_lock = asyncio.Lock()
    app.state.active_requests = 0
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=100000, sock_connect=60, sock_read=100000),
        connector=aiohttp.TCPConnector(limit=1000),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    await session.close()


@app.post("/v1/responses")
async def responses(request: Request):
    request_data = await request.json()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    images = extract_input_images(request_data)
    async with app.state.cache_lock:
        app.state.active_requests += 1
    streaming = False
    try:
        try:
            async with asyncio.TaskGroup() as group:
                for index, image in enumerate(images):
                    group.create_task(
                        prime_image(image, request_data["model"], f"{request_id}:{index}")
                    )
        except* HTTPException as errors:
            raise errors.exceptions[0]
        response = await session.post(
            f"{app.state.pd_url}/v1/responses",
            json=request_data,
            headers={"x-request-id": request_id},
        )
        media_type = response.headers["Content-Type"].split(";", 1)[0]
        if request_data.get("stream"):
            streaming = True
            return StreamingResponse(
                stream_response(response),
                status_code=response.status,
                media_type=media_type,
            )
        body = await response.read()
        response.release()
        return Response(content=body, status_code=response.status, media_type=media_type)
    finally:
        if not streaming:
            app.state.active_requests -= 1


def delete_shared_encoder_cache() -> dict:
    before = shutil.disk_usage(app.state.cache_dir).used
    removed = 0
    for path in app.state.cache_dir.iterdir():
        shutil.rmtree(path)
        removed += 1
    after = shutil.disk_usage(app.state.cache_dir).used
    print(f"encoder cache cleanup: entries={removed} used_bytes={before}->{after}", flush=True)
    return {"removed_entries": removed, "used_bytes_before": before, "used_bytes_after": after}


@app.post("/clear_encoder_cache")
async def clear_encoder_cache():
    async with app.state.cache_lock:
        if app.state.active_requests:
            raise HTTPException(status_code=409, detail="encoder cache is in use")
        for url in (app.state.encoder_url, app.state.pd_url):
            async with session.post(f"{url}/reset_encoder_cache") as response:
                body = await response.text()
                print(f"reset_encoder_cache: server={url} status={response.status} body={body}", flush=True)
                response.raise_for_status()
        return await asyncio.to_thread(delete_shared_encoder_cache)


@app.get("/v1/models")
async def models():
    async with session.get(f"{app.state.pd_url}/v1/models") as response:
        return Response(
            content=await response.read(),
            status_code=response.status,
            media_type=response.headers["Content-Type"].split(";", 1)[0],
        )


@app.get("/health")
async def health():
    encoder, pd = await asyncio.gather(
        session.get(f"{app.state.encoder_url}/health"),
        session.get(f"{app.state.pd_url}/health"),
    )
    statuses = {"encoder": encoder.status, "prefill_decode": pd.status}
    encoder.release()
    pd.release()
    status_code = 200 if all(status == 200 for status in statuses.values()) else 503
    return JSONResponse(status_code=status_code, content=statuses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--encoder-url", required=True)
    parser.add_argument("--pd-url", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    app.state.encoder_url = arguments.encoder_url
    app.state.pd_url = arguments.pd_url
    app.state.cache_dir = arguments.cache_dir
    uvicorn.run(app, host=arguments.host, port=arguments.port)
