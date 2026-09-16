import argparse
import asyncio
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages, construct_tool_dicts


app = FastAPI()
session: aiohttp.ClientSession


def extract_input_images(request_data: dict, api: str) -> list[dict]:
    images = []
    pending = [request_data["input"] if api == "responses" else request_data["messages"]]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if item.get("type") in ("input_image", "image_url", "video_url"):
                images.append(item)
            else:
                pending.extend(item.values())
    return images


async def prime_image(image: dict, request_data: dict, request_id: str, api: str) -> None:
    encoder_request = {
        "model": request_data["model"],
        "input" if api == "responses" else "messages": [{"role": "user", "content": [image]}],
        "stream": False,
    }
    for key in ("media_io_kwargs", "mm_processor_kwargs"):
        if key in request_data:
            encoder_request[key] = request_data[key]
    async with session.post(
        f"{app.state.encoder_url}/v1/{api}",
        json=encoder_request,
        headers={"x-request-id": request_id},
    ) as response:
        body = await response.read()
        if response.status >= 400:
            print(f"encoder request={request_id} api={api} media={image['type']} status={response.status} error={body.decode(errors='replace')}", flush=True)
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
        connector=aiohttp.TCPConnector(limit=1000, force_close=True),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    await session.close()


@app.post("/v1/responses")
@app.post("/v1/chat/completions")
async def responses(request: Request):
    request_data = await request.json()
    api = request.url.path.removeprefix("/v1/")
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    images = extract_input_images(request_data, api)
    async with app.state.cache_lock:
        app.state.active_requests += 1
    streaming = False
    try:
        try:
            async with asyncio.TaskGroup() as group:
                for index, image in enumerate(images):
                    group.create_task(
                        prime_image(image, request_data, f"{request_id}:{index}", api)
                    )
        except* HTTPException as errors:
            raise errors.exceptions[0]
        response = await session.post(
            f"{app.state.pd_url}/v1/{api}",
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


@app.post("/v1/responses/input_tokens")
async def input_tokens(request: Request):
    native = ResponsesRequest.model_validate(await request.json())
    messages = jsonable_encoder(construct_input_messages(request_instructions=native.instructions, request_input=native.input))
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        content = []
        for part in message["content"]:
            if part["type"] in ("input_text", "output_text"):
                part = {"type": "text", "text": part["text"]}
            elif part["type"] == "input_image":
                image_url = {"url": part["image_url"]}
                if "detail" in part:
                    image_url["detail"] = part["detail"]
                part = {"type": "image_url", "image_url": image_url}
            content.append(part)
        message["content"] = content
    chat_kwargs = native.build_chat_params(None, "auto").chat_template_kwargs
    payload = {
        "model": native.model, "messages": messages,
        "tools": construct_tool_dicts(native.tools, native.tool_choice),
        "chat_template_kwargs": chat_kwargs,
        "add_generation_prompt": chat_kwargs["add_generation_prompt"],
        "continue_final_message": chat_kwargs["continue_final_message"],
    }
    # vLLM applies the same model tokenizer, template and multimodal processor.
    # /tokenize skips the encoder cache and runs no model generation.
    async with session.post(f"{app.state.pd_url}/tokenize", json=jsonable_encoder(payload)) as response:
        body = await response.json()
        if response.status >= 400:
            raise HTTPException(status_code=response.status, detail=body)
    return {"object": "response.input_tokens", "input_tokens": body["count"]}


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
