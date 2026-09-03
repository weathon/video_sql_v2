import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


ENCODER_URL = os.environ["EPD_ENCODER_URL"].rstrip("/")
PD_URL = os.environ["EPD_PD_URL"].rstrip("/")
app = FastAPI()
encoder_session: aiohttp.ClientSession
pd_session: aiohttp.ClientSession


def collect_images(value: object, images: list[dict[str, object]], seen: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            collect_images(item, images, seen)
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "input_image":
        image_url = value["image_url"]
        if not isinstance(image_url, str):
            raise TypeError("input_image.image_url must be a string")
        if image_url not in seen:
            seen.add(image_url)
            images.append(value)
        return
    for item in value.values():
        collect_images(item, images, seen)


@app.on_event("startup")
async def startup() -> None:
    global encoder_session, pd_session
    timeout = aiohttp.ClientTimeout(total=None)
    encoder_session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0))
    pd_session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0))


@app.on_event("shutdown")
async def shutdown() -> None:
    await encoder_session.close()
    await pd_session.close()


async def prime_image(model: str, image: dict[str, object], request_id: str, headers: dict[str, str]) -> None:
    payload = {"model": model, "input": [{"role": "user", "content": [image]}], "stream": False}
    async with encoder_session.post(f"{ENCODER_URL}/v1/responses", json=payload, headers={**headers, "x-request-id": request_id}) as response:
        response_body = await response.read()
        if response.status < 200 or response.status >= 300:
            print(json.dumps({"event": "epd_encoder_error", "request_id": request_id, "status": response.status, "body": response_body.decode(errors="replace")}))
            raise HTTPException(status_code=502, detail="EPD encoder request failed")


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    body = await request.json()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length", "transfer-encoding", "connection"}}
    images: list[dict[str, object]] = []
    collect_images(body["input"], images, set())
    if images:
        print(json.dumps({"event": "epd_encoder_prime", "request_id": request_id, "images": len(images)}))
        await asyncio.gather(*(prime_image(body["model"], image, f"{request_id}:encoder:{index}", headers) for index, image in enumerate(images)))

    upstream = await pd_session.post(f"{PD_URL}/v1/responses", json=body, headers={**headers, "x-request-id": request_id})
    if upstream.status < 200 or upstream.status >= 300:
        response_body = await upstream.read()
        print(json.dumps({"event": "epd_pd_error", "request_id": request_id, "status": upstream.status, "body": response_body.decode(errors="replace")}))
        upstream.release()
        return Response(content=response_body, status_code=upstream.status, headers={"content-type": upstream.headers.get("content-type", "application/json")})
    if not body.get("stream", False):
        response_body = await upstream.read()
        upstream.release()
        return Response(content=response_body, status_code=upstream.status, headers={"content-type": upstream.headers.get("content-type", "application/json")})

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.content.iter_any():
                yield chunk
        finally:
            upstream.release()

    return StreamingResponse(stream(), status_code=upstream.status, headers={"content-type": upstream.headers.get("content-type", "text/event-stream")})


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request) -> Response:
    headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length", "transfer-encoding", "connection"}}
    async with pd_session.request(request.method, f"{PD_URL}/{path}", data=await request.body(), headers=headers) as upstream:
        response_body = await upstream.read()
        return Response(content=response_body, status_code=upstream.status, headers={"content-type": upstream.headers.get("content-type", "application/json")})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
