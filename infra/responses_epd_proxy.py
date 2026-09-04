import argparse
import asyncio
import uuid
from collections.abc import AsyncIterator

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


@app.on_event("startup")
async def startup() -> None:
    global session
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=100_000),
        connector=aiohttp.TCPConnector(limit=0),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    await session.close()


@app.post("/v1/responses")
async def responses(request: Request):
    request_data = await request.json()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    images = extract_input_images(request_data)
    await asyncio.gather(
        *(
            prime_image(image, request_data["model"], f"{request_id}:{index}")
            for index, image in enumerate(images)
        )
    )
    response = await session.post(
        f"{app.state.pd_url}/v1/responses",
        json=request_data,
        headers={"x-request-id": request_id},
    )
    media_type = response.headers["Content-Type"].split(";", 1)[0]
    if request_data.get("stream"):
        return StreamingResponse(
            stream_response(response),
            status_code=response.status,
            media_type=media_type,
        )
    body = await response.read()
    response.release()
    return Response(content=body, status_code=response.status, media_type=media_type)


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
    arguments = parser.parse_args()
    app.state.encoder_url = arguments.encoder_url
    app.state.pd_url = arguments.pd_url
    uvicorn.run(app, host=arguments.host, port=arguments.port)
