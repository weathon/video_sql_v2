import json

from starlette.responses import Response


async def expose_runtime_error(request, call_next):
    try:
        return await call_next(request)
    except Exception as error:
        print(json.dumps({"event": "vllm_request_error", "error": str(error)}))
        return Response(
            content=json.dumps({"error": str(error)}),
            status_code=500,
            media_type="application/json",
        )
