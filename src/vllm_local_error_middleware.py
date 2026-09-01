from starlette.requests import Request
from starlette.responses import JSONResponse


async def expose_runtime_error(request: Request, call_next):
    try:
        return await call_next(request)
    except RuntimeError as error:
        return JSONResponse(status_code=500, content={"exception_type": "RuntimeError", "detail": str(error)})
