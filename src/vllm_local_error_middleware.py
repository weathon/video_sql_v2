from fastapi.responses import JSONResponse


async def expose_runtime_error(request, call_next):
    try:
        return await call_next(request)
    except RuntimeError as error:
        return JSONResponse(status_code=500, content={"exception_type": type(error).__name__, "detail": str(error)})
