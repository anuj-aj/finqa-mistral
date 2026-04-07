from fastapi import Request
from fastapi.responses import JSONResponse


class ModelNotLoadedException(Exception):
    pass


class InferenceException(Exception):
    pass


class InvalidInputException(Exception):
    pass


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    return JSONResponse(
        status_code=503,
        content={"error": "MODEL_NOT_LOADED", "detail": str(exc)}
    )


async def inference_exception_handler(request: Request, exc: InferenceException):
    return JSONResponse(
        status_code=500,
        content={"error": "INFERENCE_FAILED", "detail": str(exc)}
    )


async def invalid_input_handler(request: Request, exc: InvalidInputException):
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT", "detail": str(exc)}
    )