from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.model import load_model
from app.routers import predict
from app.config import settings
from app.exceptions import (
    ModelNotLoadedException, model_not_loaded_handler,
    InferenceException, inference_exception_handler,
    InvalidInputException, invalid_input_handler
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer = load_model()
    yield
    del app.state.model
    del app.state.tokenizer

app = FastAPI(
    title="FinQA Mistral API",
    description="Financial QA using fine-tuned Mistral 7B with LoRA",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(predict.router)

app.add_exception_handler(ModelNotLoadedException, model_not_loaded_handler)
app.add_exception_handler(InferenceException, inference_exception_handler)
app.add_exception_handler(InvalidInputException, invalid_input_handler)

@app.get("/health")
def health():
    return {"status": "ok"}