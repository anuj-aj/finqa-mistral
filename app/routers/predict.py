from fastapi import APIRouter, Request
from app.model import predict
from app.schemas import QueryRequest, QueryResponse
from app.exceptions import ModelNotLoadedException, InferenceException, InvalidInputException

router = APIRouter(prefix="/api/v1", tags=["inference"])

@router.post("/predict", response_model=QueryResponse)
def predict_answer(req: QueryRequest, request: Request):
    if not hasattr(request.app.state, "model"):
        raise ModelNotLoadedException("Model is not loaded yet")
    
    if not req.context.strip() or not req.question.strip():
        raise InvalidInputException("Context and question cannot be empty")
    
    try:
        answer = predict(
            request.app.state.model,
            request.app.state.tokenizer,
            req.context,
            req.question
        )
    except Exception as e:
        raise InferenceException(f"Model inference failed: {str(e)}")
    
    return QueryResponse(
        answer=answer,
        question=req.question,
        context=req.context
    )