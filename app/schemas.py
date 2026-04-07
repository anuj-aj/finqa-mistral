from pydantic import BaseModel

class QueryRequest(BaseModel):
    context: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    question: str
    context: str