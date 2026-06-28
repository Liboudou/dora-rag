from pydantic import BaseModel
from typing import List


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    embedding: List[float]


class QueryRequest(BaseModel):
    query: str
    top_k: int


class IngestRequest(BaseModel):
    pdf_path: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
