from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml


class DocumentChunkMetadata(BaseModel):
    num_article: int | None = None
    chapitre: str | None = None
    page: int | None = None


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: DocumentChunkMetadata


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)


class Config(BaseSettings):
    data: dict = {}
    qdrant: dict = {}
    embedding: dict = {}
    llm: dict = {}
    retrieval: dict = {}
    server: dict = {}

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
