"""Data models for the RAG prototype."""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    document_name: str
    page: Optional[int]
    chunk_index: int


class Source(BaseModel):
    """Source citation returned with an answer."""
    document_name: str
    page: Optional[int] = None
    chunk_index: int
    relevance_score: float


class Answer(BaseModel):
    """Structured answer returned by the model."""
    answer: str = Field(description="The answer based only on the provided context")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    reasoning: str = Field(description="Short reasoning")
    sources: list[Source] = Field(description="Sources used for the answer")
    knows_answer: bool = Field(description="False if the context does not contain enough information")
