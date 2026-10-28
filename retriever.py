"""Retrieval logic."""

from typing import List, Tuple

from .embeddings import EmbeddingModel
from .models import Chunk
from .vector_store import SimpleVectorStore


class Retriever:
    """Retrieves relevant chunks for a query."""

    def __init__(self, store: SimpleVectorStore, embedding_model: EmbeddingModel, top_k: int = 4):
        self.store = store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Tuple[Chunk, float]]:
        """Embed the question and return top-k chunks with scores."""
        query_embedding = self.embedding_model.embed(question)
        return self.store.search(query_embedding, top_k=self.top_k)
