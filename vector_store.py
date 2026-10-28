"""Simple in-memory vector store."""

from typing import List, Tuple

import numpy as np

from .models import Chunk


class SimpleVectorStore:
    """In-memory vector store using cosine similarity."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []

    def add(self, chunk: Chunk, embedding: List[float]) -> None:
        """Add a chunk and its embedding."""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)

    def add_batch(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add multiple chunks and embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Return the top-k most similar chunks with their cosine similarity scores.
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)
        similarities: List[float] = []

        for emb in self.embeddings:
            emb_vec = np.array(emb)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
            sim = float(np.dot(query_vec, emb_vec) / denom) if denom != 0 else 0.0
            similarities.append(sim)

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.chunks[i], similarities[i]) for i in top_indices]

    def __len__(self) -> int:
        return len(self.chunks)
