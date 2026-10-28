"""Embedding interface (placeholder implementation)."""

from typing import List

import numpy as np


class EmbeddingModel:
    """
    Embedding model interface.

    Current implementation is a deterministic placeholder.
    Replace `embed` with a real embedding provider (API or local model).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for the given text."""
        # Placeholder: deterministic pseudo-random vector based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimension).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]
