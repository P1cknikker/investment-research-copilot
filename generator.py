"""LLM generation with context and 'I don't know' handling."""

import os
from typing import List, Tuple

from openai import OpenAI

from .models import Answer, Chunk, Source


class Generator:
    """Generates structured answers from retrieved context using an LLM."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        min_similarity: float = 0.25,
        temperature: float = 0.2,
    ):
        self.model = model
        self.min_similarity = min_similarity
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def has_relevant_context(self, retrieved: List[Tuple[Chunk, float]]) -> bool:
        """Check whether the best retrieved chunk meets the similarity threshold."""
        if not retrieved:
            return False
        return retrieved[0][1] >= self.min_similarity

    def _build_prompt(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        context_parts = []
        for chunk, score in retrieved:
            page_info = f", Page: {chunk.page}" if chunk.page else ""
            context_parts.append(
                f"[Document: {chunk.document_name}{page_info}, "
                f"Chunk: {chunk.chunk_index}, Score: {score:.2f}]\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        return (
            "Answer the question using ONLY the context below.\n"
            "If the context does not contain enough information to answer the question, "
            "set knows_answer to false and explain why in the reasoning.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
        )

    def generate(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> Answer:
        """Generate a validated Answer from the question and retrieved chunks."""
        if not self.has_relevant_context(retrieved):
            return Answer(
                answer=(
                    "I don't know. The available documents do not contain "
                    "sufficient information to answer this question."
                ),
                confidence=0.0,
                reasoning="No retrieved chunk exceeded the relevance threshold.",
                sources=[],
                knows_answer=False,
            )

        prompt = self._build_prompt(question, retrieved)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You always respond only with valid JSON according to the provided schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        **Answer.model_json_schema(),
                        "additionalProperties": False,
                    },
                },
            },
            temperature=self.temperature,
        )

        return Answer.model_validate_json(response.choices[0].message.content)
