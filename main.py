"""Entry point for the RAG prototype."""

from .pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    # Ingest documents (downloads PDFs on first run)
    pipeline.ingest_documents()

    # Example question
    question = "What was BlackRock's total AUM at the end of 2024?"
    print(f"\nQuestion: {question}")

    result = pipeline.ask(question)

    print("\n=== Structured Answer ===")
    print(f"Knows answer: {result.knows_answer}")
    print(f"Answer:       {result.answer}")
    print(f"Confidence:   {result.confidence:.2f}")
    print(f"Reasoning:    {result.reasoning}")
    print("Sources:")
    for src in result.sources:
        page = f", p.{src.page}" if src.page else ""
        print(f"  - {src.document_name}{page} (chunk {src.chunk_index}, score {src.relevance_score:.2f})")


if __name__ == "__main__":
    main()
