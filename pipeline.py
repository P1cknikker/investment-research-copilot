"""End-to-end RAG pipeline."""

from pathlib import Path
from typing import Dict, List
import urllib.request

from .chunker import Chunker
from .document_loader import DocumentLoader
from .embeddings import EmbeddingModel
from .generator import Generator
from .models import Answer
from .retriever import Retriever
from .vector_store import SimpleVectorStore


class RAGPipeline:
    """Orchestrates document ingestion and question answering."""

    # Public documents used for the prototype
    DOCUMENTS: List[Dict[str, str]] = [
        {
            "name": "BlackRock_2024_Annual_Report.pdf",
            "url": "https://www.sec.gov/Archives/edgar/data/2012383/000119312525073384/d865271dars.pdf",
        },
        {
            "name": "Vanguard_TCFD_Report_2024.pdf",
            "url": "https://fund-docs.vanguard.com/tcfd-climate-change-report.pdf",
        },
        {
            "name": "BlackRock_Stewardship_Report_2024.pdf",
            "url": "https://www.blackrock.com/corporate/literature/publication/annual-stewardship-report-2024.pdf",
        },
        {
            "name": "BlackRock_Sustainability_Disclosure_2024.pdf",
            "url": "https://blackrock.com/corporate/literature/continuous-disclosure-and-important-information/blackrock-2022-sasb-disclosure.pdf",
        },
        {
            "name": "BlackRock_2025_Annual_Report.pdf",
            "url": "https://www.sec.gov/Archives/edgar/data/2012383/000130817926000264/blk015204-arsa.pdf",
        },
    ]

    def __init__(
        self,
        docs_dir: str | Path = "documents",
        chunk_size: int = 800,
        overlap: int = 100,
        top_k: int = 4,
        min_similarity: float = 0.25,
    ):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DocumentLoader()
        self.chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_model = EmbeddingModel()
        self.store = SimpleVectorStore()
        self.retriever = Retriever(self.store, self.embedding_model, top_k=top_k)
        self.generator = Generator(min_similarity=min_similarity)

    def _download_if_needed(self, url: str, dest: Path) -> None:
        if dest.exists():
            return
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)

    def ingest_documents(self) -> None:
        """Download, load, chunk, embed and store all configured documents."""
        for doc in self.DOCUMENTS:
            pdf_path = self.docs_dir / doc["name"]
            self._download_if_needed(doc["url"], pdf_path)

            print(f"Processing {doc['name']} ...")
            pages = self.loader.load_pdf(pdf_path)
            chunks = self.chunker.chunk_pages(pages, document_name=doc["name"])

            embeddings = self.embedding_model.embed_batch([c.text for c in chunks])
            self.store.add_batch(chunks, embeddings)

            print(f"  → {len(chunks)} chunks added")

        print(f"\nTotal chunks in store: {len(self.store)}")

    def ask(self, question: str) -> Answer:
        """Retrieve relevant chunks and generate a structured answer."""
        retrieved = self.retriever.retrieve(question)
        return self.generator.generate(question, retrieved)
