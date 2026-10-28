"""Text chunking with metadata."""

from typing import List, Tuple

from .models import Chunk


class Chunker:
    """Splits document text into overlapping chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(
        self,
        pages: List[Tuple[str, int]],
        document_name: str,
    ) -> List[Chunk]:
        """
        Split page texts into overlapping chunks.

        Args:
            pages: List of (text, page_number) from DocumentLoader.
            document_name: Name used for metadata.

        Returns:
            List of Chunk objects.
        """
        chunks: List[Chunk] = []
        chunk_index = 0

        for text, page_num in pages:
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                content = text[start:end].strip()
                if content:
                    chunks.append(
                        Chunk(
                            text=content,
                            document_name=document_name,
                            page=page_num,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
                start += self.chunk_size - self.overlap

        return chunks
