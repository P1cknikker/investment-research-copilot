"""PDF document loading."""

from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


class DocumentLoader:
    """Loads text from PDF files, page by page."""

    def load_pdf(self, file_path: str | Path) -> List[Tuple[str, int]]:
        """
        Extract text per page from a PDF.

        Returns:
            List of (text, page_number) tuples. Page numbers are 1-based.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(path)
        pages: List[Tuple[str, int]] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, i + 1))

        return pages
