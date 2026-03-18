from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader


def extract_pdf_documents(pdf_path: Path) -> list[Document]:
    """Extract page-level documents from a PDF file."""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def extract_pdf_metadata(pdf_path: Path) -> dict[str, str]:
    """Extract PDF metadata fields relevant for indexing."""
    try:
        reader = PdfReader(str(pdf_path))
        raw = reader.metadata or {}
    except Exception:
        return {}

    title = str(raw.get("/Title") or "").strip()
    author = str(raw.get("/Author") or "").strip()
    return {
        "title": title,
        "author": author,
    }
