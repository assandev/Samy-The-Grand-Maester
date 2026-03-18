import re

from langchain_core.documents import Document


def validate_extracted_documents(
    documents: list[Document],
    min_total_chars: int = 120,
    min_alpha_ratio: float = 0.2,
) -> tuple[bool, str, dict[str, int | float]]:
    """Validate whether extracted PDF text is likely usable for indexing."""
    pages_total = len(documents)
    pages_with_text = sum(1 for doc in documents if doc.page_content.strip())

    full_text = "\n".join(doc.page_content for doc in documents).strip()
    total_chars = len(full_text)
    alpha_chars = len(re.findall(r"[A-Za-z]", full_text))
    alpha_ratio = (alpha_chars / total_chars) if total_chars else 0.0

    stats = {
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "total_chars": total_chars,
        "alpha_ratio": round(alpha_ratio, 4),
    }

    if pages_total == 0:
        return False, "No pages were extracted from the PDF.", stats

    if pages_with_text == 0 or total_chars < min_total_chars:
        return (
            False,
            "This PDF has little or no extractable text. It might be scanned handwriting or low-quality OCR.",
            stats,
        )

    if alpha_ratio < min_alpha_ratio:
        return (
            False,
            "Extracted content looks too noisy/non-readable for reliable indexing.",
            stats,
        )

    return True, "PDF validation passed.", stats
