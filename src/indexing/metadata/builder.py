import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


def build_document_id(filename: str) -> str:
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def build_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    clean_stem = _strip_uuid_prefix(stem)
    clean_stem = re.sub(r"[_-]+", " ", clean_stem)
    clean_stem = re.sub(r"\s+", " ", clean_stem).strip()
    if not clean_stem:
        return "Untitled"
    return clean_stem.title()


def build_document_info(filename: str, pdf_metadata: dict) -> dict[str, str]:
    title = str(pdf_metadata.get("title") or "").strip()
    author = str(pdf_metadata.get("author") or "").strip()

    if not title:
        title = build_title_from_filename(filename)

    if not author:
        author = "Unknown"

    return {
        "title": title,
        "author": author,
    }


def build_chunk_metadata_records(
    *,
    filename: str,
    chunks: list[Document],
    pdf_metadata: dict,
    embedding_model: str,
    extracted_at: str | None = None,
) -> list[dict]:
    document_id = build_document_id(filename)
    document_info = build_document_info(filename=filename, pdf_metadata=pdf_metadata)
    ts = extracted_at or _utc_now_iso()

    records: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_index = idx
        page_number = _extract_page_number(chunk)

        record = {
            "document_id": document_id,
            "filename": filename,
            "title": document_info["title"],
            "author": document_info["author"],
            "chunk_id": f"{document_id}_chunk_{chunk_index:03d}",
            "chunk_index": chunk_index,
            "page_number": page_number,
            "source_type": "pdf",
            "embedding_model": embedding_model,
            "extracted_at": ts,
            "chunk_text": chunk.page_content,
        }
        records.append(record)

    return records


def persist_metadata_jsonl(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _strip_uuid_prefix(value: str) -> str:
    if re.match(r"^[0-9a-fA-F]{32}_.+", value):
        return value.split("_", maxsplit=1)[1]
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_page_number(chunk: Document) -> int | None:
    page = chunk.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    return None
