import json
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from src.indexing.chunking.text_chunker import chunk_documents
from src.indexing.embeddings.ollama_provider import build_ollama_embeddings
from src.indexing.loaders.pdf_loader import extract_pdf_documents, extract_pdf_metadata
from src.indexing.metadata.builder import build_chunk_metadata_records, persist_metadata_jsonl
from src.indexing.validators.pdf_validator import validate_extracted_documents
from src.vectorstore.chroma_store import store_metadata_records_in_chroma

VALID_PDF_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


def validate_pdf_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    if file.content_type not in VALID_PDF_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content type; expected PDF")


def save_uploaded_pdf(
    *,
    file: UploadFile,
    content: bytes,
    raw_data_dir: Path,
    log_step,
) -> tuple[str, Path]:
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    saved_name = f"{uuid4().hex}_{safe_name}"
    destination = raw_data_dir / saved_name
    destination.write_bytes(content)
    log_step(
        "upload/pdf",
        {
            "original_name": file.filename,
            "saved_name": saved_name,
            "bytes": len(content),
            "destination": str(destination),
        },
    )
    return saved_name, destination


def emit_progress(callback, step: str, payload: dict) -> None:
    if callback:
        callback(step, payload)


def index_pdf_file(
    *,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    processed_data_dir: Path,
    vector_db_dir: Path,
    embedding_model: str,
    vector_collection: str,
    build_preview_items,
    log_step,
    progress_callback=None,
    extract_pdf_documents_fn=extract_pdf_documents,
    validate_extracted_documents_fn=validate_extracted_documents,
    extract_pdf_metadata_fn=extract_pdf_metadata,
    build_ollama_embeddings_fn=build_ollama_embeddings,
    store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma,
) -> dict:
    log_step(
        "index/start",
        {
            "file": pdf_path.name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")

    documents = extract_pdf_documents_fn(pdf_path)
    extraction_sample = documents[0].page_content[:250] if documents else ""
    extraction_payload = {
        "file": pdf_path.name,
        "pages_loaded": len(documents),
        "sample_text": extraction_sample,
    }
    log_step("Extracting text", extraction_payload)
    emit_progress(progress_callback, "extracting", extraction_payload)

    is_valid, validation_message, validation_stats = validate_extracted_documents_fn(documents)
    validation_payload = {
        "file": pdf_path.name,
        "valid": is_valid,
        "message": validation_message,
        "stats": validation_stats,
    }
    log_step("Validation", validation_payload)
    emit_progress(progress_callback, "validation", validation_payload)

    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": validation_message,
                "stats": validation_stats,
            },
        )

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    sample_chunk = chunks[0].page_content[:300] if chunks else ""
    chunk_payload = {
        "file": pdf_path.name,
        "chunks_created": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "sample_chunk": sample_chunk,
    }
    log_step("Creating chunks", chunk_payload)
    emit_progress(progress_callback, "chunks", chunk_payload)

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / f"{pdf_path.stem}_chunks.txt"
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, chunk in enumerate(chunks, start=1):
            handle.write(f"# Chunk {idx}\n")
            handle.write(chunk.page_content)
            handle.write("\n\n")

    pdf_metadata = extract_pdf_metadata_fn(pdf_path)
    metadata_records = build_chunk_metadata_records(
        filename=pdf_path.name,
        chunks=chunks,
        pdf_metadata=pdf_metadata,
        embedding_model=embedding_model,
    )
    metadata_output_path = processed_data_dir / f"{pdf_path.stem}_chunks.jsonl"
    persist_metadata_jsonl(metadata_output_path, metadata_records)

    metadata_sample = None
    if metadata_records:
        metadata_sample = {k: v for k, v in metadata_records[0].items() if k != "chunk_text"}

    metadata_payload = {
        "file": pdf_path.name,
        "records_built": len(metadata_records),
        "metadata_output_path": str(metadata_output_path),
        "sample_metadata": metadata_sample,
    }
    log_step("Creating metadata", metadata_payload)
    emit_progress(progress_callback, "metadata", metadata_payload)

    try:
        embedding_function = build_ollama_embeddings_fn()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        vectors_stored = store_metadata_records_in_chroma_fn(
            records=metadata_records,
            persist_directory=vector_db_dir,
            collection_name=vector_collection,
            embedding_function=embedding_function,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    vector_preview_payload = build_preview_items(limit=1)
    vector_sample = vector_preview_payload["items"][0] if vector_preview_payload["items"] else None
    chroma_payload = {
        "file": pdf_path.name,
        "collection": vector_collection,
        "vector_db_path": str(vector_db_dir),
        "vectors_stored": vectors_stored,
        "sample_embedding_record": vector_sample,
    }
    log_step("Stored in chroma db", chroma_payload)
    emit_progress(progress_callback, "chroma", chroma_payload)

    result = {
        "message": "PDF validated, extracted, chunked, embedded, and stored successfully",
        "filename": pdf_path.name,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_path": str(output_path),
        "sample_chunk": sample_chunk,
        "metadata_output_path": str(metadata_output_path),
        "metadata_records": len(metadata_records),
        "metadata_sample": metadata_sample,
        "vector_db_path": str(vector_db_dir),
        "vector_collection": vector_collection,
        "vectors_stored": vectors_stored,
        "vector_sample": vector_sample,
        "validation": {
            "valid": True,
            "message": validation_message,
            "stats": validation_stats,
        },
    }
    log_step("index/result", result)
    return result


def build_ingest_stream(
    *,
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    processed_data_dir: Path,
    vector_db_dir: Path,
    embedding_model: str,
    vector_collection: str,
    build_preview_items,
    log_step,
    extract_pdf_documents_fn=extract_pdf_documents,
    validate_extracted_documents_fn=validate_extracted_documents,
    extract_pdf_metadata_fn=extract_pdf_metadata,
    build_ollama_embeddings_fn=build_ollama_embeddings,
    store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma,
):
    def generate():
        import queue
        import threading

        event_queue: queue.Queue = queue.Queue()
        step_order = ["validation", "extracting", "chunks", "metadata", "chroma"]

        def callback(step: str, payload: dict) -> None:
            event_queue.put({"type": "step_completed", "step": step, "payload": payload})

        def worker() -> None:
            try:
                result = index_pdf_file(
                    pdf_path=pdf_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    processed_data_dir=processed_data_dir,
                    vector_db_dir=vector_db_dir,
                    embedding_model=embedding_model,
                    vector_collection=vector_collection,
                    build_preview_items=build_preview_items,
                    log_step=log_step,
                    progress_callback=callback,
                    extract_pdf_documents_fn=extract_pdf_documents_fn,
                    validate_extracted_documents_fn=validate_extracted_documents_fn,
                    extract_pdf_metadata_fn=extract_pdf_metadata_fn,
                    build_ollama_embeddings_fn=build_ollama_embeddings_fn,
                    store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma_fn,
                )
                result["ingest"] = {"uploaded": True, "auto_indexed": True}
                event_queue.put({"type": "result", "payload": result})
            except HTTPException as exc:
                event_queue.put({"type": "error", "status_code": exc.status_code, "detail": exc.detail})
            except Exception as exc:
                event_queue.put({"type": "error", "status_code": 500, "detail": str(exc)})
            finally:
                event_queue.put({"type": "done"})

        threading.Thread(target=worker, daemon=True).start()

        yield json.dumps({"type": "step_started", "step": step_order[0]}, ensure_ascii=False) + "\n"

        while True:
            event = event_queue.get()
            event_type = event.get("type")

            if event_type == "done":
                break

            yield json.dumps(event, ensure_ascii=False) + "\n"

            if event_type == "step_completed":
                step = event["step"]
                next_index = step_order.index(step) + 1
                if next_index < len(step_order):
                    next_step = step_order[next_index]
                    yield json.dumps({"type": "step_started", "step": next_step}, ensure_ascii=False) + "\n"

    return generate
