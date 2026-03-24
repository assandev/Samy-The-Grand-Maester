import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.common.storage import reset_runtime_data
from src.indexing.chunking.text_chunker import chunk_documents
from src.indexing.embeddings.ollama_provider import build_ollama_embeddings
from src.indexing.loaders.pdf_loader import extract_pdf_documents, extract_pdf_metadata
from src.indexing.metadata.builder import build_chunk_metadata_records, persist_metadata_jsonl
from src.indexing.validators.pdf_validator import validate_extracted_documents
from src.retrieval.qa_chain import answer_question_with_retrieval
from src.vectorstore.chroma_store import store_metadata_records_in_chroma

app = FastAPI(title="Naive RAG API", version="0.1.0")


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def _log_step(step: str, payload: dict) -> None:
    print(f"[{step}] {json.dumps(payload, ensure_ascii=False, default=str)}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_PATH", "data/vectorstore"))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "naive_rag_chunks")


class IndexPdfRequest(BaseModel):
    filename: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def reset_single_document_runtime() -> None:
    reset_runtime_data(RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR)
    _log_step(
        "startup/reset",
        {
            "raw_data_dir": str(RAW_DATA_DIR),
            "processed_data_dir": str(PROCESSED_DATA_DIR),
            "vector_db_dir": str(VECTOR_DB_DIR),
        },
    )


def _reset_before_new_document() -> None:
    reset_runtime_data(RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR)
    _log_step(
        "document/reset",
        {
            "raw_data_dir": str(RAW_DATA_DIR),
            "processed_data_dir": str(PROCESSED_DATA_DIR),
            "vector_db_dir": str(VECTOR_DB_DIR),
        },
    )


def _get_chroma_client():
    import chromadb

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DB_DIR))


def _get_vector_collection():
    client = _get_chroma_client()
    return client.get_or_create_collection(name=DEFAULT_VECTOR_COLLECTION)


def _get_existing_vector_collection():
    client = _get_chroma_client()
    return client.get_collection(name=DEFAULT_VECTOR_COLLECTION)


def _has_indexed_document() -> bool:
    try:
        return _get_existing_vector_collection().count() > 0
    except Exception:
        return False


def _build_preview_items(limit: int) -> dict:
    collection = _get_vector_collection()
    payload = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    embeddings = payload.get("embeddings")
    if embeddings is None:
        embeddings = []

    items = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        document = documents[index] if index < len(documents) else ""
        embedding = embeddings[index] if index < len(embeddings) else None
        embedding_preview = None
        if embedding is not None and len(embedding) > 0:
            embedding_preview = [round(float(value), 6) for value in embedding[:8]]

        items.append(
            {
                "id": chunk_id,
                "metadata": metadata,
                "document": document,
                "embedding_preview": embedding_preview,
            }
        )

    items.sort(
        key=lambda item: (item["metadata"].get("extracted_at") or ""),
        reverse=True,
    )
    top_items = items[:limit]

    return {
        "vector_db_path": str(VECTOR_DB_DIR),
        "collection": DEFAULT_VECTOR_COLLECTION,
        "count": collection.count(),
        "limit": limit,
        "items": top_items,
    }


def _save_uploaded_pdf(file: UploadFile, content: bytes) -> tuple[str, Path]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    saved_name = f"{uuid4().hex}_{safe_name}"
    destination = RAW_DATA_DIR / saved_name
    destination.write_bytes(content)
    _log_step(
        "upload/pdf",
        {
            "original_name": file.filename,
            "saved_name": saved_name,
            "bytes": len(content),
            "destination": str(destination),
        },
    )
    return saved_name, destination


def _emit_progress(callback, step: str, payload: dict) -> None:
    if callback:
        callback(step, payload)


def _index_pdf_file(pdf_path: Path, chunk_size: int, chunk_overlap: int, progress_callback=None) -> dict:
    _log_step(
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

    documents = extract_pdf_documents(pdf_path)
    is_valid, validation_message, validation_stats = validate_extracted_documents(documents)
    validation_payload = {
        "file": pdf_path.name,
        "valid": is_valid,
        "message": validation_message,
        "stats": validation_stats,
    }
    _log_step("Validation", validation_payload)
    _emit_progress(progress_callback, "validation", validation_payload)

    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": validation_message,
                "stats": validation_stats,
            },
        )

    extraction_sample = documents[0].page_content[:250] if documents else ""
    extraction_payload = {
        "file": pdf_path.name,
        "pages_loaded": len(documents),
        "sample_text": extraction_sample,
    }
    _log_step("Extracting text", extraction_payload)
    _emit_progress(progress_callback, "extracting", extraction_payload)

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
    _log_step("Creating chunks", chunk_payload)
    _emit_progress(progress_callback, "chunks", chunk_payload)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_chunks.txt"
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, chunk in enumerate(chunks, start=1):
            handle.write(f"# Chunk {idx}\n")
            handle.write(chunk.page_content)
            handle.write("\n\n")

    pdf_metadata = extract_pdf_metadata(pdf_path)
    metadata_records = build_chunk_metadata_records(
        filename=pdf_path.name,
        chunks=chunks,
        pdf_metadata=pdf_metadata,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    metadata_output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_chunks.jsonl"
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
    _log_step("Creating metadata", metadata_payload)
    _emit_progress(progress_callback, "metadata", metadata_payload)

    try:
        embedding_function = build_ollama_embeddings()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        vectors_stored = store_metadata_records_in_chroma(
            records=metadata_records,
            persist_directory=VECTOR_DB_DIR,
            collection_name=DEFAULT_VECTOR_COLLECTION,
            embedding_function=embedding_function,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    vector_preview_payload = _build_preview_items(limit=1)
    vector_sample = vector_preview_payload["items"][0] if vector_preview_payload["items"] else None
    chroma_payload = {
        "file": pdf_path.name,
        "collection": DEFAULT_VECTOR_COLLECTION,
        "vector_db_path": str(VECTOR_DB_DIR),
        "vectors_stored": vectors_stored,
        "sample_embedding_record": vector_sample,
    }
    _log_step("Stored in chroma db", chroma_payload)
    _emit_progress(progress_callback, "chroma", chroma_payload)

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
        "vector_db_path": str(VECTOR_DB_DIR),
        "vector_collection": DEFAULT_VECTOR_COLLECTION,
        "vectors_stored": vectors_stored,
        "vector_sample": vector_sample,
        "validation": {
            "valid": True,
            "message": validation_message,
            "stats": validation_stats,
        },
    }
    _log_step("index/result", result)
    return result


@app.get("/")
def index() -> dict[str, str]:
    return {
        "status": "Backend is running",
        "service": "Samy | The Grand Maester API",
        "frontend": "http://127.0.0.1:5173",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/vectorstore/collections")
def vectorstore_collections() -> dict:
    client = _get_chroma_client()
    collections = client.list_collections()
    items = []
    for collection in collections:
        item = {"name": collection.name}
        try:
            item["count"] = collection.count()
        except Exception:
            item["count"] = None
        items.append(item)

    _log_step("vectorstore/collections", {"collections": items})
    return {
        "vector_db_path": str(VECTOR_DB_DIR),
        "collections": items,
    }


@app.get("/vectorstore/preview")
def vectorstore_preview(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    preview = _build_preview_items(limit=limit)
    _log_step(
        "vectorstore/preview",
        {
            "collection": preview["collection"],
            "count": preview["count"],
            "limit": preview["limit"],
            "returned": len(preview["items"]),
        },
    )
    return preview


@app.post("/query")
def query_pdf(payload: QueryRequest) -> dict:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if not _has_indexed_document():
        raise HTTPException(
            status_code=400,
            detail="Please upload first a valid document before asking questions.",
        )

    try:
        result = answer_question_with_retrieval(
            query=payload.query,
            persist_directory=VECTOR_DB_DIR,
            collection_name=DEFAULT_VECTOR_COLLECTION,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _log_step(
        "query/result",
        {
            "query": payload.query,
            "used_pdf_context": result["used_pdf_context"],
            "citations": result["citations"],
        },
    )
    return result


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Invalid content type; expected PDF")

    _reset_before_new_document()
    content = await file.read()
    saved_name, destination = _save_uploaded_pdf(file, content)

    return {
        "message": "PDF uploaded successfully",
        "filename": saved_name,
        "path": str(destination),
    }


@app.post("/index/pdf")
def index_pdf(payload: IndexPdfRequest) -> dict:
    pdf_path = RAW_DATA_DIR / payload.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {payload.filename}")

    return _index_pdf_file(
        pdf_path=pdf_path,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Invalid content type; expected PDF")

    _log_step(
        "ingest/start",
        {
            "filename": file.filename,
            "content_type": file.content_type or "unknown",
        },
    )
    _reset_before_new_document()
    content = await file.read()
    saved_name, _ = _save_uploaded_pdf(file, content)
    pdf_path = RAW_DATA_DIR / saved_name

    result = _index_pdf_file(
        pdf_path=pdf_path,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    result["ingest"] = {
        "uploaded": True,
        "auto_indexed": True,
    }
    _log_step("ingest/success", {"filename": saved_name})
    return result


@app.post("/ingest/pdf/stream")
async def ingest_pdf_stream(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Invalid content type; expected PDF")

    _reset_before_new_document()
    content = await file.read()
    saved_name, _ = _save_uploaded_pdf(file, content)
    pdf_path = RAW_DATA_DIR / saved_name

    def generate():
        import queue
        import threading

        event_queue: queue.Queue = queue.Queue()
        step_order = ["validation", "extracting", "chunks", "metadata", "chroma"]

        def callback(step: str, payload: dict) -> None:
            event_queue.put({"type": "step_completed", "step": step, "payload": payload})

        def worker() -> None:
            try:
                result = _index_pdf_file(
                    pdf_path=pdf_path,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    progress_callback=callback,
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

    return StreamingResponse(generate(), media_type="application/x-ndjson")
