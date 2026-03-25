import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.common.storage import reset_runtime_data
from src.indexing.embeddings.ollama_provider import build_ollama_embeddings
from src.indexing.loaders.pdf_loader import extract_pdf_documents, extract_pdf_metadata
from src.indexing.validators.pdf_validator import validate_extracted_documents
from src.orchestrator.ingest import (
    build_ingest_stream,
    index_pdf_file,
    save_uploaded_pdf,
    validate_pdf_upload,
)
from src.orchestrator.query import handle_query_request
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


def _has_indexed_document() -> bool:
    try:
        return _get_vector_collection().count() > 0
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
    return handle_query_request(
        query=payload.query,
        has_indexed_document=_has_indexed_document,
        persist_directory=VECTOR_DB_DIR,
        collection_name=DEFAULT_VECTOR_COLLECTION,
        log_step=_log_step,
    )


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    validate_pdf_upload(file)

    _reset_before_new_document()
    content = await file.read()
    saved_name, destination = save_uploaded_pdf(
        file=file,
        content=content,
        raw_data_dir=RAW_DATA_DIR,
        log_step=_log_step,
    )

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

    return index_pdf_file(
        pdf_path=pdf_path,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        processed_data_dir=PROCESSED_DATA_DIR,
        vector_db_dir=VECTOR_DB_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        vector_collection=DEFAULT_VECTOR_COLLECTION,
        build_preview_items=_build_preview_items,
        log_step=_log_step,
        extract_pdf_documents_fn=extract_pdf_documents,
        validate_extracted_documents_fn=validate_extracted_documents,
        extract_pdf_metadata_fn=extract_pdf_metadata,
        build_ollama_embeddings_fn=build_ollama_embeddings,
        store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma,
    )


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    validate_pdf_upload(file)

    _log_step(
        "ingest/start",
        {
            "filename": file.filename,
            "content_type": file.content_type or "unknown",
        },
    )
    _reset_before_new_document()
    content = await file.read()
    saved_name, _ = save_uploaded_pdf(
        file=file,
        content=content,
        raw_data_dir=RAW_DATA_DIR,
        log_step=_log_step,
    )
    pdf_path = RAW_DATA_DIR / saved_name

    result = index_pdf_file(
        pdf_path=pdf_path,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        processed_data_dir=PROCESSED_DATA_DIR,
        vector_db_dir=VECTOR_DB_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        vector_collection=DEFAULT_VECTOR_COLLECTION,
        build_preview_items=_build_preview_items,
        log_step=_log_step,
        extract_pdf_documents_fn=extract_pdf_documents,
        validate_extracted_documents_fn=validate_extracted_documents,
        extract_pdf_metadata_fn=extract_pdf_metadata,
        build_ollama_embeddings_fn=build_ollama_embeddings,
        store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma,
    )
    result["ingest"] = {
        "uploaded": True,
        "auto_indexed": True,
    }
    _log_step("ingest/success", {"filename": saved_name})
    return result


@app.post("/ingest/pdf/stream")
async def ingest_pdf_stream(file: UploadFile = File(...)):
    validate_pdf_upload(file)

    _reset_before_new_document()
    content = await file.read()
    saved_name, _ = save_uploaded_pdf(
        file=file,
        content=content,
        raw_data_dir=RAW_DATA_DIR,
        log_step=_log_step,
    )
    pdf_path = RAW_DATA_DIR / saved_name

    generate = build_ingest_stream(
        pdf_path=pdf_path,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        processed_data_dir=PROCESSED_DATA_DIR,
        vector_db_dir=VECTOR_DB_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        vector_collection=DEFAULT_VECTOR_COLLECTION,
        build_preview_items=_build_preview_items,
        log_step=_log_step,
        extract_pdf_documents_fn=extract_pdf_documents,
        validate_extracted_documents_fn=validate_extracted_documents,
        extract_pdf_metadata_fn=extract_pdf_metadata,
        build_ollama_embeddings_fn=build_ollama_embeddings,
        store_metadata_records_in_chroma_fn=store_metadata_records_in_chroma,
    )
    return StreamingResponse(generate(), media_type="application/x-ndjson")
