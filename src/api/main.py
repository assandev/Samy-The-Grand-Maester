import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.indexing.chunking.text_chunker import chunk_documents
from src.indexing.embeddings.ollama_provider import build_ollama_embeddings
from src.indexing.loaders.pdf_loader import extract_pdf_documents, extract_pdf_metadata
from src.indexing.metadata.builder import build_chunk_metadata_records, persist_metadata_jsonl
from src.indexing.validators.pdf_validator import validate_extracted_documents
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
UI_FILE = BASE_DIR / "static" / "index.html"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "naive_rag_chunks")


class IndexPdfRequest(BaseModel):
    filename: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def _get_chroma_client():
    import chromadb

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DB_DIR))


def _get_vector_collection():
    client = _get_chroma_client()
    return client.get_or_create_collection(name=DEFAULT_VECTOR_COLLECTION)


def _save_uploaded_pdf(file: UploadFile, content: bytes) -> tuple[str, Path]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    saved_name = f"{uuid4().hex}_{safe_name}"
    destination = RAW_DATA_DIR / saved_name
    destination.write_bytes(content)
    print(
        f"[upload/pdf] original_name={file.filename} saved_name={saved_name} "
        f"bytes={len(content)} destination={destination}"
    )
    return saved_name, destination


def _index_pdf_file(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> dict:
    print(
        f"[index/start] file={pdf_path.name} chunk_size={chunk_size} "
        f"chunk_overlap={chunk_overlap}"
    )

    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")

    print(f"[extract/start] file={pdf_path.name} path={pdf_path}")
    documents = extract_pdf_documents(pdf_path)
    print(f"[extract/done] file={pdf_path.name} pages_loaded={len(documents)}")

    print(f"[validate/start] file={pdf_path.name}")
    is_valid, validation_message, validation_stats = validate_extracted_documents(documents)

    print(
        f"[validate/pdf] file={pdf_path.name} valid={is_valid} "
        f"pages={validation_stats['pages_total']} pages_with_text={validation_stats['pages_with_text']} "
        f"chars={validation_stats['total_chars']} alpha_ratio={validation_stats['alpha_ratio']}"
    )

    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": validation_message,
                "stats": validation_stats,
            },
        )

    print(
        f"[chunk/start] file={pdf_path.name} source_pages={len(documents)} "
        f"chunk_size={chunk_size} overlap={chunk_overlap}"
    )
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"[chunk/done] file={pdf_path.name} chunks_created={len(chunks)}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_chunks.txt"
    print(f"[chunks/write/start] file={pdf_path.name} output={output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, chunk in enumerate(chunks, start=1):
            handle.write(f"# Chunk {idx}\n")
            handle.write(chunk.page_content)
            handle.write("\n\n")
    print(f"[chunks/write/done] file={pdf_path.name} output={output_path}")

    print(f"[metadata/extract/start] file={pdf_path.name}")
    pdf_metadata = extract_pdf_metadata(pdf_path)
    print(
        f"[metadata/extract/done] file={pdf_path.name} "
        f"title={pdf_metadata.get('title') or 'None'} author={pdf_metadata.get('author') or 'None'}"
    )

    print(
        f"[metadata/build/start] file={pdf_path.name} embedding_model={DEFAULT_EMBEDDING_MODEL}"
    )
    metadata_records = build_chunk_metadata_records(
        filename=pdf_path.name,
        chunks=chunks,
        pdf_metadata=pdf_metadata,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    print(f"[metadata/build/done] file={pdf_path.name} records_built={len(metadata_records)}")

    metadata_output_path = PROCESSED_DATA_DIR / f"{pdf_path.stem}_chunks.jsonl"
    print(f"[metadata/write/start] file={pdf_path.name} output={metadata_output_path}")
    persist_metadata_jsonl(metadata_output_path, metadata_records)
    print(f"[metadata/write/done] file={pdf_path.name} output={metadata_output_path}")

    print(
        f"[embeddings/start] file={pdf_path.name} model={DEFAULT_EMBEDDING_MODEL} "
        f"records={len(metadata_records)}"
    )
    try:
        embedding_function = build_ollama_embeddings()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    print(
        f"[vectorstore/start] file={pdf_path.name} collection={DEFAULT_VECTOR_COLLECTION} "
        f"persist_directory={VECTOR_DB_DIR}"
    )
    try:
        vectors_stored = store_metadata_records_in_chroma(
            records=metadata_records,
            persist_directory=VECTOR_DB_DIR,
            collection_name=DEFAULT_VECTOR_COLLECTION,
            embedding_function=embedding_function,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    print(
        f"[vectorstore/done] file={pdf_path.name} collection={DEFAULT_VECTOR_COLLECTION} "
        f"vectors_stored={vectors_stored}"
    )

    print(
        f"[index/pdf] file={pdf_path.name} chunks={len(chunks)} "
        f"chunk_size={chunk_size} overlap={chunk_overlap}"
    )
    print(f"[metadata/pdf] file={pdf_path.name} records={len(metadata_records)}")

    metadata_sample = None
    if metadata_records:
        metadata_sample = {k: v for k, v in metadata_records[0].items() if k != "chunk_text"}

    result = {
        "message": "PDF validated, extracted, chunked, embedded, and stored successfully",
        "filename": pdf_path.name,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_path": str(output_path),
        "sample_chunk": chunks[0].page_content[:300] if chunks else "",
        "metadata_output_path": str(metadata_output_path),
        "metadata_records": len(metadata_records),
        "metadata_sample": metadata_sample,
        "vector_db_path": str(VECTOR_DB_DIR),
        "vector_collection": DEFAULT_VECTOR_COLLECTION,
        "vectors_stored": vectors_stored,
        "validation": {
            "valid": True,
            "message": validation_message,
            "stats": validation_stats,
        },
    }
    print(
        f"[index/success] file={pdf_path.name} chunks={len(chunks)} "
        f"metadata_records={len(metadata_records)} vectors_stored={vectors_stored}"
    )
    return result


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(UI_FILE)


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

    print(f"[vectorstore/collections] collections={len(items)}")
    return {
        "vector_db_path": str(VECTOR_DB_DIR),
        "collections": items,
    }


@app.get("/vectorstore/preview")
def vectorstore_preview(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    collection = _get_vector_collection()
    payload = collection.get(limit=limit, include=["documents", "metadatas"])

    items = []
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    for index, chunk_id in enumerate(ids):
        items.append(
            {
                "id": chunk_id,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "document": documents[index] if index < len(documents) else "",
            }
        )

    print(f"[vectorstore/preview] collection={DEFAULT_VECTOR_COLLECTION} items={len(items)}")
    return {
        "vector_db_path": str(VECTOR_DB_DIR),
        "collection": DEFAULT_VECTOR_COLLECTION,
        "count": collection.count(),
        "items": items,
    }


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Invalid content type; expected PDF")

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

    print(
        f"[ingest/start] filename={file.filename} content_type={file.content_type or 'unknown'}"
    )
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
    print(f"[ingest/success] filename={saved_name}")
    return result
