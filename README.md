# Naive RAG (Starter Scaffold)

This repository follows a simple RAG architecture with two main flows:

1. Indexing flow (document -> vectors)
2. Query flow (question -> retrieve context -> LLM response)

## Project Structure

- `src/api/`: API or endpoints (upload/query)
- `src/orchestrator/`: Coordinates retrieval + LLM calls
- `src/indexing/`: End-to-end indexing pipeline
- `src/indexing/loaders/`: PDF loading/extraction
- `src/indexing/validators/`: File and content validation
- `src/indexing/chunking/`: Text splitting/chunk strategy
- `src/indexing/metadata/`: Metadata creation/enrichment
- `src/indexing/embeddings/`: Embedding generation
- `src/retrieval/`: Similarity search + ranking
- `src/vectorstore/`: Vector database adapter
- `src/llm/`: LLM client and prompt assembly
- `src/config/`: Settings and environment loading
- `src/common/`: Shared utilities and schemas
- `data/raw/`: Uploaded source files
- `data/processed/`: Intermediate extracted/chunked data
- `data/vectorstore/`: Local vector DB persisted files
- `docs/`: Architecture and design notes
- `tests/`: Unit/integration tests
- `scripts/`: Local helper scripts

## API (Current)

### Setup

```bash
source myenv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
```

### Run API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Ingestion flow

`POST /ingest/pdf` now performs:
- upload
- validation
- text extraction
- chunking
- metadata generation
- embeddings creation with Ollama
- storage in Chroma

Artifacts written:
- `data/processed/<pdf_name>_chunks.txt`
- `data/processed/<pdf_name>_chunks.jsonl`
- `data/vectorstore/`
