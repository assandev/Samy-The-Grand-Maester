# Samy | The Grand Maester

An educational PDF retrieval assistant that indexes documents locally and answers questions with citations, page jumps, and guided inspection of the full RAG pipeline.

## Problem

Technical PDFs are dense, slow to navigate, and easy to forget. When you are learning from manuals, research papers, or setup guides, the hard part is not only finding an answer, but also understanding where it came from and how the retrieval pipeline works under the hood.

This project is built for students, builders, and curious engineers who want a local-first assistant that can:

- ingest a PDF
- explain what happened at every indexing step
- answer questions with citations
- help the user inspect the source material page by page

## Solution

Samy, the Grand Maester is a local full-stack RAG application powered by FastAPI, React, Chroma, and Ollama.

- **PDF Ingestion Pipeline**: Upload a PDF and run validation, text extraction, chunking, metadata generation, and vector storage in sequence.
- **Grounded Question Answering**: Ask questions against the indexed document and receive answers with citations.
- **Educational Step Inspector**: Click any completed pipeline card to inspect what that step produced with readable previews instead of raw logs.
- **Citation-Aware Viewer**: Citation pills jump the PDF viewer to the cited page and show the supporting passage in a dedicated highlight panel.
- **Local-First Architecture**: Everything runs on your machine using local embeddings, local vector storage, and a local Ollama model.

## Demo

[![Samy Demo](https://img.youtube.com/vi/CeO8IfL8Xbw/maxresdefault.jpg)](https://www.youtube.com/watch?v=CeO8IfL8Xbw)

### Example Questions

| Question | Expected Behavior | Output Style |
|---|---|---|
| *"What is this PDF about?"* | Summarize the indexed document | Grounded answer with citations |
| *"What are the steps to download an Ollama model in Windows?"* | Combine PDF support with practical guidance | Structured steps + citations |
| *"Which page mentions chunk overlap?"* | Retrieve a specific document detail | Short answer + citation page jump |

## Results

- **Indexing pipeline**: validation → extraction → chunking → metadata → Chroma storage
- **Cost**: $0 cloud cost — runs locally with Ollama
- **Frontend**: React + Vite + Tailwind dashboard with viewer and citation navigation
- **Data persistence**: local files in `data/raw`, `data/processed`, and `data/vectorstore`
- **API contract**: simple document ingestion and query endpoints for a single-document workflow

## Architecture

```text
┌─────────────────────┐        HTTP        ┌─────────────────────────┐
│                     │  ─────────────────▶│                         │
│   React Frontend    │                    │    FastAPI Backend      │
│   (Vite + Tailwind) │  ◀──────────────── │                         │
│   :5173             │                    │    :8000                │
└──────────┬──────────┘                    └──────────┬──────────────┘
           │                                          │
           │ local PDF viewer + citations             │
           │                                          │
           │                              embeddings + retrieval
           │                                          │
           ▼                                          ▼
┌─────────────────────┐                    ┌─────────────────────────┐
│   Uploaded PDF      │                    │    Chroma Vector DB     │
│   in browser state  │                    │    data/vectorstore     │
└─────────────────────┘                    └──────────┬──────────────┘
                                                      │
                                                      │ local model calls
                                                      ▼
                                            ┌─────────────────────┐
                                            │   Ollama            │
                                            │   Local LLM +       │
                                            │   Embeddings        │
                                            └─────────────────────┘
```

**Flow**: PDF upload → FastAPI ingestion stream → validation/extraction/chunking/metadata/vector storage → React status dashboard → user query → Chroma retrieval → Ollama answer generation → citations → viewer jump + highlight panel.

## Tech Stack

- **AI / Retrieval**: Ollama, LangChain, LangChain Chroma
- **Backend**: Python 3.12, FastAPI, Uvicorn, Pydantic, PyPDF
- **Frontend**: React 19, Vite 5, Tailwind CSS 3
- **Vector Store**: ChromaDB
- **Deployment**: local development

## Quickstart

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.com/) installed and running

### 1. Pull the local embedding model

```bash
ollama pull nomic-embed-text
```

If you want to use a specific local chat model for answering, make sure that model is also available in Ollama.

### 2. Start the backend

From the project root:

```bash
source myenv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and you should see a simple backend-running message.

### 3. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Repo Structure

```text
naive_rag/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app, ingestion/query endpoints
│   │   └── static/index.html       # Legacy static page (unused by intended UI)
│   ├── common/
│   │   └── storage.py              # Runtime cleanup/reset helpers
│   ├── indexing/
│   │   ├── chunking/text_chunker.py
│   │   ├── embeddings/ollama_provider.py
│   │   ├── loaders/pdf_loader.py
│   │   ├── metadata/builder.py
│   │   └── validators/pdf_validator.py
│   ├── llm/
│   │   ├── ollama_client.py        # Local LLM wrapper
│   │   └── prompts.py              # Prompt strategy for grounded answers
│   ├── retrieval/
│   │   └── qa_chain.py             # Retrieval + citation assembly
│   └── vectorstore/
│       └── chroma_store.py         # Chroma write path
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main dashboard, chat, viewer, citations
│   │   ├── index.css               # Global styles
│   │   └── main.jsx                # React entry point
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── raw/                        # Uploaded PDFs
│   ├── processed/                  # Chunk and metadata artifacts
│   └── vectorstore/                # Chroma persistence
├── tests/                          # Unit/integration tests
├── requirements.txt
└── README.md
```

**Where to look**:
- API and ingestion/query flow → `src/api/main.py`
- Prompting and answer style → `src/llm/prompts.py`
- Retrieval and citations → `src/retrieval/qa_chain.py`
- Main UI and viewer behavior → `frontend/src/App.jsx`

## Tradeoffs / Learnings

- **Single-document runtime**: The app currently resets runtime data on startup and on new document upload. That keeps the educational workflow simple, but it is not yet designed for multi-document history.
- **Local-first over cloud-first**: Running Ollama and Chroma locally gives privacy and zero API cost, but answer quality and latency depend heavily on the selected model and machine resources.
- **Educational UI over minimal API output**: The frontend intentionally exposes pipeline steps, previews, and citations to teach the user what the system is doing, rather than hiding the retrieval mechanics.
- **Native browser PDF viewer**: Using an iframe with a local object URL makes page navigation easy, but exact text highlighting inside the rendered PDF is limited compared to a full PDF.js renderer.
- **Prompt-controlled synthesis**: Better answers required moving beyond strict context regurgitation and toward prompts that explicitly combine PDF evidence with labeled practical guidance.

## Roadmap

- PDF.js viewer for true in-document text highlighting
- Multi-document indexing and session history
- Better citation grouping and answer-side rationale
- Document-side search within the viewer
- Model selection UI for local Ollama answer models
