# Architecture Notes

## Query-time (Naive RAG)

1. User sends query
2. Orchestrator retrieves relevant chunks from vector store
3. Orchestrator sends prompt + retrieved context to LLM
4. LLM returns generated response

## Indexing-time

1. Upload PDF
2. Validate PDF
3. Extract text (e.g., LangChain loaders)
4. Create chunks
5. Create metadata per chunk
6. Generate embeddings
7. Store vectors in vector DB
