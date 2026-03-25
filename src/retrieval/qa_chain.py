from pathlib import Path

from src.indexing.embeddings.ollama_provider import build_ollama_embeddings
from src.llm.ollama_client import build_ollama_llm
from src.llm.prompts import build_pdf_qa_prompt


def _build_context_from_documents(source_documents: list) -> str:
    sections = []

    for index, doc in enumerate(source_documents, start=1):
        metadata = dict(doc.metadata)
        title = metadata.get("title") or metadata.get("filename") or f"Chunk {index}"
        page_number = metadata.get("page_number")
        page_label = f"Page {page_number}" if page_number is not None else "Page unknown"
        chunk_text = (doc.page_content or "").strip()
        if not chunk_text:
            continue

        sections.append(f"[Source {index}] {title} | {page_label}\n{chunk_text}")

    return "\n\n".join(sections).strip()


def _build_excerpt(text: str, limit: int = 280) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def answer_question_with_retrieval(*, query: str, persist_directory: Path, collection_name: str) -> dict:
    try:
        from langchain_chroma import Chroma
    except Exception as exc:
        raise RuntimeError(
            "Chroma retrieval dependencies are not installed correctly. Run 'pip install -r requirements.txt' in your active environment."
        ) from exc

    try:
        embedding_function = build_ollama_embeddings()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
        create_collection_if_not_exists=False,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    source_documents = retriever.invoke(query)

    llm = build_ollama_llm()

    if not source_documents:
        fallback_prompt = build_pdf_qa_prompt(question=query, context="")
        fallback_answer = llm.invoke(fallback_prompt)
        return {
            "answer": fallback_answer,
            "used_pdf_context": False,
            "source_documents": [],
            "citations": [],
        }

    context = _build_context_from_documents(source_documents)
    answer = llm.invoke(build_pdf_qa_prompt(question=query, context=context))

    raw_sources = source_documents
    indexed_citations = []
    for index, doc in enumerate(raw_sources):
        metadata = dict(doc.metadata)
        indexed_citations.append(
            {
                "_order": index,
                "chunk_id": metadata.get("chunk_id"),
                "filename": metadata.get("filename"),
                "page_number": metadata.get("page_number"),
                "title": metadata.get("title"),
                "excerpt": _build_excerpt(doc.page_content),
            }
        )

    indexed_citations.sort(
        key=lambda citation: (
            citation["page_number"] is None,
            citation["page_number"] if citation["page_number"] is not None else float("inf"),
            citation["_order"],
        )
    )
    citations = [{key: value for key, value in citation.items() if key != "_order"} for citation in indexed_citations]

    return {
        "answer": answer,
        "used_pdf_context": True,
        "source_documents": [
            {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for doc in raw_sources
        ],
        "citations": citations,
    }
