from pathlib import Path

from fastapi import HTTPException

from src.retrieval.qa_chain import answer_question_with_retrieval


def handle_query_request(
    *,
    query: str,
    has_indexed_document,
    persist_directory: Path,
    collection_name: str,
    log_step,
) -> dict:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if not has_indexed_document():
        raise HTTPException(
            status_code=400,
            detail="Please upload first a valid document before asking questions.",
        )

    try:
        result = answer_question_with_retrieval(
            query=query,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log_step(
        "query/result",
        {
            "query": query,
            "used_pdf_context": result["used_pdf_context"],
            "citations": result["citations"],
        },
    )
    return result
