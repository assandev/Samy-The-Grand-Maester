from pathlib import Path


def store_metadata_records_in_chroma(
    *,
    records: list[dict],
    persist_directory: Path,
    collection_name: str,
    embedding_function,
) -> int:
    if not records:
        return 0

    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain_chroma'. Install it with "
            "'pip install langchain-chroma chromadb' in your active environment."
        ) from exc

    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )

    ids = [record["chunk_id"] for record in records]
    texts = [record["chunk_text"] for record in records]
    metadatas = [{k: v for k, v in record.items() if k != "chunk_text"} for record in records]

    vector_store.delete(ids=ids)
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(records)
