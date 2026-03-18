import os


def build_ollama_embeddings():
    try:
        from langchain_ollama import OllamaEmbeddings
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain_ollama'. Install it with "
            "'pip install langchain-ollama' in your active environment."
        ) from exc

    return OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
