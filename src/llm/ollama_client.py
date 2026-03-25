import os


def build_ollama_llm(*, temperature: float = 0.2):
    try:
        from langchain_ollama import OllamaLLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain_ollama'. Install it with "
            "'pip install langchain-ollama' in your active environment."
        ) from exc

    return OllamaLLM(
        model=os.getenv("OPENAI_MODEL", os.getenv("OLLAMA_LLM_MODEL", "mistral:latest")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )
