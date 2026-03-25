import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retrieval.qa_chain import answer_question_with_retrieval


class RetrievalQaTests(unittest.TestCase):
    def test_returns_fallback_when_no_source_documents(self) -> None:
        fake_chroma_module = types.ModuleType("langchain_chroma")
        retriever = MagicMock()
        retriever.invoke.return_value = []
        vector_store = MagicMock()
        vector_store.as_retriever.return_value = retriever
        fake_chroma_module.Chroma = MagicMock(return_value=vector_store)

        original_chroma = sys.modules.get("langchain_chroma")
        sys.modules["langchain_chroma"] = fake_chroma_module
        try:
            with patch("src.retrieval.qa_chain.build_ollama_llm") as build_llm:
                llm = MagicMock()
                llm.invoke.return_value = "I did not find relevant information in the uploaded PDF. General knowledge: ..."
                build_llm.return_value = llm
                result = answer_question_with_retrieval(
                    query="What is AI?",
                    persist_directory=Path("/tmp/vectorstore"),
                    collection_name="naive_rag_chunks",
                )
        finally:
            if original_chroma is None:
                del sys.modules["langchain_chroma"]
            else:
                sys.modules["langchain_chroma"] = original_chroma

        self.assertFalse(result["used_pdf_context"])
        self.assertEqual(result["citations"], [])

    def test_returns_citations_sorted_by_page_with_excerpt(self) -> None:
        fake_chroma_module = types.ModuleType("langchain_chroma")
        retriever = MagicMock()

        doc_page_six = MagicMock()
        doc_page_six.page_content = "Page six explanation about installing and running the model locally."
        doc_page_six.metadata = {
            "chunk_id": "chunk-6",
            "filename": "guide.pdf",
            "page_number": 6,
            "title": "Guide",
        }

        doc_page_two = MagicMock()
        doc_page_two.page_content = "Page two contains the setup instructions for Windows users."
        doc_page_two.metadata = {
            "chunk_id": "chunk-2",
            "filename": "guide.pdf",
            "page_number": 2,
            "title": "Guide",
        }

        retriever.invoke.return_value = [doc_page_six, doc_page_two]
        vector_store = MagicMock()
        vector_store.as_retriever.return_value = retriever
        fake_chroma_module.Chroma = MagicMock(return_value=vector_store)

        original_chroma = sys.modules.get("langchain_chroma")
        sys.modules["langchain_chroma"] = fake_chroma_module
        try:
            with patch("src.retrieval.qa_chain.build_ollama_llm") as build_llm:
                llm = MagicMock()
                llm.invoke.return_value = "Structured answer."
                build_llm.return_value = llm
                result = answer_question_with_retrieval(
                    query="How do I install the model?",
                    persist_directory=Path("/tmp/vectorstore"),
                    collection_name="naive_rag_chunks",
                )
        finally:
            if original_chroma is None:
                del sys.modules["langchain_chroma"]
            else:
                sys.modules["langchain_chroma"] = original_chroma

        self.assertTrue(result["used_pdf_context"])
        self.assertEqual([citation["page_number"] for citation in result["citations"]], [2, 6])
        self.assertIn("excerpt", result["citations"][0])
        self.assertIn("Windows users", result["citations"][0]["excerpt"])


if __name__ == "__main__":
    unittest.main()
