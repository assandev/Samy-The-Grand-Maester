import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.vectorstore.chroma_store import store_metadata_records_in_chroma


class ChromaStoreTests(unittest.TestCase):
    def test_store_metadata_records_in_chroma_writes_records(self) -> None:
        fake_module = types.ModuleType("langchain_chroma")
        vector_store = MagicMock()
        fake_module.Chroma = MagicMock(return_value=vector_store)

        original = sys.modules.get("langchain_chroma")
        sys.modules["langchain_chroma"] = fake_module
        try:
            count = store_metadata_records_in_chroma(
                records=[
                    {
                        "chunk_id": "doc_123_chunk_001",
                        "chunk_text": "hello world",
                        "title": "Test",
                    }
                ],
                persist_directory=Path("/tmp/chroma-test"),
                collection_name="naive_rag_chunks",
                embedding_function=MagicMock(),
            )
        finally:
            if original is None:
                del sys.modules["langchain_chroma"]
            else:
                sys.modules["langchain_chroma"] = original

        self.assertEqual(count, 1)
        vector_store.delete.assert_called_once_with(ids=["doc_123_chunk_001"])
        vector_store.add_texts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
