import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.main as main


class IndexPdfIntegrationTests(unittest.TestCase):
    def test_index_pdf_writes_metadata_jsonl_and_stores_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            processed_dir = tmp_path / "processed"
            vector_dir = tmp_path / "vectorstore"
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            vector_dir.mkdir(parents=True, exist_ok=True)

            pdf_name = "test_doc.pdf"
            (raw_dir / pdf_name).write_bytes(b"%PDF-1.4 mock")

            fake_docs = [Document(page_content="hello world on page one", metadata={"page": 0})]

            with patch.object(main, "RAW_DATA_DIR", raw_dir), patch.object(
                main, "PROCESSED_DATA_DIR", processed_dir
            ), patch.object(main, "VECTOR_DB_DIR", vector_dir), patch.object(
                main, "extract_pdf_documents", return_value=fake_docs
            ), patch.object(
                main,
                "validate_extracted_documents",
                return_value=(True, "PDF validation passed.", {"pages_total": 1, "pages_with_text": 1, "total_chars": 23, "alpha_ratio": 0.8}),
            ), patch.object(
                main, "extract_pdf_metadata", return_value={"title": "Test Doc", "author": "Unknown"}
            ), patch.object(
                main, "build_ollama_embeddings", return_value=MagicMock()
            ), patch.object(
                main, "store_metadata_records_in_chroma", return_value=1
            ) as mock_store:
                client = TestClient(main.app)
                response = client.post(
                    "/index/pdf",
                    json={
                        "filename": pdf_name,
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()

            self.assertEqual(body["metadata_records"], body["chunks_created"])
            self.assertEqual(body["vectors_stored"], body["chunks_created"])
            self.assertEqual(body["vector_db_path"], str(vector_dir))
            self.assertEqual(body["vector_collection"], main.DEFAULT_VECTOR_COLLECTION)

            metadata_path = Path(body["metadata_output_path"])
            self.assertTrue(metadata_path.exists())

            lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), body["metadata_records"])

            first = json.loads(lines[0])
            for key in [
                "document_id",
                "filename",
                "title",
                "author",
                "chunk_id",
                "chunk_index",
                "page_number",
                "source_type",
                "embedding_model",
                "extracted_at",
                "chunk_text",
            ]:
                self.assertIn(key, first)

            mock_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
