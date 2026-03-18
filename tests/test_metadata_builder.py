import unittest

from langchain_core.documents import Document

from src.indexing.metadata.builder import (
    build_chunk_metadata_records,
    build_document_id,
    build_document_info,
)


class MetadataBuilderTests(unittest.TestCase):
    def test_document_id_is_deterministic(self) -> None:
        filename = "abc123_sample.pdf"
        self.assertEqual(build_document_id(filename), build_document_id(filename))

    def test_title_author_from_pdf_metadata_with_fallback(self) -> None:
        from_pdf = build_document_info(
            filename="anything.pdf",
            pdf_metadata={"title": "Introduction to AI", "author": "Ada"},
        )
        self.assertEqual(from_pdf["title"], "Introduction to AI")
        self.assertEqual(from_pdf["author"], "Ada")

        fallback = build_document_info(
            filename="f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4_intro_to_ai.pdf",
            pdf_metadata={},
        )
        self.assertEqual(fallback["title"], "Intro To Ai")
        self.assertEqual(fallback["author"], "Unknown")

    def test_chunk_metadata_shape_and_page_mapping(self) -> None:
        chunks = [
            Document(page_content="chunk one", metadata={"page": 0}),
            Document(page_content="chunk two", metadata={}),
        ]

        records = build_chunk_metadata_records(
            filename="doc.pdf",
            chunks=chunks,
            pdf_metadata={"title": "Doc", "author": "Unknown"},
            embedding_model="nomic-embed-text",
            extracted_at="2026-03-18T10:00:00Z",
        )

        self.assertEqual(records[0]["chunk_id"].endswith("_chunk_001"), True)
        self.assertEqual(records[0]["chunk_index"], 1)
        self.assertEqual(records[0]["page_number"], 1)
        self.assertIsNone(records[1]["page_number"])
        self.assertIn("chunk_text", records[0])


if __name__ == "__main__":
    unittest.main()
