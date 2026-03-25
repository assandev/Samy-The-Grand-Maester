import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api.main as main


class QueryApiTests(unittest.TestCase):
    def test_query_requires_uploaded_document(self) -> None:
        client = TestClient(main.app)
        with patch.object(main, "_has_indexed_document", return_value=False):
            response = client.post("/query", json={"query": "What is this PDF about?"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Please upload first a valid document", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
