import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.server import app, documents, extracted_facts_db
from app.models import Context, DocumentData, ExtractedFacts, Fact, PageData, Provenance


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        documents.clear()
        extracted_facts_db.clear()

        # Seed sample document
        self.doc = DocumentData(
            doc_id="test_doc",
            filename="report.pdf",
            page_count=1,
            pages=[
                PageData(
                    page_number=0,
                    raw_text="Net income was $19.69 billion for the third quarter.",
                    text_blocks=[],
                    tables=[],
                )
            ],
            raw_text_index={0: "Net income was $19.69 billion for the third quarter."},
        )
        documents[self.doc.doc_id] = self.doc

    def test_extract_missing_document_returns_404(self):
        resp = self.client.post("/documents/nonexistent/extract")
        self.assertEqual(resp.status_code, 404)

    def test_get_facts_before_extraction_returns_404(self):
        resp = self.client.get("/documents/test_doc/facts")
        self.assertEqual(resp.status_code, 404)

    @patch("app.server.GeminiFactExtractor")
    def test_extract_facts_success(self, MockExtractor):
        mock_instance = MagicMock()
        MockExtractor.return_value = mock_instance

        mock_instance.extract_and_verify.return_value = ExtractedFacts(
            doc_id="test_doc",
            page_number=0,
            facts=[
                Fact(
                    subject="Alphabet",
                    predicate="net_income",
                    value="$19.69 billion",
                    numeric_value=19690000000.0,
                    unit="USD",
                    context=Context(temporal="Q3", scope="Consolidated"),
                    confidence=1.00,
                    provenance=Provenance(
                        doc_id="test_doc",
                        page=0,
                        evidence_quote="Net income was $19.69 billion for the third quarter.",
                        verified=True,
                        match_type="exact",
                    ),
                )
            ],
            model_used="gemini-3.8-flash",
            fallback_attempts=0,
        )

        resp = self.client.post("/documents/test_doc/extract?page=0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["doc_id"], "test_doc")
        self.assertEqual(data["model_used"], "gemini-3.8-flash")
        self.assertEqual(len(data["facts"]), 1)
        self.assertEqual(data["facts"][0]["subject"], "Alphabet")
        self.assertEqual(data["facts"][0]["confidence"], 1.00)
        self.assertTrue(data["facts"][0]["provenance"]["verified"])

        # Check retrieval endpoint
        get_resp = self.client.get("/documents/test_doc/facts")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["facts"][0]["value"], "$19.69 billion")


if __name__ == "__main__":
    unittest.main()
