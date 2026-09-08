import io
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.server import app, documents, extracted_facts_db, storage, reconciliation_db
from app.models import Context, Dimension, DocumentData, ExtractedFacts, Fact, PageData, Provenance


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        documents.clear()
        extracted_facts_db.clear()
        reconciliation_db.clear()
        storage.clear_all()

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
        storage.save_document(self.doc)

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

        # Check persistence in SQLite storage directly
        persisted = storage.get_extracted_facts("test_doc")
        self.assertIsNotNone(persisted)
        self.assertEqual(len(persisted.facts), 1)

    @patch("app.server.parse_pdf")
    def test_upload_pdf_endpoint(self, mock_parse_pdf):
        mock_doc = DocumentData(
            doc_id="uploaded_123",
            filename="uploaded_file.pdf",
            page_count=3,
            pages=[],
            raw_text_index={},
        )
        mock_parse_pdf.return_value = mock_doc

        file_content = b"%PDF-1.4 dummy pdf content"
        resp = self.client.post(
            "/upload",
            files={"file": ("uploaded_file.pdf", io.BytesIO(file_content), "application/pdf")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["doc_id"], "uploaded_123")
        self.assertEqual(data["filename"], "uploaded_file.pdf")

        # Verify storage contains it
        self.assertTrue(storage.has_document("uploaded_123"))

    def test_reconcile_and_cases_endpoints(self):
        # Seed doc1
        f1 = Fact(
            subject="Delhivery",
            predicate="revenue",
            value="₹100 Cr",
            numeric_value=100.0,
            canonical_value=1000000000.0,
            canonical_unit="INR",
            confidence=1.0,
            provenance=Provenance(doc_id="doc1", page=1, evidence_quote="revenue ₹100 Cr", verified=True, match_type="exact"),
        )
        ef1 = ExtractedFacts(doc_id="doc1", facts=[f1])
        storage.save_extracted_facts("doc1", ef1)
        extracted_facts_db["doc1"] = ef1
        doc1_meta = DocumentData(doc_id="doc1", filename="doc1.pdf", page_count=1)
        storage.save_document(doc1_meta)
        documents["doc1"] = doc1_meta

        # Seed doc2
        f2 = Fact(
            subject="Delhivery",
            predicate="revenue",
            value="₹100 Cr",
            numeric_value=100.0,
            canonical_value=1000000000.0,
            canonical_unit="INR",
            confidence=1.0,
            provenance=Provenance(doc_id="doc2", page=2, evidence_quote="revenue ₹100 Cr", verified=True, match_type="exact"),
        )
        ef2 = ExtractedFacts(doc_id="doc2", facts=[f2])
        storage.save_extracted_facts("doc2", ef2)
        extracted_facts_db["doc2"] = ef2
        doc2_meta = DocumentData(doc_id="doc2", filename="doc2.pdf", page_count=1)
        storage.save_document(doc2_meta)
        documents["doc2"] = doc2_meta

        # Call /reconcile
        resp = self.client.post("/reconcile")
        self.assertEqual(resp.status_code, 200)
        graph_data = resp.json()
        self.assertEqual(graph_data["total_facts"], 2)
        self.assertEqual(len(graph_data["clusters"]), 1)

        # Call /reconcile/cases
        cases_resp = self.client.get("/reconcile/cases")
        self.assertEqual(cases_resp.status_code, 200)
        cases_data = cases_resp.json()
        self.assertIn("case_1_corroborated", cases_data)
        self.assertIsNotNone(cases_data["case_1_corroborated"])

    def test_reconcile_append_endpoint(self):
        # Initial doc1
        f1 = Fact(
            subject="RBI",
            predicate="repo_rate",
            value="6.5%",
            numeric_value=6.5,
            canonical_value=6.5,
            confidence=1.0,
            provenance=Provenance(doc_id="doc1", page=1, evidence_quote="repo rate 6.5%", verified=True, match_type="exact"),
        )
        ef1 = ExtractedFacts(doc_id="doc1", facts=[f1])
        storage.save_extracted_facts("doc1", ef1)
        extracted_facts_db["doc1"] = ef1
        doc1_meta = DocumentData(doc_id="doc1", filename="rbi.pdf", page_count=1)
        storage.save_document(doc1_meta)
        documents["doc1"] = doc1_meta

        # Initial reconcile
        self.client.post("/reconcile")

        # Now add doc2
        f2 = Fact(
            subject="RBI",
            predicate="repo_rate",
            value="6.5%",
            numeric_value=6.5,
            canonical_value=6.5,
            confidence=1.0,
            provenance=Provenance(doc_id="doc2", page=3, evidence_quote="repo rate maintained at 6.5%", verified=True, match_type="exact"),
        )
        ef2 = ExtractedFacts(doc_id="doc2", facts=[f2])
        storage.save_extracted_facts("doc2", ef2)
        extracted_facts_db["doc2"] = ef2
        doc2_meta = DocumentData(doc_id="doc2", filename="survey.pdf", page_count=1)
        storage.save_document(doc2_meta)
        documents["doc2"] = doc2_meta

        # Call /reconcile/append
        resp = self.client.post("/reconcile/append?doc_id=doc2")
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("graph", result)
        self.assertIn("incremental_stats", result)
        stats = result["incremental_stats"]
        self.assertEqual(stats["new_facts_count"], 1)
        self.assertEqual(result["graph"]["total_facts"], 2)

    def test_persistence_across_memory_clear(self):
        """Simulate memory loss and verify endpoints retrieve from SQLite storage."""
        f1 = Fact(
            subject="TestCo",
            predicate="ebitda",
            value="$50M",
            numeric_value=50.0,
            canonical_value=50000000.0,
            confidence=1.0,
            provenance=Provenance(doc_id="persisted_doc", page=1, evidence_quote="$50M", verified=True, match_type="exact"),
        )
        ef = ExtractedFacts(doc_id="persisted_doc", facts=[f1])
        storage.save_extracted_facts("persisted_doc", ef)

        # Clear in-memory dictionary
        extracted_facts_db.clear()

        # GET /documents/{doc_id}/facts should hydrate from storage
        resp = self.client.get("/documents/persisted_doc/facts")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["doc_id"], "persisted_doc")
        self.assertEqual(data["facts"][0]["subject"], "TestCo")

    def test_reconcile_max_facts_exceeded_returns_413(self):
        f1 = Fact(
            subject="TestCo",
            predicate="revenue",
            value="$100M",
            numeric_value=100.0,
            canonical_value=100000000.0,
            confidence=1.0,
            provenance=Provenance(doc_id="doc1", page=1, evidence_quote="$100M", verified=True, match_type="exact"),
        )
        ef1 = ExtractedFacts(doc_id="doc1", facts=[f1])
        storage.save_extracted_facts("doc1", ef1)
        extracted_facts_db["doc1"] = ef1

        resp = self.client.post("/reconcile?max_facts=0")
        self.assertEqual(resp.status_code, 413)
        self.assertIn("exceeds maximum allowed limit", resp.json()["detail"])

    def test_extract_invalid_page_range_returns_400(self):
        # self.doc has page_count=1
        resp = self.client.post("/documents/test_doc/extract?start_page=5&end_page=10")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid page range", resp.json()["detail"])

    def test_custom_dimension_and_dispute_extensibility(self):
        # Verify Fact accepts open custom dimension string
        custom_fact = Fact(
            subject="Turbine",
            predicate="rpm",
            value="3500 rpm",
            numeric_value=3500.0,
            dimension="rotational_speed",  # custom non-enum dimension
            confidence=1.0,
            provenance=Provenance(doc_id="eng_doc", page=1, evidence_quote="3500 rpm", verified=True, match_type="exact"),
        )
        self.assertEqual(custom_fact.dimension, "rotational_speed")

        # Verify serialization and deserialization
        fact_json = custom_fact.model_dump_json()
        restored = Fact.model_validate_json(fact_json)
        self.assertEqual(restored.dimension, "rotational_speed")

    def test_partially_scanned_pdf_surfaces_warnings_and_scanned_pages(self):
        doc_with_scans = DocumentData(
            doc_id="scanned_doc",
            filename="partial_scan.pdf",
            page_count=3,
            pages=[
                PageData(page_number=0, raw_text="Valid text on page 0"),
                PageData(page_number=1, raw_text=""),  # empty scan
                PageData(page_number=2, raw_text="Valid text on page 2"),
            ],
            raw_text_index={0: "Valid text on page 0", 1: "", 2: "Valid text on page 2"},
            scanned_pages=[1],
            warnings=["1 of 3 pages appear to be scanned images with no extractable text: pages [1]"],
        )
        storage.save_document(doc_with_scans)
        documents["scanned_doc"] = doc_with_scans

        # GET /documents/{id} should preserve scanned_pages and warnings
        retrieved = self.client.get("/documents/scanned_doc").json()
        self.assertEqual(retrieved["scanned_pages"], [1])
        self.assertEqual(len(retrieved["warnings"]), 1)
        self.assertIn("scanned images", retrieved["warnings"][0])

    def test_token_overlap_semantic_fallback_without_voyage_key(self):
        from app.embeddings import is_semantic_match

        # Test token overlap fallback matches related predicates without external API call
        self.assertTrue(is_semantic_match("revenue_from_contracts", "revenue", kind="metric"))
        self.assertTrue(is_semantic_match("total_revenue", "revenue", kind="metric"))
        self.assertTrue(is_semantic_match("Delhivery Limited", "Delhivery", kind="entity"))


if __name__ == "__main__":
    unittest.main()
