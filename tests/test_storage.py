"""Unit tests for SQLite persistent storage layer."""

import tempfile
import unittest
from pathlib import Path

from app.models import (
    CaseType,
    ClaimEdge,
    ClaimGraph,
    Context,
    ContradictionReport,
    Dimension,
    DocumentData,
    EdgeType,
    EvidenceEntry,
    ExtractedFacts,
    Fact,
    FactCluster,
    PageData,
    Provenance,
)
from app.storage import SQLiteStorage


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_facts.db"
        self.storage = SQLiteStorage(self.db_path)

    def tearDown(self):
        self.storage.close()
        self.temp_dir.cleanup()

    def test_wal_mode_enabled(self):
        conn = self.storage._get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_save_and_get_document(self):
        doc = DocumentData(
            doc_id="doc_test_1",
            filename="prospectus.pdf",
            page_count=2,
            pages=[
                PageData(page_number=0, raw_text="Page 0 text content"),
                PageData(page_number=1, raw_text="Page 1 text content"),
            ],
            raw_text_index={0: "Page 0 text content", 1: "Page 1 text content"},
        )

        self.storage.save_document(doc)
        self.assertTrue(self.storage.has_document("doc_test_1"))

        retrieved = self.storage.get_document("doc_test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.doc_id, "doc_test_1")
        self.assertEqual(retrieved.filename, "prospectus.pdf")
        self.assertEqual(retrieved.page_count, 2)
        self.assertEqual(len(retrieved.pages), 2)
        self.assertEqual(retrieved.raw_text_index[1], "Page 1 text content")

        docs_list = self.storage.list_documents()
        self.assertEqual(len(docs_list), 1)
        self.assertEqual(docs_list[0]["doc_id"], "doc_test_1")

    def test_save_and_get_extracted_facts(self):
        facts = [
            Fact(
                subject="Delhivery",
                predicate="revenue",
                value="₹4,810.5 million",
                numeric_value=4810.5,
                base_unit="INR",
                scale=6,
                dimension=Dimension.MONETARY,
                canonical_value=4810500000.0,
                canonical_unit="INR",
                context=Context(temporal="FY2021", scope="Consolidated"),
                confidence=1.0,
                provenance=Provenance(
                    doc_id="doc_test_1",
                    page=16,
                    evidence_quote="revenue of ₹4,810.5 million",
                    verified=True,
                    match_type="exact",
                ),
            )
        ]
        extracted = ExtractedFacts(
            doc_id="doc_test_1",
            page_number=16,
            facts=facts,
            model_used="gemini-3.8-flash",
            fallback_attempts=0,
        )

        self.storage.save_extracted_facts("doc_test_1", extracted)
        self.assertTrue(self.storage.has_extracted_facts("doc_test_1"))

        retrieved = self.storage.get_extracted_facts("doc_test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.doc_id, "doc_test_1")
        self.assertEqual(len(retrieved.facts), 1)
        self.assertEqual(retrieved.facts[0].subject, "Delhivery")
        self.assertEqual(retrieved.facts[0].canonical_value, 4810500000.0)

        all_facts = self.storage.get_all_extracted_facts()
        self.assertIn("doc_test_1", all_facts)
        self.assertEqual(len(all_facts["doc_test_1"]), 1)

    def test_save_and_get_claim_graph(self):
        graph = ClaimGraph(
            documents={"doc1": "delhivery.pdf", "doc2": "annual_report.pdf"},
            total_facts=2,
            clusters=[
                FactCluster(
                    cluster_id="c12345",
                    subject="Delhivery",
                    predicate="revenue",
                    fact_indices=[0, 1],
                    case_type=CaseType.CORROBORATED,
                    dispute_code="AGREEMENT_EXACT",
                    evidence=[
                        EvidenceEntry(
                            doc_id="doc1",
                            doc_filename="delhivery.pdf",
                            page=1,
                            value="₹500 Cr",
                            canonical_value=5000000000.0,
                            canonical_unit="INR",
                            evidence_quote="revenue was ₹500 Cr",
                            confidence=1.0,
                            match_type="exact",
                        ),
                        EvidenceEntry(
                            doc_id="doc2",
                            doc_filename="annual_report.pdf",
                            page=5,
                            value="₹500 Cr",
                            canonical_value=5000000000.0,
                            canonical_unit="INR",
                            evidence_quote="reported revenue was ₹500 Cr",
                            confidence=1.0,
                            match_type="exact",
                        ),
                    ],
                    edges=[
                        ClaimEdge(
                            source_fact_idx=0,
                            target_fact_idx=1,
                            edge_type=EdgeType.CORROBORATES,
                            detection_method="symbolic_numeric",
                            confidence=1.0,
                            explanation="Exact match",
                        )
                    ],
                    explanation="Both sources agree on ₹500 Cr",
                    doc_count=2,
                )
            ],
            pipeline_metadata={"duration_seconds": 1.25},
        )

        self.storage.save_claim_graph(graph, graph_id="latest")
        self.assertTrue(self.storage.has_claim_graph("latest"))

        retrieved = self.storage.get_claim_graph("latest")
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved.documents), 2)
        self.assertEqual(retrieved.total_facts, 2)
        self.assertEqual(len(retrieved.clusters), 1)
        self.assertEqual(retrieved.clusters[0].case_type, CaseType.CORROBORATED)
        self.assertEqual(retrieved.clusters[0].doc_count, 2)

    def test_persistence_across_server_restart(self):
        """Simulate process shutdown and restart against the same SQLite database file."""
        doc = DocumentData(
            doc_id="doc_restart",
            filename="rbi_report.pdf",
            page_count=5,
            pages=[],
            raw_text_index={},
        )
        self.storage.save_document(doc)

        # Close first connection instance
        self.storage.close()

        # Reopen new storage instance on the same file path
        new_storage = SQLiteStorage(self.db_path)
        try:
            self.assertTrue(new_storage.has_document("doc_restart"))
            retrieved = new_storage.get_document("doc_restart")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.filename, "rbi_report.pdf")
        finally:
            new_storage.close()


if __name__ == "__main__":
    unittest.main()
