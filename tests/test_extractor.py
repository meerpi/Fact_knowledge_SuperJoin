import unittest
from unittest.mock import MagicMock
from google.genai import errors

from app.extractor import GeminiFactExtractor, DEFAULT_MODEL_CASCADE
from app.models import DocumentData, Fact, PageData, RawFactExtraction, RawFactItem


class TestGeminiFactExtractor(unittest.TestCase):
    def test_missing_api_key_raises_helpful_error(self):
        extractor = GeminiFactExtractor(api_key=None)
        extractor.client = None
        with self.assertRaises(ValueError) as ctx:
            extractor.extract_from_text("some text")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_fallback_cascade_on_rate_limit(self):
        mock_client = MagicMock()

        sample_fact_json = RawFactExtraction(
            facts=[
                RawFactItem(
                    subject="Alphabet Inc.",
                    predicate="revenue",
                    value="$76.69B",
                    numeric_value=76690000000.0,
                    unit="USD",
                    temporal="Q3 2023",
                    scope="Consolidated",
                    conditions="Restated",
                    evidence_quote="Consolidated revenues were $76.69 billion",
                )
            ]
        ).model_dump_json()

        mock_response_success = MagicMock()
        mock_response_success.text = sample_fact_json

        rate_limit_error = errors.ClientError(
            code=429,
            response_json={"error": {"message": "Resource has been exhausted."}}
        )

        mock_client.models.generate_content.side_effect = [
            rate_limit_error,
            mock_response_success,
        ]

        extractor = GeminiFactExtractor(
            api_key="test_key",
            model_cascade=["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
            client=mock_client,
        )

        facts, model_used, attempts = extractor.extract_from_text("Sample text")

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Alphabet Inc.")
        self.assertEqual(facts[0].numeric_value, 76690000000.0)
        self.assertEqual(model_used, "gemini-3.7-flash")
        self.assertEqual(attempts, 1)

    def test_all_models_fail_raises_runtime_error(self):
        mock_client = MagicMock()
        server_error = errors.ServerError(code=503, response_json={"error": {"message": "Model capacity exhausted"}})
        mock_client.models.generate_content.side_effect = server_error

        extractor = GeminiFactExtractor(
            api_key="test_key",
            model_cascade=["gemini-3.8-flash", "gemini-3.7-flash"],
            client=mock_client,
        )

        with self.assertRaises(RuntimeError) as ctx:
            extractor.extract_from_text("Sample text")
        self.assertIn("All Gemini models in fallback cascade failed", str(ctx.exception))

    def test_quote_verification_and_confidence_calibration(self):
        doc = DocumentData(
            doc_id="doc_123",
            filename="financial.pdf",
            page_count=1,
            pages=[
                PageData(
                    page_number=0,
                    raw_text="Total revenue for Q3 was $76.69 billion, up 11% year over year.",
                    text_blocks=[],
                    tables=[],
                )
            ],
            raw_text_index={
                0: "Total revenue for Q3 was $76.69 billion, up 11% year over year."
            },
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = RawFactExtraction(
            facts=[
                RawFactItem(
                    subject="Alphabet",
                    predicate="revenue",
                    value="$76.69 billion",
                    numeric_value=76690000000.0,
                    temporal="Q3",
                    scope="Consolidated",
                    evidence_quote="Total revenue for Q3 was $76.69 billion",
                ),
                RawFactItem(
                    subject="Alphabet",
                    predicate="net_margin",
                    value="42.5%",
                    numeric_value=0.425,
                    evidence_quote="Net profit margin reached an unprecedented 42.5%",
                ),
            ]
        ).model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        extractor = GeminiFactExtractor(api_key="test_key", client=mock_client)
        result = extractor.extract_and_verify(doc, page_num=0)

        self.assertEqual(len(result.facts), 2)
        
        # 6-Tuple Verification: Fact 1 (Verified Exact Match)
        f1 = result.facts[0]
        self.assertEqual(f1.subject, "Alphabet")
        self.assertEqual(f1.predicate, "revenue")
        self.assertEqual(f1.value, "$76.69 billion")
        self.assertEqual(f1.context.temporal, "Q3")
        self.assertEqual(f1.context.scope, "Consolidated")
        self.assertEqual(f1.confidence, 1.00)
        self.assertTrue(f1.provenance.verified)
        self.assertEqual(f1.provenance.match_type, "exact")
        self.assertEqual(f1.provenance.doc_id, "doc_123")

        # 6-Tuple Verification: Fact 2 (Hallucinated Quote -> low confidence)
        f2 = result.facts[1]
        self.assertEqual(f2.subject, "Alphabet")
        self.assertEqual(f2.predicate, "net_margin")
        self.assertEqual(f2.confidence, 0.10)
        self.assertFalse(f2.provenance.verified)
        self.assertEqual(f2.provenance.match_type, "unverified")

    def test_rate_limiter_pacing(self):
        import time
        from app.extractor import RateLimiter

        limiter = RateLimiter(min_interval_seconds=0.05)
        t0 = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - t0
        self.assertGreaterEqual(elapsed, 0.045)

    def test_micro_batching_groups_pages(self):
        doc = DocumentData(
            doc_id="doc_multipage",
            filename="macro.pdf",
            page_count=4,
            pages=[
                PageData(page_number=i, raw_text=f"Page {i} data content line", text_blocks=[], tables=[])
                for i in range(4)
            ],
            raw_text_index={
                i: f"Page {i} data content line" for i in range(4)
            },
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = RawFactExtraction(
            facts=[
                RawFactItem(
                    subject="India",
                    predicate="gdp_growth",
                    value="7.2%",
                    numeric_value=7.2,
                    evidence_quote="data content line",
                    page_number=1,
                )
            ]
        ).model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        extractor = GeminiFactExtractor(api_key="test_key", client=mock_client)
        result = extractor.extract_and_verify(doc, pages=[0, 1, 2, 3], batch_size=2)

        # 4 pages with batch_size=2 should result in exactly 2 LLM calls!
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        self.assertEqual(len(result.facts), 2)

    def test_circuit_breaker_bypasses_failed_model(self):
        extractor = GeminiFactExtractor(
            api_key="test_key",
            model_cascade=["model-a", "model-b"],
            client=MagicMock(),
        )
        self.assertTrue(extractor.is_model_available("model-a"))
        extractor.trip_circuit_breaker("model-a", duration=100.0)
        self.assertFalse(extractor.is_model_available("model-a"))
        self.assertTrue(extractor.is_model_available("model-b"))

    def test_semantic_batches_layout_continuity(self):
        from app.models import TableBlock
        # Create a 6-page doc where pages 1 and 2 have tables (continuation)
        pages = [
            PageData(page_number=0, raw_text="Short title page", text_blocks=[], tables=[]),
            PageData(page_number=1, raw_text="P1 text", text_blocks=[], tables=[TableBlock(headers=["ColA"], rows=[["Val1"]], page=1)]),
            PageData(page_number=2, raw_text="P2 text", text_blocks=[], tables=[TableBlock(headers=["ColA"], rows=[["Val2"]], page=2)]),
            PageData(page_number=3, raw_text="P3 narrative " * 100, text_blocks=[], tables=[]),
            PageData(page_number=4, raw_text="P4 narrative " * 100, text_blocks=[], tables=[]),
            PageData(page_number=5, raw_text="P5 narrative " * 100, text_blocks=[], tables=[]),
        ]
        doc = DocumentData(doc_id="d_sem", filename="f.pdf", page_count=6, pages=pages)
        batches = GeminiFactExtractor._create_semantic_batches(doc, list(range(6)), target_tokens=300, max_pages_per_batch=4)
        # Pages 1 and 2 should stay together due to table continuation
        found_together = any(1 in b and 2 in b for b in batches)
        self.assertTrue(found_together)


if __name__ == "__main__":
    unittest.main()
