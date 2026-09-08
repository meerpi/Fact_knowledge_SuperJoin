import asyncio
import logging
import os
import threading
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

from app.models import (
    Context,
    DocumentData,
    ExtractedFacts,
    Fact,
    Provenance,
    RawFactExtraction,
    RawFactItem,
)
from app.normalizer import (
    canonicalize_unit,
    extract_currency,
    extract_scale_from_unit,
    parse_numeric_value,
    standardize_predicate,
)
from app.pdf_parser import chunk_document, verify_quote

logger = logging.getLogger(__name__)

# Fallback sequence: best reasoning / complex layout at top down to lowest latency / cost
DEFAULT_MODEL_CASCADE = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

EXTRACTION_SYSTEM_PROMPT = """You are a high-precision financial and numerical fact extraction engine.
Your task is to decompose text into atomic assertions with structured unit/scale classification (following the iXBRL standard).

For each fact, extract:
1. subject: Entity, subsidiary, or subject of assertion (e.g., 'Delhivery', 'Spoton', 'Restated Summary Consolidated Profit and Loss Data').
2. predicate: Attribute, relation, or measured metric in snake_case (e.g., 'revenue_from_contracts', 'pin_code_reach', 'total_expenses', 'ebitda').
3. value: Exact text string representation as shown in document (e.g., '48,105.30', '14.5%', '13,087', '(8,911.39)').
4. numeric_value: Normalized float value if quantitative (e.g., 48105.3, 14.5, 13087.0, -8911.39). Null if non-numeric.
5. base_unit: The FUNDAMENTAL unit of measurement WITHOUT any scale prefix.
   - CORRECT: 'INR', 'USD', '%', 'square feet', 'employees', 'PIN codes', 'shipments', 'locations'
   - WRONG: 'INR million', 'million square feet', 'USD billion' — never include scale words in the unit.
   - Null if the value is dimensionless.
6. scale: The power of 10 that the displayed value must be multiplied by, as an INTEGER.
   Read the table header, column header, or footnotes for phrases like '₹ in million', 'in thousands', '₹ in crore'.
   Common values: 0 (ones/units), 3 (thousands), 5 (lakhs), 6 (millions), 7 (crores), 9 (billions), 12 (trillions).
   Default to 0 if no scale indicator is found.
7. dimension: Category of quantity. One of: 'monetary', 'count', 'area', 'percentage', 'ratio', 'duration', 'weight', 'volume', 'length', 'other'.
8. temporal: Applicable fiscal period, quarter, or date (e.g., 'Nine months ended Dec 31, 2021', 'FY2021'). Null if unspecified.
9. scope: Business scope or entity boundary (e.g., 'Consolidated', 'Standalone', 'Spoton subsidiary'). Null if general.
10. conditions: Accounting or qualifying conditions (e.g., 'Restated', 'Excluding ESOP', 'Pre-tax', 'Includes Spoton results'). Null if standard.
11. evidence_quote: CRITICAL. Must be an exact verbatim substring from the provided text supporting this fact. Do not paraphrase or alter characters.
12. page_number: Integer page number where this assertion appears in the provided document section.

IMPORTANT scale examples:
- Table header says '₹ in million': every INR value in that table gets scale=6
- Table header says 'in thousands': scale=3
- Table header says '₹ in crore': scale=7
- A value like '2.62 million square feet': scale=6, base_unit='square feet'
- A standalone count like '12,764 PIN codes': scale=0, base_unit='PIN codes'
- A percentage like '14.5%': scale=0, base_unit='%', dimension='percentage'"""


def _calculate_confidence(verified: bool, match_type: str) -> float:
    if not verified:
        return 0.10
    if match_type == "exact":
        return 1.00
    if match_type == "normalized":
        return 0.95
    if match_type == "fuzzy":
        return 0.80
    return 0.50


class RateLimiter:
    """Thread-safe rate limiter to enforce a minimum interval between live requests.
    Prevents HTTP 429 RESOURCE_EXHAUSTED on the Gemini API (15 RPM cap).
    Default 4.2s interval enforces a maximum rate of ~14.2 requests/minute.
    """

    def __init__(self, min_interval_seconds: float = 4.2):
        self.min_interval = min_interval_seconds
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return

        with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_call_time = time.time()


class AsyncRateLimiter:
    """Async-compatible rate limiter using asyncio.Lock.
    Ensures minimum interval between API calls in async context.
    """

    def __init__(self, min_interval_seconds: float = 4.2):
        self.min_interval = min_interval_seconds
        self.last_call_time = 0.0
        self._lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        # Lazy init to avoid binding to wrong event loop
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def wait(self):
        if self.min_interval <= 0:
            return

        async with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                await asyncio.sleep(sleep_time)
            self.last_call_time = time.time()


class GeminiFactExtractor:
    def __init__(
        self,
        api_key: str | None = None,
        model_cascade: list[str] | None = None,
        client: genai.Client | None = None,
        min_request_interval: float = 4.2,
        max_concurrent: int = 3,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.model_cascade = model_cascade or list(DEFAULT_MODEL_CASCADE)
        self.max_concurrent = max_concurrent

        if client is not None:
            self.client = client
            self._is_real_client = isinstance(client, genai.Client)
            # When mock client is passed without explicit interval, disable artificial delay
            if min_request_interval == 4.2:
                min_request_interval = 0.0
        elif self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self._is_real_client = True
        else:
            self.client = None
            self._is_real_client = False

        self.rate_limiter = RateLimiter(min_interval_seconds=min_request_interval)
        self.async_rate_limiter = AsyncRateLimiter(min_interval_seconds=min_request_interval)

        # Context cache for the system prompt (lazily created)
        self._context_cache_name: str | None = None
        self._context_cache_model: str | None = None

    # ------------------------------------------------------------------
    # Context Caching: cache the system prompt to avoid re-processing
    # ------------------------------------------------------------------

    def _get_or_create_context_cache(self, model_name: str) -> str | None:
        """Create a Gemini context cache for the system prompt.

        This caches the 700+ token system prompt so it's processed once
        instead of on every API call. Cached tokens cost 75% less and
        have lower time-to-first-token latency.

        Returns the cache name or None if caching is unavailable.
        """
        if self._context_cache_name and self._context_cache_model == model_name:
            return self._context_cache_name

        if not self.client:
            return None

        try:
            cache = self.client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    display_name="fact-extraction-system-prompt",
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    ttl="3600s",  # 1 hour
                ),
            )
            # Validate that cache.name is a real string (not a MagicMock from tests)
            if not isinstance(cache.name, str):
                logger.debug("Context cache returned non-string name (likely mock client), skipping.")
                return None
            self._context_cache_name = cache.name
            self._context_cache_model = model_name
            logger.info(
                "Created context cache '%s' for model '%s' (system prompt cached, 75%% cost reduction on prefix)",
                cache.name, model_name,
            )
            return cache.name
        except Exception as e:
            logger.debug(
                "Context caching unavailable for model '%s': %s. Falling back to inline system prompt.",
                model_name, e,
            )
            return None

    def _build_config(self, model_name: str) -> types.GenerateContentConfig:
        """Build GenerateContentConfig, using context cache if available."""
        cache_name = self._get_or_create_context_cache(model_name)

        if cache_name:
            # With context cache: system prompt is in the cache, don't send it again
            return types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
                response_schema=RawFactExtraction,
                temperature=0.0,
            )
        else:
            # Without context cache: inline system prompt
            return types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RawFactExtraction,
                temperature=0.0,
            )

    # ------------------------------------------------------------------
    # Synchronous extraction (backward-compatible)
    # ------------------------------------------------------------------

    def _call_model_with_fallback(self, prompt: str) -> tuple[RawFactExtraction, str, int]:
        if not self.client:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free API key at https://aistudio.google.com/apikey and set it via export GEMINI_API_KEY=..."
            )

        failures = []
        for attempt, model_name in enumerate(self.model_cascade):
            try:
                config = self._build_config(model_name)
                self.rate_limiter.wait()
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

                if not response.text:
                    failures.append(f"{model_name}: empty response")
                    continue

                extracted = RawFactExtraction.model_validate_json(response.text)
                return extracted, model_name, attempt

            except errors.APIError as e:
                logger.warning("Gemini model '%s' failed (HTTP %s): %s. Falling back.", model_name, e.code, e.message)
                failures.append(f"{model_name} (HTTP {e.code}): {e.message}")
                # If context cache caused the error, invalidate it
                if e.code in (400, 404) and self._context_cache_name:
                    self._context_cache_name = None
                    self._context_cache_model = None
            except Exception as e:
                logger.warning("Gemini model '%s' failed with unexpected error: %s. Falling back.", model_name, e)
                failures.append(f"{model_name}: {str(e)}")

        raise RuntimeError(f"All Gemini models in fallback cascade failed: {'; '.join(failures)}")

    # ------------------------------------------------------------------
    # Async extraction (new — parallel page-batch processing)
    # ------------------------------------------------------------------

    async def _async_call_model(self, prompt: str, model_name: str) -> tuple[RawFactExtraction, str]:
        """Single async API call to a specific model."""
        if not self.client:
            raise ValueError("GEMINI_API_KEY not set")

        config = self._build_config(model_name)
        await self.async_rate_limiter.wait()

        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise ValueError(f"{model_name}: empty response")

        extracted = RawFactExtraction.model_validate_json(response.text)
        return extracted, model_name

    async def _async_call_with_fallback(self, prompt: str) -> tuple[RawFactExtraction, str, int]:
        """Async version of _call_model_with_fallback with model cascade."""
        failures = []
        for attempt, model_name in enumerate(self.model_cascade):
            try:
                extracted, used_model = await self._async_call_model(prompt, model_name)
                return extracted, used_model, attempt
            except errors.APIError as e:
                logger.warning("Async: model '%s' failed (HTTP %s): %s", model_name, e.code, e.message)
                failures.append(f"{model_name} (HTTP {e.code}): {e.message}")
                if e.code in (400, 404) and self._context_cache_name:
                    self._context_cache_name = None
                    self._context_cache_model = None
            except Exception as e:
                logger.warning("Async: model '%s' failed: %s", model_name, e)
                failures.append(f"{model_name}: {str(e)}")

        raise RuntimeError(f"All models failed (async): {'; '.join(failures)}")

    async def _async_extract_page_batch(
        self,
        doc: DocumentData,
        batch_pages: list[int],
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[tuple[RawFactItem, int | None]], str, int]:
        """Extract facts from a page batch, respecting concurrency limit."""
        sections = []
        for p in batch_pages:
            if p < 0 or p >= doc.page_count:
                continue
            page = doc.pages[p]
            text = page.raw_text
            if not text.strip() and page.tables:
                text = "\n".join(
                    " | ".join(t.headers) + "\n" + "\n".join(" | ".join(row) for row in t.rows)
                    for t in page.tables
                )
            if text.strip():
                sections.append(f"=== Page {p} ===\n{text}\n=== End of Page {p} ===")

        if not sections:
            return [], "", 0

        pages_str = ", ".join(f"Page {p}" for p in batch_pages)
        prompt = (
            f"Document: {doc.filename}\n"
            f"Extract all quantitative and financial facts from the following section ({pages_str}).\n"
            f"For each fact, set 'page_number' to the exact integer page number where it appears.\n\n"
            + "\n\n".join(sections)
        )

        async with semaphore:
            extracted, model_used, attempts = await self._async_call_with_fallback(prompt)

        items_with_page: list[tuple[RawFactItem, int | None]] = []
        for item in extracted.facts:
            hint_page = (
                item.page_number
                if (item.page_number is not None and item.page_number in batch_pages)
                else (item.page_number or batch_pages[0])
            )
            items_with_page.append((item, hint_page))

        return items_with_page, model_used, attempts

    async def _async_extract_all_batches(
        self,
        doc: DocumentData,
        page_batches: list[list[int]],
    ) -> list[tuple[list[tuple[RawFactItem, int | None]], str, int]]:
        """Extract all page batches concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [
            self._async_extract_page_batch(doc, batch, semaphore)
            for batch in page_batches
        ]

        t_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - t_start

        # Separate successes from failures
        successful = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Async batch %d (pages %s) failed: %s",
                    i, page_batches[i], result,
                )
                successful.append(([], "", 0))
            else:
                successful.append(result)

        logger.info(
            "Async extraction: %d batches completed in %.1fs (%.1fx faster than sequential estimate of %.1fs)",
            len(page_batches), duration,
            (len(page_batches) * self.rate_limiter.min_interval) / max(duration, 0.001),
            len(page_batches) * self.rate_limiter.min_interval,
        )

        return successful

    # ------------------------------------------------------------------
    # Public API (backward-compatible, uses async internally when possible)
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str, page_num: int | None = None) -> tuple[list[RawFactItem], str, int]:
        page_context = f"\n[Context: Page {page_num}]" if page_num is not None else ""
        prompt = f"Document content to extract facts from:{page_context}\n\n{text}"

        extracted, model_used, attempts = self._call_model_with_fallback(prompt)
        return extracted.facts, model_used, attempts

    def _extract_page_batch(
        self,
        doc: DocumentData,
        batch_pages: list[int],
    ) -> tuple[list[tuple[RawFactItem, int | None]], str, int]:
        """Extract facts from a micro-batch of pages (typically 2 pages)."""
        sections = []
        for p in batch_pages:
            if p < 0 or p >= doc.page_count:
                continue
            page = doc.pages[p]
            text = page.raw_text
            if not text.strip() and page.tables:
                text = "\n".join(
                    " | ".join(t.headers) + "\n" + "\n".join(" | ".join(row) for row in t.rows)
                    for t in page.tables
                )
            if text.strip():
                sections.append(f"=== Page {p} ===\n{text}\n=== End of Page {p} ===")

        if not sections:
            return [], "", 0

        pages_str = ", ".join(f"Page {p}" for p in batch_pages)
        prompt = (
            f"Document: {doc.filename}\n"
            f"Extract all quantitative and financial facts from the following section ({pages_str}).\n"
            f"For each fact, set 'page_number' to the exact integer page number where it appears.\n\n"
            + "\n\n".join(sections)
        )

        extracted, model_used, attempts = self._call_model_with_fallback(prompt)
        items_with_page: list[tuple[RawFactItem, int | None]] = []
        for item in extracted.facts:
            hint_page = (
                item.page_number
                if (item.page_number is not None and item.page_number in batch_pages)
                else (item.page_number or batch_pages[0])
            )
            items_with_page.append((item, hint_page))

        return items_with_page, model_used, attempts

    def _ground_facts(
        self,
        doc: DocumentData,
        raw_items_with_page: list[tuple[RawFactItem, int | None]],
        page_num: int | None = None,
    ) -> list[Fact]:
        """Ground every extracted fact against the parsed PDF text index into 6-tuples.

        This is the deterministic post-processing step — zero LLM calls.
        """
        structured_facts: list[Fact] = []
        for item, hint_page in raw_items_with_page:
            target_page = item.page_number if item.page_number is not None else hint_page
            if item.evidence_quote:
                verification = verify_quote(doc, quote=item.evidence_quote, page=target_page)
                if not verification["found"] and target_page != hint_page:
                    verification = verify_quote(doc, quote=item.evidence_quote, page=hint_page)
                if not verification["found"]:
                    verification = verify_quote(doc, quote=item.evidence_quote, page=None)

                verified = verification["found"]
                match_type = verification["match_type"] or "unverified"
                found_page = verification.get("page") if verification.get("page") is not None else target_page
            else:
                verified = False
                match_type = "unverified"
                found_page = target_page

            confidence = _calculate_confidence(verified, match_type)

            context = Context(
                temporal=item.temporal,
                scope=item.scope,
                conditions=item.conditions,
            )

            provenance = Provenance(
                doc_id=doc.doc_id,
                page=found_page,
                evidence_quote=item.evidence_quote,
                verified=verified,
                match_type=match_type,
            )

            fact = Fact(
                subject=item.subject,
                predicate=item.predicate,
                value=item.value,
                numeric_value=item.numeric_value,
                base_unit=item.base_unit,
                scale=item.scale,
                dimension=item.dimension,
                context=context,
                confidence=confidence,
                provenance=provenance,
            )

            # --- iXBRL-pattern canonicalization (deterministic, zero LLM calls) ---

            # 1. Compute canonical_value = numeric_value × 10^scale
            base_num, scale_in_value, _ = parse_numeric_value(fact.value)
            effective_num = base_num if base_num is not None else fact.numeric_value

            if effective_num is not None:
                llm_scale = fact.scale or 0

                # Cross-check: if value string itself had a scale word (e.g. "$1,500 million"),
                # the symbolic parser already applied it, so don't double-apply
                if scale_in_value is not None:
                    # Scale was in the value string — symbolic parser already multiplied
                    fact.canonical_value = base_num * (10 ** 0)  # base_num already has it
                    _, _, canon_from_value = parse_numeric_value(fact.value)
                    fact.canonical_value = canon_from_value
                else:
                    # Scale comes from LLM's structured field (read from table header)
                    fact.canonical_value = effective_num * (10 ** llm_scale)

                # Symbolic cross-check: verify LLM scale against normalizer
                symbolic_scale = extract_scale_from_unit(fact.base_unit)
                if symbolic_scale > 1.0:
                    # base_unit contains a scale word the LLM should have separated out
                    logger.warning(
                        "Scale word found in base_unit '%s' — LLM should have put this in scale field",
                        fact.base_unit,
                    )

            # 2. Reconstruct legacy unit field for display/backward compat
            scale_names = {3: "thousand", 5: "lakh", 6: "million", 7: "crore", 9: "billion", 12: "trillion"}
            if fact.base_unit and fact.scale and fact.scale in scale_names:
                fact.unit = f"{fact.base_unit} {scale_names[fact.scale]}"
            else:
                fact.unit = fact.base_unit

            # 3. Canonical unit = base_unit (already clean, no scale words)
            currency = extract_currency(fact.value) or extract_currency(fact.base_unit or "")
            fact.canonical_unit = fact.base_unit or currency

            # 4. Standardize predicate + extract conditions
            std_pred, condition = standardize_predicate(fact.predicate)
            fact.predicate = std_pred
            if condition and not fact.context.conditions:
                fact.context.conditions = condition

            structured_facts.append(fact)

        return structured_facts

    def extract_and_verify(
        self,
        doc: DocumentData,
        page_num: int | None = None,
        pages: list[int] | None = None,
        batch_size: int = 2,
        use_async: bool = True,
    ) -> ExtractedFacts:
        raw_items_with_page: list[tuple[RawFactItem, int | None]] = []
        model_used = ""
        total_attempts = 0

        if page_num is not None:
            if page_num < 0 or page_num >= doc.page_count:
                raise ValueError(f"Page {page_num} is out of bounds (document has {doc.page_count} pages)")
            page_batches = [[page_num]]
        elif pages is not None:
            page_batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
        else:
            all_pages = list(range(doc.page_count))
            page_batches = [all_pages[i:i + batch_size] for i in range(0, len(all_pages), batch_size)]

        # Decide: use async parallel extraction or sequential
        # Only use async with real genai.Client (mock clients can't be awaited)
        use_parallel = (
            use_async
            and self._is_real_client
            and len(page_batches) > 1
            and page_num is None  # Don't parallelize single-page requests
        )

        if use_parallel:
            # ---- ASYNC PARALLEL EXTRACTION ----
            logger.info(
                "Using async parallel extraction for %d batches (max_concurrent=%d)",
                len(page_batches), self.max_concurrent,
            )
            try:
                # Get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context — can't use asyncio.run()
                    # Fall back to sequential
                    use_parallel = False
                except RuntimeError:
                    # No running loop — safe to use asyncio.run()
                    pass

                if use_parallel:
                    results = asyncio.run(
                        self._async_extract_all_batches(doc, page_batches)
                    )
                    for items_with_page, m_used, attempts in results:
                        if not model_used and m_used:
                            model_used = m_used
                        total_attempts = max(total_attempts, attempts)
                        raw_items_with_page.extend(items_with_page)

            except Exception as e:
                logger.warning(
                    "Async extraction failed (%s), falling back to sequential", e,
                )
                use_parallel = False
                raw_items_with_page = []

        if not use_parallel:
            # ---- SEQUENTIAL EXTRACTION (original behavior) ----
            for batch in page_batches:
                if len(batch) == 1 and page_num is not None:
                    p = batch[0]
                    page = doc.pages[p]
                    text_to_extract = page.raw_text
                    if not text_to_extract.strip() and page.tables:
                        text_to_extract = "\n".join(
                            " | ".join(t.headers) + "\n" + "\n".join(" | ".join(row) for row in t.rows)
                            for t in page.tables
                        )
                    if text_to_extract.strip():
                        items, m_used, attempts = self.extract_from_text(text_to_extract, page_num=p)
                        if not model_used:
                            model_used = m_used
                        total_attempts = max(total_attempts, attempts)
                        raw_items_with_page.extend((item, p) for item in items)
                else:
                    items_with_page, m_used, attempts = self._extract_page_batch(doc, batch)
                    if not model_used:
                        model_used = m_used
                    total_attempts = max(total_attempts, attempts)
                    raw_items_with_page.extend(items_with_page)

        # Ground every extracted fact against the parsed PDF text index
        structured_facts = self._ground_facts(doc, raw_items_with_page, page_num)

        return ExtractedFacts(
            doc_id=doc.doc_id,
            page_number=page_num,
            facts=structured_facts,
            model_used=model_used,
            fallback_attempts=total_attempts,
        )
