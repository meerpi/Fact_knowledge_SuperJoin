import asyncio
import logging
import os
import re
import threading
import time
from typing import Callable, Any
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
# Fallback sequence: high-availability production models first, down to lite models
DEFAULT_MODEL_CASCADE = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
]

# Circuit breaker cooldown duration for high demand (503) or rate limits (429)
_CIRCUIT_COOLDOWN_SECONDS = 300.0  # 5 minutes cooldown

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
7. dimension: Category of quantity. Standard types: 'monetary', 'count', 'area', 'percentage', 'ratio', 'duration', 'weight', 'volume', 'length', or any domain-specific category (e.g., 'speed', 'temperature', 'energy'), or 'other'.
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
- A percentage like '14.5%': scale=0, base_unit='%', dimension='percentage'

IMPORTANT table multi-indicator guidance:
- When a table contains multiple indicators (e.g., Level/Amount in USD/INR vs Annual Growth in % vs Share of GDP in %):
  * Disambiguate each row's predicate and dimension explicitly.
  * Never name a growth-rate row with the base metric predicate (e.g., use 'merchandise_exports_annual_growth' for growth rate, NOT 'merchandise_exports').
  * An absolute level is ALWAYS dimension='monetary' (or 'count'). A growth rate is ALWAYS dimension='percentage' with base_unit='%'.
  * Never merge or confuse level values with growth-rate percentages."""


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
    """Token-bucket async rate limiter allowing concurrent bursts up to capacity
    while enforcing sustainable average requests per minute (RPM).
    """

    def __init__(self, min_interval_seconds: float = 4.2, capacity: int = 5):
        self.min_interval = min_interval_seconds
        self.fill_rate = (1.0 / max(min_interval_seconds, 0.01)) if min_interval_seconds > 0 else 100.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def wait(self):
        if self.min_interval <= 0:
            return

        sleep_time = 0.0
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)

            if self.tokens >= 1.0:
                self.tokens -= 1.0
            else:
                needed = 1.0 - self.tokens
                sleep_time = needed / self.fill_rate
                self.tokens = 0.0

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


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

        self.model_cascade = list(model_cascade) if model_cascade else self._discover_models()
        self.max_concurrent = max_concurrent

        self.rate_limiter = RateLimiter(min_interval_seconds=min_request_interval)
        self.async_rate_limiter = AsyncRateLimiter(min_interval_seconds=min_request_interval)

        # Instance-level circuit breaker: {model_name: cooldown_timestamp}
        self._circuit_breaker: dict[str, float] = {}

        # Context cache for the system prompt (lazily created)
        self._context_cache_name: str | None = None
        self._context_cache_model: str | None = None

    def _discover_models(self) -> list[str]:
        """Dynamically discover available text models from the Gemini API at runtime.
        Adapts dynamically to model updates and regional availability without hardcoding.
        """
        if not self.client or not self._is_real_client:
            return list(DEFAULT_MODEL_CASCADE)

        try:
            excluded = ("tts", "audio", "image", "embedding", "robotics", "computer-use", "live", "transcribe", "customtools")
            flash_models: list[str] = []
            other_models: list[str] = []
            for m in self.client.models.list():
                clean_name = getattr(m, "name", "").replace("models/", "")
                if not clean_name or any(kw in clean_name.lower() for kw in excluded):
                    continue
                if "flash" in clean_name.lower():
                    flash_models.append(clean_name)
                elif "gemini" in clean_name.lower():
                    other_models.append(clean_name)

            flash_models.sort(reverse=True)
            other_models.sort(reverse=True)
            discovered = flash_models + other_models
            if discovered:
                logger.info("Dynamically discovered %d Gemini models: %s", len(discovered), discovered[:4])
                return discovered
        except Exception as e:
            logger.debug("Dynamic Gemini model discovery skipped (%s)", e)

        return list(DEFAULT_MODEL_CASCADE)

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is currently healthy or in circuit-breaker cooldown."""
        cooldown = self._circuit_breaker.get(model_name, 0.0)
        return time.time() >= cooldown

    def trip_circuit_breaker(
        self,
        model_name: str,
        duration: float = _CIRCUIT_COOLDOWN_SECONDS,
        err_msg: str = "",
    ) -> None:
        """Trip circuit breaker to temporarily bypass failing/congested model.
        Dynamically extracts exact retry-after duration if reported by the API.
        """
        if err_msg:
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
            if m:
                duration = max(5.0, float(m.group(1)) + 1.0)
            elif "404" in err_msg or "not found" in err_msg.lower():
                duration = 3600.0  # 1 hour cooldown for deprecated endpoints
            elif "503" in err_msg:
                duration = 60.0    # 1 minute cooldown for temporary high demand
        self._circuit_breaker[model_name] = time.time() + duration
        logger.warning("Circuit breaker tripped for model '%s' (cooling down for %.1fs)", model_name, duration)

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

        available_models = [m for m in self.model_cascade if self.is_model_available(m)]
        models_to_try = available_models if available_models else self.model_cascade

        failures = []
        for attempt, model_name in enumerate(models_to_try):
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
                if e.code in (404, 429, 503):
                    self.trip_circuit_breaker(model_name, err_msg=e.message or str(e))
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
        available_models = [m for m in self.model_cascade if self.is_model_available(m)]
        models_to_try = available_models if available_models else self.model_cascade

        failures = []
        for attempt, model_name in enumerate(models_to_try):
            try:
                extracted, used_model = await self._async_call_model(prompt, model_name)
                return extracted, used_model, attempt
            except errors.APIError as e:
                logger.warning("Async: model '%s' failed (HTTP %s): %s", model_name, e.code, e.message)
                failures.append(f"{model_name} (HTTP {e.code}): {e.message}")
                if e.code in (404, 429, 503):
                    self.trip_circuit_breaker(model_name, err_msg=e.message or str(e))
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
            if page.tables:
                table_md_blocks = [
                    " | ".join(t.headers) + "\n" + "\n".join(" | ".join(row) for row in t.rows)
                    for t in page.tables
                    if t.headers or t.rows
                ]
                if table_md_blocks:
                    tables_text = "\n\n".join(table_md_blocks)
                    if not text.strip():
                        text = tables_text
                    else:
                        text = text + "\n\n[Structured Tables from Document Layout]:\n" + tables_text
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
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[tuple[list[tuple[RawFactItem, int | None]], str, int]]:
        """Extract all page batches concurrently with bounded parallelism and progress reporting."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        total_pages = sum(len(b) for b in page_batches)
        pages_done = 0

        async def _run_batch(idx: int, batch: list[int]):
            nonlocal pages_done
            try:
                res = await self._async_extract_page_batch(doc, batch, semaphore)
                pages_done += len(batch)
                if on_progress:
                    on_progress(pages_done, total_pages, f"Extracted batch {idx + 1} of {len(page_batches)}")
                return idx, res
            except Exception as e:
                logger.error(
                    "Async batch %d (pages %s) failed: %s",
                    idx, batch, e,
                )
                pages_done += len(batch)
                if on_progress:
                    on_progress(pages_done, total_pages, f"Batch {idx + 1} completed with error")
                return idx, ([], "", 0)

        t_start = time.time()
        tasks = [_run_batch(i, b) for i, b in enumerate(page_batches)]
        results_with_idx = await asyncio.gather(*tasks)
        results_with_idx.sort(key=lambda x: x[0])
        successful = [r[1] for r in results_with_idx]
        duration = time.time() - t_start

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

    @staticmethod
    def _create_semantic_batches(
        doc: DocumentData,
        requested_pages: list[int],
        target_tokens: int = 4500,
        max_pages_per_batch: int = 8,
    ) -> list[list[int]]:
        """Create adaptive layout-aware batches respecting table boundaries and token budgets.

        Instead of fixed N-page slicing which cuts multi-page tables and notes in half,
        this groups contiguous pages up to target_tokens (~4,500) while keeping multi-page
        tables together.
        """
        if not requested_pages:
            return []

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = 0

        for idx, p in enumerate(requested_pages):
            if p < 0 or p >= doc.page_count or p >= len(doc.pages):
                continue

            page = doc.pages[p]
            text_tokens = len(page.raw_text) // 4
            table_tokens = sum(
                (len(" ".join(t.headers)) + sum(len(" ".join(r)) for r in t.rows)) // 4
                for t in page.tables
            )
            page_tokens = max(50, text_tokens + table_tokens)

            next_p = requested_pages[idx + 1] if idx + 1 < len(requested_pages) else None
            has_table = bool(page.tables)
            next_has_table = bool(doc.pages[next_p].tables) if (next_p is not None and next_p < len(doc.pages)) else False
            table_continuation = has_table and next_has_table

            tokens_exceeded = (current_tokens + page_tokens) > target_tokens
            pages_exceeded = len(current_batch) >= max_pages_per_batch

            if current_batch and (tokens_exceeded or pages_exceeded):
                if table_continuation and len(current_batch) < (max_pages_per_batch + 2) and (current_tokens + page_tokens) < (target_tokens * 1.3):
                    current_batch.append(p)
                    current_tokens += page_tokens
                else:
                    batches.append(current_batch)
                    current_batch = [p]
                    current_tokens = page_tokens
            else:
                current_batch.append(p)
                current_tokens += page_tokens

        if current_batch:
            batches.append(current_batch)

        return batches if batches else [requested_pages]

    async def aextract_and_verify(
        self,
        doc: DocumentData,
        page_num: int | None = None,
        pages: list[int] | None = None,
        batch_size: int = 4,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> ExtractedFacts:
        """Asynchronously extract and verify facts, running concurrently without blocking the event loop."""
        raw_items_with_page: list[tuple[RawFactItem, int | None]] = []
        model_used = ""
        total_attempts = 0

        if page_num is not None:
            if page_num < 0 or page_num >= doc.page_count:
                raise ValueError(f"Page {page_num} is out of bounds (document has {doc.page_count} pages)")
            requested_pages = [page_num]
            page_batches = [[page_num]]
        elif pages is not None:
            requested_pages = pages
            if batch_size != 4:
                page_batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
            else:
                page_batches = self._create_semantic_batches(doc, pages)
        else:
            requested_pages = list(range(doc.page_count))
            if batch_size != 4:
                page_batches = [requested_pages[i:i + batch_size] for i in range(0, len(requested_pages), batch_size)]
            else:
                page_batches = self._create_semantic_batches(doc, requested_pages)

        # Track skipped/empty pages for transparency and observability
        skipped_pages: list[int] = []
        for p in requested_pages:
            if p < len(doc.pages):
                page = doc.pages[p]
                if not page.raw_text.strip() and not page.tables:
                    skipped_pages.append(p)

        warnings: list[str] = list(doc.warnings)
        if skipped_pages:
            warnings.append(
                f"{len(skipped_pages)} of {len(requested_pages)} pages contained no extractable text or tables (scanned/image-only) and were skipped: pages {skipped_pages}"
            )

        use_parallel = self._is_real_client and len(page_batches) > 1 and page_num is None

        if use_parallel:
            logger.info(
                "Using non-blocking async parallel extraction for %d batches (max_concurrent=%d)",
                len(page_batches), self.max_concurrent,
            )
            try:
                results = await self._async_extract_all_batches(doc, page_batches, on_progress=on_progress)
                for items_with_page, m_used, attempts in results:
                    if not model_used and m_used:
                        model_used = m_used
                    total_attempts = max(total_attempts, attempts)
                    raw_items_with_page.extend(items_with_page)
            except Exception as e:
                logger.warning("Async extraction failed (%s), falling back to sequential", e)
                use_parallel = False
                raw_items_with_page = []

        if not use_parallel:
            pages_done = 0
            total_req_pages = len(requested_pages)
            for batch in page_batches:
                items_with_page, m_used, attempts = self._extract_page_batch(doc, batch)
                if not model_used:
                    model_used = m_used
                total_attempts = max(total_attempts, attempts)
                raw_items_with_page.extend(items_with_page)
                pages_done += len(batch)
                if on_progress:
                    on_progress(pages_done, total_req_pages, f"Extracted {pages_done} of {total_req_pages} pages")

        structured_facts = self._ground_facts(doc, raw_items_with_page, page_num)

        return ExtractedFacts(
            doc_id=doc.doc_id,
            page_number=page_num,
            facts=structured_facts,
            model_used=model_used,
            fallback_attempts=total_attempts,
            skipped_pages=skipped_pages,
            warnings=warnings,
        )

    def extract_and_verify(
        self,
        doc: DocumentData,
        page_num: int | None = None,
        pages: list[int] | None = None,
        batch_size: int = 4,
        use_async: bool = True,
    ) -> ExtractedFacts:
        """Synchronous wrapper for extract_and_verify."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(
                    asyncio.run,
                    self.aextract_and_verify(doc, page_num=page_num, pages=pages, batch_size=batch_size),
                ).result()
        else:
            return asyncio.run(
                self.aextract_and_verify(doc, page_num=page_num, pages=pages, batch_size=batch_size)
            )
