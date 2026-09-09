"""PDF ingestion — two-layer extraction:
  Layer 1: PyMuPDF for raw text index + word-level bboxes (fast, deterministic)
  Layer 2: Docling for structured tables + layout-aware reading order (ML-powered)

Docling (IBM Research, 60K+ GitHub stars) replaces pdfplumber for structured
extraction. It uses DocLayNet + Heron layout model for correct reading order,
and TableFormer for ML-powered table structure recovery (merged cells,
borderless tables, multi-page table continuity).

PyMuPDF is retained for Layer 1 because it provides:
  - 10x faster raw text extraction for quote verification
  - Word-level bounding boxes for PDF viewer highlighting
  - Ground-truth text index that is independent of the layout model
"""

import hashlib
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pymupdf

from app.models import BoundingBox, DocumentData, PageData, TableBlock, TextBlock

logger = logging.getLogger(__name__)

# Lazy-loaded Docling converter (downloads ~1GB of model weights on first run)
_docling_converter = None
_DOCLING_AVAILABLE = None


def _get_docling_converter():
    """Lazy-initialize the Docling DocumentConverter.

    First call downloads model weights (~1GB). Subsequent calls reuse the
    cached converter instance.
    """
    global _docling_converter, _DOCLING_AVAILABLE

    if _DOCLING_AVAILABLE is False:
        return None

    if _docling_converter is not None:
        return _docling_converter

    try:
        import torch
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
            AcceleratorOptions,
            AcceleratorDevice,
        )

        pipeline_options = PdfPipelineOptions()
        # For born-digital PDFs, bypass OCR to extract native digital text directly (10-50x faster)
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        # Auto-detect CUDA GPU (e.g. RTX 3060)
        if torch.cuda.is_available():
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                num_threads=4,
            )
            logger.info("Docling using CUDA GPU acceleration on %s", torch.cuda.get_device_name(0))
        else:
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=4,
            )
            logger.info("Docling using CPU acceleration (num_threads=4)")

        _docling_converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        _DOCLING_AVAILABLE = True
        logger.info("Docling converter initialized (TableFormer FAST mode, OCR=False)")
        return _docling_converter

    except ImportError:
        _DOCLING_AVAILABLE = False
        logger.warning(
            "Docling not installed — falling back to pdfplumber. "
            "Install with: pip install docling"
        )
        return None
    except Exception as e:
        _DOCLING_AVAILABLE = False
        logger.warning("Docling initialization failed: %s — falling back to pdfplumber", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(
    filepath: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> DocumentData:
    filepath = Path(filepath)
    doc_id = _make_doc_id(filepath)

    # Layer 1: PyMuPDF — fast raw text for quote verification + bboxes
    raw_index, page_count = _extract_raw_text_index(filepath)
    if progress_callback:
        progress_callback(0, page_count, f"Indexed {page_count} pages with PyMuPDF")

    scanned_pages = _detect_scanned_pages(raw_index)
    if scanned_pages and len(scanned_pages) == page_count:
        # Docling has OCR — try it before giving up
        converter = _get_docling_converter()
        if converter is None:
            raise ValueError(
                f"All {page_count} pages appear to be scanned images with no text layer. "
                "Install docling for OCR support: pip install docling"
            )

    header_footer_lines = _detect_header_footer(raw_index)

    # Layer 2: Try Docling first, fall back to pdfplumber
    converter = _get_docling_converter()
    if converter is not None:
        try:
            pages = _extract_structured_pages_docling(
                converter, filepath, raw_index, header_footer_lines, page_count, progress_callback
            )
            logger.info("PDF parsed with Docling (ML-powered layout + table recovery)")
        except Exception as e:
            logger.warning("Docling extraction failed: %s — falling back to pdfplumber", e)
            pages = _extract_structured_pages_pdfplumber(
                filepath, raw_index, header_footer_lines, progress_callback
            )
    else:
        pages = _extract_structured_pages_pdfplumber(
            filepath, raw_index, header_footer_lines, progress_callback
        )

    if progress_callback:
        progress_callback(page_count, page_count, "Document structure parsed successfully")

    warnings = []
    if scanned_pages and len(scanned_pages) < page_count:
        msg = f"{len(scanned_pages)} of {page_count} pages appear to be scanned images with no extractable text: pages {scanned_pages}"
        warnings.append(msg)
        logger.warning(msg)

    return DocumentData(
        doc_id=doc_id,
        filename=filepath.name,
        page_count=page_count,
        pages=pages,
        raw_text_index=raw_index,
        scanned_pages=scanned_pages,
        warnings=warnings,
    )


def verify_quote(doc: DocumentData, quote: str, page: int | None = None) -> dict:
    """Three-tier check: exact → whitespace-normalized → 80% word-sequence.
    Returns {found, match_type, page}.
    """
    if not quote or not quote.strip():
        return {"found": False, "match_type": None, "page": None}

    pages_to_check = [page] if page is not None else sorted(doc.raw_text_index.keys())

    for pg in pages_to_check:
        raw = doc.raw_text_index.get(pg, "")
        if not raw:
            continue

        if quote.strip() in raw:
            return {"found": True, "match_type": "exact", "page": pg}

        norm_quote = _normalize_whitespace(quote)
        norm_raw = _normalize_whitespace(raw)
        if norm_quote in norm_raw:
            return {"found": True, "match_type": "normalized", "page": pg}

        if _fuzzy_sequence_match(norm_quote, norm_raw, threshold=0.8):
            return {"found": True, "match_type": "fuzzy", "page": pg}

    return {"found": False, "match_type": None, "page": None}


# ---------------------------------------------------------------------------
# Layer 1: PyMuPDF — raw text index with coordinates
# ---------------------------------------------------------------------------

def _extract_raw_text_index(filepath: Path) -> tuple[dict[int, str], int]:
    """PyMuPDF is ~10x faster than pdfplumber for raw text — we use it
    as the ground-truth index that quote verification runs against."""
    index = {}
    with pymupdf.open(str(filepath)) as doc:
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            index[page_num] = page.get_text("text")
    return index, page_count


def _detect_scanned_pages(raw_index: dict[int, str]) -> list[int]:
    return [
        pg for pg, text in raw_index.items()
        if len(text.strip()) < 20  # <4 words = likely a scanned image
    ]


def get_word_bboxes(filepath: str | Path, page_num: int) -> list[dict]:
    """Word-level bboxes for PDF viewer highlighting."""
    with pymupdf.open(str(filepath)) as doc:
        if page_num >= len(doc):
            return []
        page = doc[page_num]
        words = page.get_text("words")
        return [
            {"word": w[4], "bbox": BoundingBox(x0=w[0], y0=w[1], x1=w[2], y1=w[3])}
            for w in words
        ]


# ---------------------------------------------------------------------------
# Header/footer detection
# ---------------------------------------------------------------------------

def _detect_header_footer(raw_index: dict[int, str], min_pages: int = 3) -> set[str]:
    """Lines appearing on 60%+ of pages (first/last 2 lines each) are
    almost certainly headers or footers."""
    if len(raw_index) < min_pages:
        return set()

    edge_lines = Counter()
    total_pages = len(raw_index)

    for text in raw_index.values():
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 4:
            continue

        candidates = lines[:2] + lines[-2:]
        for line in candidates:
            cleaned = re.sub(r"[\d\s\-–—|/]", "", line)
            if len(cleaned) < 3:  # pure page numbers: "12", "- 12 -"
                edge_lines[_normalize_whitespace(line)] += 1
                continue
            edge_lines[_normalize_whitespace(line)] += 1

    threshold = total_pages * 0.6
    return {line for line, count in edge_lines.items() if count >= threshold}


def _strip_header_footer(text: str, hf_lines: set[str]) -> str:
    if not hf_lines:
        return text
    out_lines = []
    for line in text.splitlines():
        if _normalize_whitespace(line) not in hf_lines:
            out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Layer 2a: Docling — ML-powered structured extraction (preferred)
# ---------------------------------------------------------------------------

def _extract_structured_pages_docling(
    converter,
    filepath: Path,
    raw_index: dict[int, str],
    hf_lines: set[str],
    page_count: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[PageData]:
    """Extract structured pages using Docling's ML-powered pipeline.

    Docling provides:
    - Correct reading order via Heron layout model (RT-DETRv2)
    - ML table detection via TableFormer (handles merged cells, borderless)
    - Multi-page table continuity
    - OCR for scanned pages

    For large PDFs, pages are processed in batches via page_range to control
    peak memory and enable progress reporting.  TableFormer quality is
    preserved — no tables are skipped.
    """
    from docling_core.types.doc.document import ContentLayer

    DOCLING_PAGE_BATCH_SIZE = 25  # pages per Docling batch

    # Build per-page collections
    page_tables: dict[int, list[TableBlock]] = {i: [] for i in range(page_count)}
    page_texts: dict[int, list[TextBlock]] = {i: [] for i in range(page_count)}

    # Process in page_range batches for memory control & progress reporting
    t_total_start = time.perf_counter()
    num_batches = (page_count + DOCLING_PAGE_BATCH_SIZE - 1) // DOCLING_PAGE_BATCH_SIZE

    for batch_idx in range(num_batches):
        batch_start = batch_idx * DOCLING_PAGE_BATCH_SIZE  # 0-indexed
        batch_end = min(batch_start + DOCLING_PAGE_BATCH_SIZE, page_count)
        # Docling page_range is 1-indexed inclusive
        page_range = (batch_start + 1, batch_end)

        t_batch = time.perf_counter()
        result = converter.convert(str(filepath), page_range=page_range)
        dt_batch = time.perf_counter() - t_batch
        logger.info(
            "Docling batch %d/%d (pages %d-%d): %.1fs (%.1f pages/sec)",
            batch_idx + 1, num_batches, batch_start, batch_end - 1,
            dt_batch, (batch_end - batch_start) / max(dt_batch, 0.001),
        )
        doc = result.document

        _merge_docling_result_into_pages(
            doc, page_tables, page_texts, page_count,
            hf_lines, batch_start,
        )
        if progress_callback:
            progress_callback(
                batch_end,
                page_count,
                f"Parsed layout for page {batch_end} of {page_count}",
            )

    dt_total = time.perf_counter() - t_total_start
    logger.info(
        "Docling parsing complete: %d pages in %.1fs (%.1f pages/sec, %d tables found)",
        page_count, dt_total, page_count / max(dt_total, 0.001),
        sum(len(t) for t in page_tables.values()),
    )

    # Assemble PageData objects
    pages = []
    for page_num in range(page_count):
        pages.append(PageData(
            page_number=page_num,
            raw_text=raw_index.get(page_num, ""),
            text_blocks=page_texts.get(page_num, []),
            tables=page_tables.get(page_num, []),
        ))

    return pages


def _merge_docling_result_into_pages(
    doc,
    page_tables: dict[int, list[TableBlock]],
    page_texts: dict[int, list[TextBlock]],
    page_count: int,
    hf_lines: set[str],
    batch_page_offset: int,
) -> None:
    """Merge a single Docling batch result into the page-level collections.

    batch_page_offset is the 0-indexed page number of the first page in this batch.
    Docling page_range returns 1-indexed page numbers relative to the original document.
    """
    # Process all items in reading order
    for item, level in doc.iterate_items():
        # Get page number from provenance (Docling uses 1-indexed pages)
        if not hasattr(item, 'prov') or not item.prov:
            continue

        page_no = item.prov[0].page_no - 1  # Convert to 0-indexed

        if page_no < 0 or page_no >= page_count:
            continue

        # Extract bounding box if available
        bbox = None
        if hasattr(item.prov[0], 'bbox') and item.prov[0].bbox is not None:
            b = item.prov[0].bbox
            # Docling bbox format: check if it has l,t,r,b attributes
            if hasattr(b, 'l'):
                bbox = BoundingBox(x0=b.l, y0=b.t, x1=b.r, y1=b.b)
            elif hasattr(b, 'x0'):
                bbox = BoundingBox(x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1)

        # Check if this is a table item
        item_type = type(item).__name__

        if item_type == 'TableItem' or (hasattr(item, 'export_to_dataframe')):
            # Table extraction
            try:
                try:
                    df = item.export_to_dataframe(doc=doc)
                except TypeError:
                    df = item.export_to_dataframe()
                if df.empty or len(df) < 1:
                    continue

                headers = [str(c) for c in df.columns.tolist()]
                rows = [[str(cell) for cell in row] for row in df.values.tolist()]

                # Clean cells
                headers = [_clean_cell(h) for h in headers]
                rows = [[_clean_cell(c) for c in row] for row in rows]

                if all(h == "" for h in headers):
                    continue

                page_tables[page_no].append(TableBlock(
                    headers=headers,
                    rows=rows,
                    page=page_no,
                    bbox=bbox,
                ))
            except Exception as e:
                logger.debug("Failed to extract table on page %d: %s", page_no, e)

        elif hasattr(item, 'text') and item.text:
            # Text block extraction
            text = item.text.strip()
            if not text or len(text) < 5:
                continue

            # Strip header/footer lines
            text = _strip_header_footer(text, hf_lines)
            if not text.strip():
                continue

            page_texts[page_no].append(TextBlock(
                content=text,
                page=page_no,
                bbox=bbox,
            ))


# ---------------------------------------------------------------------------
# Layer 2b: pdfplumber — fallback structured extraction
# ---------------------------------------------------------------------------

def _extract_structured_pages_pdfplumber(
    filepath: Path,
    raw_index: dict[int, str],
    hf_lines: set[str],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[PageData]:
    """Fallback extraction using pdfplumber when Docling is unavailable."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("Neither docling nor pdfplumber available — using raw text only")
        return [
            PageData(page_number=pg, raw_text=text)
            for pg, text in sorted(raw_index.items())
        ]

    with pdfplumber.open(str(filepath)) as pdf:
        pages = []
        total_pgs = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            pages.append(_process_single_page_pdfplumber(page, page_num, raw_index, hf_lines))
            if progress_callback:
                progress_callback(
                    page_num + 1,
                    total_pgs,
                    f"Parsed page {page_num + 1} of {total_pgs}",
                )
        return pages


def _process_single_page_pdfplumber(page, page_num, raw_index, hf_lines) -> PageData:
    tables = _extract_tables_pdfplumber(page, page_num)
    text_blocks = _extract_text_blocks_pdfplumber(page, page_num, tables, hf_lines)
    raw = raw_index.get(page_num, "")

    return PageData(
        page_number=page_num,
        raw_text=raw,
        text_blocks=text_blocks,
        tables=tables,
    )


def _extract_tables_pdfplumber(page, page_num: int) -> list[TableBlock]:
    extracted = []
    try:
        # bordered tables first, then text-alignment fallback for borderless
        table_settings = {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
        }
        tables = page.find_tables(table_settings=table_settings)

        if not tables:
            table_settings = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "min_words_vertical": 3,
                "min_words_horizontal": 2,
            }
            tables = page.find_tables(table_settings=table_settings)
    except Exception:
        return extracted

    for table in tables:
        try:
            raw_rows = table.extract()
        except Exception:
            continue

        if not raw_rows or len(raw_rows) < 2:
            continue

        headers = [_clean_cell(c) for c in raw_rows[0]]

        if all(h == "" for h in headers):
            continue

        rows = [[_clean_cell(c) for c in row] for row in raw_rows[1:]]

        bbox = None
        if table.bbox:
            bbox = BoundingBox(
                x0=table.bbox[0], y0=table.bbox[1],
                x1=table.bbox[2], y1=table.bbox[3],
            )

        extracted.append(TableBlock(
            headers=headers, rows=rows, page=page_num, bbox=bbox,
        ))

    return extracted


def _extract_text_blocks_pdfplumber(
    page, page_num: int, tables: list[TableBlock], hf_lines: set[str],
) -> list[TextBlock]:
    table_bboxes = [t.bbox for t in tables if t.bbox]

    # crop out table regions so we don't double-extract their text
    filtered_page = page
    for tb in table_bboxes:
        try:
            filtered_page = filtered_page.outside_bbox(
                (tb.x0, tb.y0, tb.x1, tb.y1)
            )
        except Exception:
            pass

    # layout=True respects visual column positions instead of raw draw order
    try:
        raw = filtered_page.extract_text(
            layout=True,
            x_density=7.25,
            y_density=13,
        ) or ""
    except Exception:
        raw = filtered_page.extract_text(x_tolerance=2, y_tolerance=2) or ""

    if not raw.strip():
        return []

    raw = _strip_header_footer(raw, hf_lines)
    paragraphs = _split_paragraphs(raw)
    blocks = []
    for para in paragraphs:
        text = _collapse_layout_whitespace(para.strip())
        if not text or len(text) < 10:
            continue
        blocks.append(TextBlock(content=text, page=page_num))

    return blocks


# ---------------------------------------------------------------------------
# Chunking — group blocks into LLM-friendly chunks
# ---------------------------------------------------------------------------

def chunk_document(doc: DocumentData, max_tokens: int = 2500) -> list[dict]:
    """Group blocks into chunks under max_tokens. Tables stay isolated."""
    chunks = []

    for page in doc.pages:

        for table in page.tables:
            table_md = _table_to_markdown(table)
            if table_md.strip():
                chunks.append({
                    "content": table_md,
                    "pages": [page.page_number],
                    "chunk_type": "table",
                    "doc_id": doc.doc_id,
                    "doc_filename": doc.filename,
                })

        buffer = []
        buffer_len = 0

        for block in page.text_blocks:
            block_len = _estimate_tokens(block.content)

            if buffer and (buffer_len + block_len) > max_tokens:
                chunks.append({
                    "content": "\n\n".join(buffer),
                    "pages": [page.page_number],
                    "chunk_type": "text",
                    "doc_id": doc.doc_id,
                    "doc_filename": doc.filename,
                })
                buffer = []
                buffer_len = 0

            buffer.append(block.content)
            buffer_len += block_len

        if buffer:
            chunks.append({
                "content": "\n\n".join(buffer),
                "pages": [page.page_number],
                "chunk_type": "text",
                "doc_id": doc.doc_id,
                "doc_filename": doc.filename,
            })

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_id(filepath: Path) -> str:
    """Hash first 8KB + size — fast dedup without reading the whole file."""
    size = filepath.stat().st_size
    with open(filepath, "rb") as f:
        head = f.read(8192)
    key = f"{size}:{hashlib.md5(head).hexdigest()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _clean_cell(cell) -> str:
    if cell is None:
        return ""
    return str(cell).strip().replace("\n", " ")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _collapse_layout_whitespace(text: str) -> str:
    """layout=True pads with spaces to mimic visual columns — collapse them."""

    lines = []
    for line in text.splitlines():
        collapsed = re.sub(r"  {2,}", "  ", line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def _split_paragraphs(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text)


def _table_to_markdown(table: TableBlock) -> str:
    if not table.headers:
        return ""
    lines = [f"[Table on page {table.page}]"]
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
    for row in table.rows:
        padded = row + [""] * (len(table.headers) - len(row))
        lines.append("| " + " | ".join(padded[:len(table.headers)]) + " |")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _fuzzy_sequence_match(needle: str, haystack: str, threshold: float = 0.8) -> bool:
    """Last-resort match — catches LLM-rephrased quotes where 80%+ of
    words still appear in the same order in the source text."""
    needle_words = needle.split()
    if not needle_words:
        return False

    haystack_words = haystack.split()
    matched = 0
    hay_idx = 0

    for nw in needle_words:
        while hay_idx < len(haystack_words):
            if haystack_words[hay_idx] == nw:
                matched += 1
                hay_idx += 1
                break
            hay_idx += 1

    return (matched / len(needle_words)) >= threshold
