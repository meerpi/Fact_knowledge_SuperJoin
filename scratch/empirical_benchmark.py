"""Empirical benchmark comparing Docling vs pdfplumber on Delhivery documents."""

import json
import time
from pathlib import Path
import pdfplumber
import pymupdf

TEST_CASES = [
    {
        "name": "Q4 FY24 Earnings Presentation - Financial Summary & KPIs",
        "file": "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        "pages": [3, 9, 13],  # Page 3: Financial Summary; Page 9: Network/volume cards; Page 13: Financial statement
        "descriptions": {
            3: "Slide banner & multi-column financial cards",
            9: "Grid of KPI metric cards (volume, network reach)",
            13: "Adjusted EBITDA reconciliation table with margin footnotes",
        },
    },
    {
        "name": "Prospectus 2022 - Restated Financial Information",
        "file": "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
        "pages": [10, 11],  # Complex multi-year financial statements with restated figures
        "descriptions": {
            10: "IPO Shareholder table with merged headers & multiple columns",
            11: "Restated Consolidated Balance Sheet (multi-period financial columns)",
        },
    },
    {
        "name": "Annual Report FY24 - Multi-column Layout & Highlights",
        "file": "starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "pages": [0, 2],  # Page 0: Header navigation tabs; Page 2: Multi-column overview
        "descriptions": {
            0: "Page header with navigation tab strip",
            2: "Multi-column statutory narrative and operational overview",
        },
    },
]


def evaluate_pdfplumber(pdf_path: str, page_indices: list[int]) -> dict:
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        for p_idx in page_indices:
            t0 = time.time()
            page = pdf.pages[p_idx]
            raw_text = page.extract_text() or ""
            tables = page.extract_tables() or []
            duration = time.time() - t0

            # Analyze table quality
            table_stats = []
            for t_i, tbl in enumerate(tables):
                n_rows = len(tbl)
                n_cols = max(len(r) for r in tbl) if tbl else 0
                null_cells = sum(row.count(None) + row.count("") for row in tbl)
                total_cells = n_rows * n_cols if n_cols > 0 else 1
                null_ratio = null_cells / total_cells
                
                # Check for fragmentation signs: 1x1, 1x2 or single row/column tables
                is_fragment = (n_rows <= 1 and n_cols <= 2) or (total_cells <= 2)

                table_stats.append({
                    "table_id": t_i,
                    "rows": n_rows,
                    "cols": n_cols,
                    "null_cell_ratio": round(null_ratio, 3),
                    "is_likely_fragment": is_fragment,
                    "header_sample": tbl[0] if tbl else [],
                    "row_sample": tbl[1] if len(tbl) > 1 else [],
                })

            results[p_idx] = {
                "duration_sec": round(duration, 4),
                "text_length": len(raw_text),
                "text_snippet": raw_text[:300].replace("\n", " "),
                "table_count": len(tables),
                "fragment_table_count": sum(1 for t in table_stats if t["is_likely_fragment"]),
                "tables": table_stats,
            }
    return results


def slice_pdf_pages(src_path: str, page_indices: list[int], out_path: str):
    """Extract specific pages into a temporary PDF for focused page evaluation."""
    src = pymupdf.open(src_path)
    dst = pymupdf.open()
    for idx in page_indices:
        dst.insert_pdf(src, from_page=idx, to_page=idx)
    dst.save(out_path)
    dst.close()
    src.close()


def evaluate_docling_sliced(converter, pdf_path: str, page_indices: list[int]) -> dict:
    results = {}
    import tempfile

    for original_idx in page_indices:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        slice_pdf_pages(pdf_path, [original_idx], tmp_path)

        t0 = time.time()
        conv_result = converter.convert(tmp_path)
        duration = time.time() - t0
        doc = conv_result.document

        page_tables = []
        for tbl in doc.tables:
            try:
                try:
                    df = tbl.export_to_dataframe(doc=doc)
                except TypeError:
                    df = tbl.export_to_dataframe()

                n_rows, n_cols = df.shape
                null_ratio = df.isna().sum().sum() / (n_rows * n_cols) if (n_rows * n_cols) > 0 else 0
                is_fragment = (n_rows <= 1 and n_cols <= 2) or ((n_rows * n_cols) <= 2)

                page_tables.append({
                    "rows": n_rows,
                    "cols": n_cols,
                    "null_cell_ratio": round(null_ratio, 3),
                    "is_likely_fragment": is_fragment,
                    "headers": [str(c) for c in df.columns],
                    "first_row": {str(k): str(v) for k, v in df.iloc[0].to_dict().items()} if n_rows > 0 else {},
                })
            except Exception as e:
                page_tables.append({"error": str(e)})

        page_texts = [txt.text for txt in doc.texts]
        joined_text = " ".join(page_texts)
        try:
            md_snippet = doc.export_to_markdown()[:400].replace("\n", " ")
        except Exception:
            md_snippet = ""

        results[original_idx] = {
            "duration_sec": round(duration, 4),
            "text_length": len(joined_text),
            "text_snippet": joined_text[:300].replace("\n", " "),
            "markdown_snippet": md_snippet,
            "table_count": len(page_tables),
            "fragment_table_count": sum(1 for t in page_tables if t.get("is_likely_fragment", False)),
            "tables": page_tables,
        }

        Path(tmp_path).unlink(missing_ok=True)

    return results


def main():
    import torch
    from app.pdf_parser import _get_docling_converter

    print("=" * 80)
    print("EMPIRICAL BENCHMARK: DOCLING VS PDFPLUMBER ON DELHIVERY STARTER DATASETS")
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"Hardware Acceleration: CUDA={cuda_avail} ({device_name})")
    print("=" * 80)

    converter = _get_docling_converter()
    if converter is None:
        print("ERROR: Docling converter failed to initialize.")
        return

    full_report = {
        "device": device_name,
        "cuda_available": cuda_avail,
        "test_cases": {},
    }

    for tc in TEST_CASES:
        name = tc["name"]
        file_path = tc["file"]
        pages = tc["pages"]
        descriptions = tc.get("descriptions", {})

        print(f"\n==================================================")
        print(f"Testing: {name}")
        print(f"File: {file_path}")
        print(f"Target Pages: {pages}")
        print(f"==================================================")

        print("\n[1/2] Running pdfplumber extraction...")
        plumber_res = evaluate_pdfplumber(file_path, pages)
        for pg, pdata in plumber_res.items():
            print(f"  Page {pg} ({descriptions.get(pg, '')}): {pdata['table_count']} tables detected ({pdata['fragment_table_count']} fragments) in {pdata['duration_sec']:.3f}s")

        print("\n[2/2] Running Docling ML extraction...")
        docling_res = evaluate_docling_sliced(converter, file_path, pages)
        for pg, ddata in docling_res.items():
            print(f"  Page {pg} ({descriptions.get(pg, '')}): {ddata['table_count']} tables detected ({ddata['fragment_table_count']} fragments) in {ddata['duration_sec']:.3f}s")

        full_report["test_cases"][name] = {
            "file": file_path,
            "pages": pages,
            "descriptions": descriptions,
            "pdfplumber": plumber_res,
            "docling": docling_res,
        }

    out_file = Path("scratch/benchmark_results.json")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(full_report, indent=2))
    print(f"\n==================================================")
    print(f"Benchmark completed! Full results saved to {out_file}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
