"""Text extraction from various document formats.

Supported: PDF, TXT, CSV, Excel (.xlsx)
Each extractor returns a list of text blocks with metadata.
"""

import csv
import hashlib
import io
from pathlib import Path

from pypdf import PdfReader


def extract_text(file_path: Path) -> dict:
    """Extract text from a document. Returns metadata + text content.

    Returns:
        {
            "filename": str,
            "doc_id": str,        # SHA256 of filename + content
            "format": str,        # pdf, txt, csv, xlsx
            "text": str,          # full extracted text
            "page_count": int,    # for PDFs, number of pages
            "warnings": list[str],
        }
    """
    suffix = file_path.suffix.lower()
    extractors = {
        ".pdf": _extract_pdf,
        ".txt": _extract_text,
        ".csv": _extract_csv,
        ".xlsx": _extract_excel,
    }

    extractor = extractors.get(suffix)
    if not extractor:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {list(extractors.keys())}")

    result = extractor(file_path)
    result["filename"] = file_path.name
    result["doc_id"] = _compute_doc_id(file_path.name, result["text"])
    result["format"] = suffix.lstrip(".")
    return result


def _extract_pdf(file_path: Path) -> dict:
    """Extract text from PDF. Handles text-based PDFs."""
    reader = PdfReader(str(file_path))
    pages = []
    warnings = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
        else:
            warnings.append(f"Page {i+1}: no text extracted (may be scanned/image-based)")

    full_text = "\n\n".join(pages)
    return {
        "text": full_text,
        "page_count": len(reader.pages),
        "warnings": warnings,
    }


def _extract_text(file_path: Path) -> dict:
    """Read plain text file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return {
        "text": text,
        "page_count": 1,
        "warnings": [],
    }


def _extract_csv(file_path: Path) -> dict:
    """Extract text from CSV. Each row becomes a text block."""
    rows = []
    warnings = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert row to readable text
            parts = []
            for key, value in row.items():
                if value and str(value).strip():
                    parts.append(f"{key}: {value}")
            if parts:
                rows.append("\n".join(parts))

    if not rows:
        warnings.append("CSV file is empty or has no data rows")

    full_text = "\n\n".join(rows)
    return {
        "text": full_text,
        "page_count": 1,
        "warnings": warnings,
    }


def _extract_excel(file_path: Path) -> dict:
    """Extract text from Excel (.xlsx). Each sheet's data becomes text."""
    from openpyxl import load_workbook

    wb = load_workbook(str(file_path), read_only=True, data_only=True)
    try:
        sheet_names = list(wb.sheetnames)
        sheets_text = []
        warnings = []

        for sheet_name in sheet_names:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                sheets_text.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))

        if not sheets_text:
            warnings.append("Excel file contains no data")

        full_text = "\n\n".join(sheets_text)
        return {
            "text": full_text,
            "page_count": len(sheet_names),
            "warnings": warnings,
        }
    finally:
        # sheet_names was read before close, so no post-close attribute access.
        wb.close()


def _compute_doc_id(filename: str, content: str) -> str:
    """Generate a stable document ID from filename + content hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    name_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    return f"{name_hash}_{content_hash}"
