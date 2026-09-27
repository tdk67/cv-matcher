"""Text extraction from various document formats.

Supported: PDF, TXT, CSV, Excel (.xlsx)
Each extractor returns a list of text blocks with metadata.

Memory safety: the 50 MB upload cap bounds the input, but a small file can
still expand to gigabytes of text (a zip-bomb xlsx, a 5000-page PDF, a CSV
with millions of rows). Every extractor therefore accumulates at most
`settings.max_extracted_chars` characters and bails out of its loops as
soon as the cap is reached - extraction never materializes the full
unbounded text in memory (F4-06).
"""

import csv
import hashlib
import io
from pathlib import Path

from pypdf import PdfReader

from src.config import settings


class _TextBudget:
    """Accumulate extracted text with a hard character budget.

    `append` returns False once the budget is exhausted so callers can stop
    early instead of grinding through every page/row of a hostile file.
    """

    __slots__ = ("limit", "parts", "size", "truncated")

    def __init__(self, limit: int | None = None):
        self.limit = settings.max_extracted_chars if limit is None else limit
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def append(self, text: str, sep: str = "\n\n") -> bool:
        if self.limit <= 0:
            return False
        if self.truncated:
            return False
        # Account for the separator that would be inserted before this text
        # (when parts already exist) so the TOTAL stays <= limit.
        pending_sep = len(sep) if self.parts else 0
        if self.size + pending_sep + len(text) > self.limit:
            # Partial append up to the budget, then stop. Keeps the text
            # deterministic for hashing while bounding memory.
            room = self.limit - self.size
            if room > 0:
                self.parts.append(text[:room])
                self.size = self.limit
            self.truncated = True
            return False
        if self.parts:
            self.parts.append(sep)
            self.size += len(sep)
        self.parts.append(text)
        self.size += len(text)
        return True

    def value(self) -> str:
        return "".join(self.parts)


def extract_text(file_path: Path) -> dict:
    """Extract text from a document. Returns metadata + text content.

    Returns:
        {
            "filename": str,
            "doc_id": str,        # SHA256 of filename + content
            "format": str,        # pdf, txt, csv, xlsx
            "text": str,          # full extracted text (capped)
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
    budget = _TextBudget()
    warnings = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            if not budget.append(text.strip()):
                warnings.append(f"Extraction stopped on page {i+1}: text exceeded the character cap.")
                break
        else:
            warnings.append(f"Page {i+1}: no text extracted (may be scanned/image-based)")

    return {
        "text": budget.value(),
        "page_count": len(reader.pages),
        "warnings": warnings,
    }


def _extract_text(file_path: Path) -> dict:
    """Read plain text file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > settings.max_extracted_chars > 0
    text = text[: settings.max_extracted_chars] if settings.max_extracted_chars > 0 else ""
    warnings = ["Extracted text exceeds the character cap; truncated."] if truncated else []
    return {
        "text": text,
        "page_count": 1,
        "warnings": warnings,
    }


def _extract_csv(file_path: Path) -> dict:
    """Extract text from CSV. Each row becomes a text block."""
    budget = _TextBudget()
    warnings = []
    row_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert row to readable text
            parts = []
            for key, value in row.items():
                if value and str(value).strip():
                    parts.append(f"{key}: {value}")
            if parts:
                joined = "\n".join(parts)
                if not budget.append(joined):
                    row_count += 1
                    warnings.append(
                        f"Extraction stopped at row {row_count}: text exceeded the character cap."
                    )
                    break
                row_count += 1

    if row_count == 0 and not budget.value():
        warnings.append("CSV file is empty or has no data rows")

    return {
        "text": budget.value(),
        "page_count": 1,
        "warnings": warnings,
    }


def _extract_excel(file_path: Path) -> dict:
    """Extract text from Excel (.xlsx). Each sheet's data becomes text."""
    from openpyxl import load_workbook

    wb = load_workbook(str(file_path), read_only=True, data_only=True)
    try:
        sheet_names = list(wb.sheetnames)
        budget = _TextBudget()
        warnings = []
        capped = False

        for sheet_name in sheet_names:
            ws = wb[sheet_name]
            header = f"[Sheet: {sheet_name}]\n"
            budget.append(header)
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    joined = " | ".join(cells)
                    if not budget.append(joined, sep="\n"):
                        capped = True
                        break
            if capped or budget.truncated:
                warnings.append("Extraction stopped: text exceeded the character cap.")
                break

        if not budget.value() and not any(warnings):
            warnings.append("Excel file contains no data")

        return {
            "text": budget.value(),
            "page_count": len(sheet_names),
            "warnings": warnings,
        }
    finally:
        wb.close()


def _compute_doc_id(filename: str, content: str) -> str:
    """Generate a stable document ID from filename + content hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    name_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    return f"{name_hash}_{content_hash}"
