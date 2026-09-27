"""Synthetic CV generator - CLI entry point.

Usage:
    python -m src.data.cli --count 20 --output ./sample_data
    python -m src.data.cli --count 10 --output ./sample_data --format csv
    python -m src.data.cli --count 20 --seed 123

Also exposes :func:`generate` as a plain function so the FastAPI synthetic
endpoint and pytest conftest can reuse this code path in-process (the old
`subprocess` indirection was removed in review Round A, F-01).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.data.generator import generate_personas

VALID_FORMATS = {"all", "txt", "csv", "pdf"}


def _write_csv(personas, csv_path: Path) -> None:
    rows = [p.to_dict() for p in personas]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_pdfs(personas, pdf_dir: Path) -> tuple[int, int]:
    """Write one PDF per persona using fpdf2 (latin-1 safe).

    Returns (written, bytes_written).
    """
    from fpdf import FPDF

    pdf_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, p in enumerate(personas, start=1):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _latin1(p.name))
        pdf.ln(12)
        pdf.set_font("Helvetica", "", 11)
        for line in p.to_text().splitlines():
            if line.startswith("#"):
                continue
            pdf.multi_cell(0, 6, _latin1(line))
            pdf.ln(1)
        out = pdf_dir / f"persona_{idx:02d}.pdf"
        pdf.output(str(out))
        written += 1
    return written, 0


def _latin1(text: str) -> str:
    """Coerce to latin-1-safe text for fpdf (drops exotic unicode chars)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate(count: int, output: Path | str, fmt: str = "all", seed: int = 0) -> dict:
    """Generate `count` synthetic personas under `output`.

    Args:
        count: Number of personas (>= 1).
        output: Destination directory (created if missing).
        fmt: One of "all" | "txt" | "csv" | "pdf".
        seed: PRNG seed for determinism.

    Returns:
        A small stats dict: {count, output_dir, txt_count, pdf_count, csv_count}.
    """
    if fmt not in VALID_FORMATS:
        raise ValueError(f"Invalid format: {fmt!r}. Must be one of {sorted(VALID_FORMATS)}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    personas = generate_personas(count, seed=seed)

    txt_count = pdf_count = csv_count = 0

    if fmt in ("all", "txt"):
        txt_dir = output_path / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        for idx, p in enumerate(personas, start=1):
            (txt_dir / f"persona_{idx:02d}.txt").write_text(p.to_text(), encoding="utf-8")
        txt_count = count

    if fmt in ("all", "pdf"):
        pdf_count, _ = _write_pdfs(personas, output_path / "pdf")

    if fmt in ("all", "csv"):
        _write_csv(personas, output_path / "personas.csv")
        csv_count = 1

    return {
        "count": count,
        "output_dir": str(output_path),
        "txt_count": txt_count,
        "pdf_count": pdf_count,
        "csv_count": csv_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic CV personas (no real data).")
    parser.add_argument("--count", type=int, default=20, help="Number of personas (1-50)")
    parser.add_argument("--output", type=str, default="./sample_data", help="Output directory")
    parser.add_argument(
        "--format", dest="fmt", choices=sorted(VALID_FORMATS), default="all",
        help="Output formats (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for determinism")
    args = parser.parse_args(argv)

    if not (1 <= args.count <= 50):
        print(f"error: count must be between 1 and 50, got {args.count}", file=sys.stderr)
        return 2

    try:
        stats = generate(args.count, args.output, args.fmt, args.seed)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catches and reports
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {stats['count']} personas -> {stats['output_dir']} "
        f"(txt={stats['txt_count']}, pdf={stats['pdf_count']}, csv={stats['csv_count']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())