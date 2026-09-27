"""Synthetic data API - generate sample CVs on demand."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.data.cli import generate

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_FORMATS = {"all", "txt", "csv", "pdf"}
_MIN_COUNT = 1
_MAX_COUNT = 50

# CWD-relative output directory for generated personas. Module-level so tests
# can point generation at a tmp dir (the endpoint itself stays unchanged).
_OUTPUT_DIR = Path("sample_data").resolve()


class GenerateRequest(BaseModel):
    count: int = Field(default=20, ge=_MIN_COUNT, le=_MAX_COUNT)
    format: str = "all"  # pdf, txt, csv, all
    seed: int | None = None


class GenerateResponse(BaseModel):
    success: bool
    count: int
    output_dir: str
    pdf_count: int
    txt_count: int
    csv_count: int
    message: str


@router.post("/generate", response_model=GenerateResponse)
def generate_synthetic_data(request: GenerateRequest):
    """Generate synthetic CV data for testing."""
    if request.format not in _VALID_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {request.format!r}. Must be one of {sorted(_VALID_FORMATS)}.",
        )

    output_dir = _OUTPUT_DIR

    try:
        # Clean the target directory first so earlier runs' stale files
        # (e.g. higher-numbered personas after a smaller regeneration) are
        # not counted or later ingested by upstream ingestion paths.
        for sub in ("txt", "pdf"):
            d = output_dir / sub
            if d.exists():
                # Iterate each glob unconditionally: Path.glob returns a
                # generator, which is ALWAYS truthy even when it matches
                # nothing - `a or b` short-circuits on the first glob and the
                # second pattern is never iterated.
                for pattern in ("persona_*.txt", "persona_*.pdf"):
                    for p in d.glob(pattern):
                        p.unlink()
        csv_path = output_dir / "personas.csv"
        if csv_path.exists():
            csv_path.unlink()

        stats = generate(request.count, output_dir, request.format, seed=request.seed or 0)
    except Exception as e:
        logger.exception("Synthetic generation failed")
        return GenerateResponse(
            success=False,
            count=0,
            output_dir=str(output_dir),
            pdf_count=0,
            txt_count=0,
            csv_count=0,
            message="Generation failed.",
        )

    return GenerateResponse(
        success=True,
        count=stats["count"],
        output_dir=str(output_dir),
        pdf_count=stats["pdf_count"],
        txt_count=stats["txt_count"],
        csv_count=stats["csv_count"],
        message=(
            f"Generated {stats['count']} personas with "
            f"{stats['pdf_count']} PDFs, {stats['txt_count']} TXTs, {stats['csv_count']} CSV"
        ),
    )