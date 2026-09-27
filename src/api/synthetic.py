"""Synthetic data API - generate sample CVs on demand."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.cli import generate

router = APIRouter()

_VALID_FORMATS = {"all", "txt", "csv", "pdf"}
_MIN_COUNT = 1
_MAX_COUNT = 50


class GenerateRequest(BaseModel):
    count: int = 20
    format: str = "all"  # pdf, txt, csv, all


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
    if not (_MIN_COUNT <= request.count <= _MAX_COUNT):
        raise HTTPException(
            status_code=400,
            detail=f"count must be between {_MIN_COUNT} and {_MAX_COUNT}, got {request.count}.",
        )

    output_dir = Path("sample_data").resolve()

    try:
        generate(request.count, output_dir, request.format, seed=0)
    except Exception as e:
        return GenerateResponse(
            success=False,
            count=0,
            output_dir=str(output_dir),
            pdf_count=0,
            txt_count=0,
            csv_count=0,
            message=f"Generation failed: {e}",
        )

    pdf_count = len(list((output_dir / "pdf").glob("*.pdf"))) if (output_dir / "pdf").exists() else 0
    txt_count = len(list((output_dir / "txt").glob("*.txt"))) if (output_dir / "txt").exists() else 0
    csv_count = 1 if (output_dir / "personas.csv").exists() else 0

    return GenerateResponse(
        success=True,
        count=request.count,
        output_dir=str(output_dir),
        pdf_count=pdf_count,
        txt_count=txt_count,
        csv_count=csv_count,
        message=f"Generated {request.count} personas with {pdf_count} PDFs, {txt_count} TXTs, {csv_count} CSV",
    )
