"""Synthetic data API — generate sample CVs on demand."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings

router = APIRouter()


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
    import subprocess
    import sys

    output_dir = str(settings.upload_dir.parent / "sample_data")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.data.cli",
            "--count", str(request.count),
            "--output", output_dir,
            "--format", request.format,
        ],
        capture_output=True,
        text=True,
        cwd=str(settings.upload_dir.parent),
    )

    if result.returncode != 0:
        return GenerateResponse(
            success=False,
            count=0,
            output_dir=output_dir,
            pdf_count=0,
            txt_count=0,
            csv_count=0,
            message=f"Generation failed: {result.stderr[:500]}",
        )

    from pathlib import Path
    out = Path(output_dir)
    pdf_count = len(list((out / "pdf").glob("*.pdf"))) if (out / "pdf").exists() else 0
    txt_count = len(list((out / "txt").glob("*.txt"))) if (out / "txt").exists() else 0
    csv_count = 1 if (out / "personas.csv").exists() else 0

    return GenerateResponse(
        success=True,
        count=request.count,
        output_dir=output_dir,
        pdf_count=pdf_count,
        txt_count=txt_count,
        csv_count=csv_count,
        message=f"Generated {request.count} personas with {pdf_count} PDFs, {txt_count} TXTs, {csv_count} CSV",
    )
