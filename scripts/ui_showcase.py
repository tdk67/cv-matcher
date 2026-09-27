#!/usr/bin/env python3
"""CV Matcher — automated UI showcase (Playwright).

Drives the real running app (Streamlit frontend + FastAPI backend),
takes full-page screenshots of every page and key interaction, and
assembles them into a single PDF report for capstone submission.

Design rules for this report:
  * NO personal data — the app is fed generated synthetic personas only.
  * NO repo URL / owner / deployment URL — screenshots of the running app
    show the app title in browser chrome, but the report text is neutral.

Requires:
  * The app running at SHOWCASE_FRONTEND (default http://127.0.0.1:8501)
  * playwright (pip install playwright) + browsers (playwright install chromium)
  * reportlab for PDF assembly (pip install reportlab)

Usage:
    python scripts/ui_showcase.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
SHOTS = REPO / "artifacts" / "showcase"
SHOTS.mkdir(parents=True, exist_ok=True)

FRONTEND = os.getenv("SHOWCASE_FRONTEND", "http://127.0.0.1:8501")
TIMEOUT = int(os.getenv("SHOWCASE_TIMEOUT", "90000"))

# A neutral, real question that produces good matching on synthetic data.
QUERY_TEXT = "Find a Java developer with Spring Boot experience"
QUESTION2_TEXT = "Who has experience with Docker and Kubernetes?"

CAPTIONS: list[tuple[str, str]] = []


def snap(page, name: str, caption: str, full: bool = True) -> str:
    """Screenshot + record caption for the PDF assembly step."""
    path = SHOTS / f"{name}.png"
    page.wait_for_timeout(1500)  # let Streamlit rerun settle
    page.screenshot(path=str(path), full_page=full)
    CAPTIONS.append((name, caption))
    print(f"  [shot] {path.name}")
    return str(path)


def goto(page, path: str = "/"):
    page.goto(FRONTEND + path, wait_until="networkidle", timeout=TIMEOUT)
    page.wait_for_timeout(2500)


def click_sidebar(page, label: str) -> None:
    """Click a sidebar radio page. Streamlit renders radios via aria-checked
    state; we target the label element directly."""
    try:
        page.get_by_label(label).first.click(timeout=15000)
    except Exception:
        # fallback: the sidebar radio items are buttons with the label text
        page.locator(f'label:has-text("{label}")').first.click(timeout=15000)
    page.wait_for_timeout(3000)


def type_textarea(page, placeholder_substr: str, text: str) -> None:
    """Streamlit text_area renders as a <textarea> with the placeholder in an
    aria-label; find it and type."""
    ta = None
    for candidate in page.locator("textarea").all():
        label = candidate.get_attribute("aria-label") or ""
        if placeholder_substr.lower() in label.lower():
            ta = candidate
            break
    if ta is None:
        # fallback: first textarea on the page
        ta = page.locator("textarea").first
    ta.click()
    ta.fill(text)
    page.wait_for_timeout(800)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1500, "height": 950},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.set_default_timeout(TIMEOUT)

        # ---------- 1) Home ----------
        goto(page, "/")
        snap(page, "01-home", "Home — the landing page presents the agentic RAG workflow, "
                           "keyword search, upload and synthetic-data generation.")

        # ---------- 2) Query page (first question) ----------
        click_sidebar(page, "Query")
        snap(page, "02-query-empty", "Query page — ask a natural-language question about the "
                                     "CV knowledge base.")
        type_textarea(page, "Ask a question", QUERY_TEXT)
        page.get_by_role("button", name="🔍 Search").first.click(timeout=15000)
        page.wait_for_timeout(30000)  # agentic pipeline (planner->retriever->responder->validator)
        snap(page, "03-query-answer", "Answer with ranked matches — the Responder agent returns "
                                      "cited candidates with match scores and evidence.")

        # ---------- 3) Documents ----------
        click_sidebar(page, "Documents")
        snap(page, "04-documents", "Documents page — shows ingested CVs, chunk counts and "
                                   "per-document controls.")

        # ---------- 4) Dashboard ----------
        click_sidebar(page, "Dashboard")
        snap(page, "05-dashboard", "Dashboard — ingestion stats, format/section distribution "
                                   "charts and query performance summary.")

        # ---------- 5) Evaluation (shows last completed run) ----------
        click_sidebar(page, "Evaluation")
        snap(page, "06-evaluation", "Evaluation — agentic QA runs the pipeline against a "
                                    "question set and reports pass rate, failures and latency.")

        browser.close()

    print(f"\nCaptured {len(CAPTIONS)} screenshots to {SHOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())