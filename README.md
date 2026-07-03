# Agentic RAG CV Matcher

An agentic Retrieval-Augmented Generation (RAG) system for CV expertise matching with safety guardrails. Upload a pool of CVs and ask natural language questions to find the best-matched experts, with citation to source documents and match confidence scores.

Built as a capstone project demonstrating mastery of GenAI and Agentic AI concepts across 10 course tasks.

## Problem

When hiring or staffing a project, managers need to quickly identify which people in a pool of candidates have the right skills and experience. Manually reading dozens of CVs is slow and error-prone. This tool automates the process: it ingests CVs, builds a semantic knowledge base, and answers natural language queries with ranked, cited matches.

## Features

- **Agentic RAG pipeline**: 4-agent architecture (Planner → Retriever → Responder → Validator) with retry loop
- **Multi-format ingestion**: PDF, TXT, CSV, Excel (.xlsx)
- **Section-aware chunking**: Splits CVs on semantic sections (Experience, Skills, Education) for cleaner retrieval
- **Similarity search**: ChromaDB vector store with cosine similarity ranking
- **Match scoring**: 0-100% confidence scores with evidence citations
- **Guardrails**: LLM Guard for prompt injection detection (input + ingestion), out-of-scope query rejection, output quality validation
- **QA retry loop**: Validator checks answer quality; if rejected, Responder retries with feedback (max 3 attempts)
- **Dashboard**: Ingestion stats, query performance, evaluation results
- **Synthetic data generator**: 20 diverse personas across 8 role categories

## Architecture

```
User → Streamlit Frontend (app.py) → FastAPI Backend (src/main.py)
                                            ↓
                                    Agentic Pipeline (src/agents/orchestrator.py)
                                            ↓
                              ┌──────────────┼──────────────┐
                              ↓              ↓              ↓
                           Planner       Retriever      Validator
                       (classify +    (ChromaDB       (quality
                        extract)       similarity)     check)
                              ↓              ↓              ↓
                          LLM Client    VectorStore    LLM Client
                       (OpenRouter)   (all-MiniLM)   (OpenRouter)
                                              ↓
                                         Responder
                                      (answer + citations)
```

### Agent Pipeline Flow

1. **Planner** (`planner.py`): Classifies query as keyword/similarity/complex/out-of-scope. Extracts required skills. Rejects non-CV queries with polite message.
2. **Retriever** (`retriever.py`): Builds search query from extracted requirements. Executes ChromaDB similarity search. Ranks and deduplicates results.
3. **Responder** (`responder.py`): Generates grounded answer with match scores and citations using LLM. Incorporates Validator feedback on retries.
4. **Validator** (`validator.py`): Checks answer against 5 quality criteria (grounded, cited, correct, complete, scoped). Returns structured feedback for retry if failed.

### Guardrails

| Layer | Mechanism | What it catches |
|-------|-----------|-----------------|
| **Input** | LLM Guard PromptInjection scanner (DeBERTa) | Prompt injection in user queries |
| **Ingestion** | Same scanner on document chunks | Instructions embedded in uploaded documents |
| **Scope** | Planner offline/LLM classification | Out-of-scope queries (weather, code, poems) |
| **Output** | Validator agent with retry loop | Hallucinated claims, missing citations, incomplete answers |

## Installation

### Prerequisites

- Python 3.12+
- pip

### Step by step

```bash
# 1. Clone the repository
git clone <repository-url>
cd agentic-rag-cv

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your OpenRouter API key:
# OPENROUTER_API_KEY=sk-or-your-key-here

# 5. Generate sample data (optional, for testing)
python -m src.data.cli --count 20 --output ./sample_data

# 6. Start the backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 7. Start the frontend (new terminal)
streamlit run app.py --server.port 8501
```

The app is now running:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API docs**: http://localhost:8000/docs

## Usage

### Web Interface

1. Open http://localhost:8501
2. Go to **Home** → Upload CV files (PDF/TXT/CSV/Excel)
3. Go to **Query** → Ask questions like "Find a Java developer with Spring Boot experience"
4. View match scores, citations, and evidence

### CLI — Synthetic Data Generator

```bash
# Generate 20 sample CVs
python -m src.data.cli --count 20 --output ./sample_data

# Generate 10 CSVs only
python -m src.data.cli --count 10 --output ./sample_data --format csv

# Custom seed for reproducibility
python -m src.data.cli --count 20 --seed 123
```

### API

```bash
# Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@cv.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "Find a Java developer"}'

# List documents
curl http://localhost:8000/api/documents/

# Dashboard stats
curl http://localhost:8000/api/dashboard/stats

# Run evaluation
curl -X POST http://localhost:8000/api/evaluation/run
```

## Libraries Used

| Library | Purpose |
|---------|---------|
| **FastAPI** | REST API backend |
| **Streamlit** | Interactive web frontend |
| **ChromaDB** | Vector database for semantic search |
| **sentence-transformers** | Text embeddings (all-MiniLM-L6-v2) |
| **langchain-text-splitters** | Recursive text splitting for chunking |
| **LLM Guard** (Protect AI) | Prompt injection detection (DeBERTa classifier) |
| **pypdf** | PDF text extraction |
| **openpyxl** | Excel file reading |
| **fpdf2** | Synthetic CV PDF generation |
| **Faker** | Synthetic persona data |
| **httpx** | OpenRouter API client |
| **pydantic-settings** | Configuration from .env |
| **uvicorn** | ASGI server for FastAPI |

## File Structure

```
agentic-rag-cv/
├── app.py                      # Streamlit frontend (5 pages)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment configuration template
├── README.md                   # This file
├── run.sh                      # Convenience run script
├── src/
│   ├── config.py               # Pydantic Settings (all config from .env)
│   ├── main.py                 # FastAPI app with route registration
│   ├── api/                    # API route handlers
│   │   ├── documents.py        # Upload, list, remove documents
│   │   ├── query.py            # Query knowledge base
│   │   ├── dashboard.py        # Dashboard statistics
│   │   ├── evaluation.py       # Run evaluation suite
│   │   └── synthetic.py        # Generate sample data
│   ├── agents/                 # Agentic RAG pipeline
│   │   ├── context.py          # PipelineContext dataclass
│   │   ├── llm_client.py       # OpenRouter API wrapper
│   │   ├── planner.py          # Query classification + scope detection
│   │   ├── retriever.py        # ChromaDB similarity search + ranking
│   │   ├── responder.py        # Answer generation with citations
│   │   ├── validator.py        # Output quality validation
│   │   └── orchestrator.py     # Pipeline orchestration + retry loop
│   ├── ingestion/              # Document processing
│   │   ├── extractors.py       # PDF/TXT/CSV/Excel text extraction
│   │   ├── chunker.py          # Section-aware chunking
│   │   └── pipeline.py         # Extract → Chunk → Scan → Store
│   ├── vectorstore/
│   │   └── store.py            # ChromaDB wrapper (add, query, delete, list)
│   ├── guardrails/
│   │   └── scanner.py          # LLM Guard wrapper
│   ├── data/                   # Synthetic data generator
│   │   ├── personas.py         # 20 persona templates (8 role categories)
│   │   ├── generator.py        # PDF/TXT/CV renderer
│   │   └── cli.py              # CLI entry point
│   └── utils/
│       └── query_log.py        # Query statistics tracking
├── tests/                      # pytest test suite (55 tests)
│   ├── test_extractors.py      # Document extraction tests
│   ├── test_chunker.py         # Section-aware chunking tests
│   ├── test_guardrails.py      # Prompt injection detection tests
│   ├── test_pipeline.py        # Agent pipeline integration tests
│   └── test_api.py             # API route tests
├── sample_data/                # Generated synthetic CVs
│   ├── pdf/                    # 20 PDF CVs
│   ├── txt/                    # 20 TXT CVs
│   └── personas.csv            # All personas in CSV
└── data/                       # Runtime data (gitignored)
    ├── chromadb/               # Vector store persistence
    ├── uploads/                # Uploaded documents
    ├── evaluation/             # Evaluation results (eval_results.json)
    └── query_log.json          # Query statistics
```

## Limitations and Challenges

1. **Embedding quality**: Uses all-MiniLM-L6-v2 (384-dim) for speed. For production, swap to OpenAI text-embedding-3-small via config change.

2. **Offline classifier precision**: The keyword-based fallback classifier (used when no OpenRouter key is configured) has lower precision than the LLM-powered classifier. It may reject some valid queries or accept some out-of-scope ones.

3. **PDF extraction**: Text-based PDFs only. Scanned/image-based PDFs will produce empty text. OCR was dropped from scope.

4. **Similarity scores**: Scores from all-MiniLM-L6-v2 are relatively low (0.3-0.5 range). The ranking is correct; the absolute values are not comparable across embedding models.

5. **No persistence of query history**: Query log is JSON-file-backed. Not suitable for high-volume production use.

6. **Single-user**: No authentication, no multi-user support. Designed for demonstration, not shared deployment.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Course Task Mapping

| # | Task | Implementation |
|---|------|----------------|
| 1 | Project foundation | Python + FastAPI + Streamlit, `.env` config |
| 2 | User interaction layer | Streamlit frontend (5 pages) |
| 3 | Document ingestion | PDF/TXT/CSV/Excel extraction + section-aware chunking |
| 4 | Semantic search prep | Recursive + section-aware chunking with metadata |
| 5 | Vector knowledge store | ChromaDB + all-MiniLM-L6-v2 embeddings |
| 6 | Intelligent retrieval | Similarity search with ranking + deduplication |
| 7 | RAG pipeline | Retrieve → LLM generates grounded, cited answer |
| 8 | Agent-based reasoning | 4-agent architecture with retry loop |
| 9 | Safety controls | LLM Guard injection defense + scope rejection + output validation |
| 10 | Deploy + document | FastAPI + Streamlit deployment, this README |

