# Agentic RAG CV Matcher

An agentic Retrieval-Augmented Generation (RAG) system for CV expertise matching with safety guardrails. Upload a pool of CVs and ask natural language questions to find the best-matched experts, with citation to source documents and match confidence scores.

Built as a capstone project demonstrating mastery of GenAI and Agentic AI concepts across 10 course tasks.

## Problem

When hiring or staffing a project, managers need to quickly identify which people in a pool of candidates have the right skills and experience. Manually reading dozens of CVs is slow and error-prone. This tool automates the process: it ingests CVs, builds a semantic knowledge base, and answers natural language queries with ranked, cited matches.

## Features

- **Agentic RAG pipeline**: 4-agent architecture (Planner -> Retriever -> Responder -> Validator) with retry loop
- **Multi-format ingestion**: PDF, TXT, CSV, Excel (.xlsx)
- **Section-aware chunking**: Splits CVs on semantic sections (Experience, Skills, Education) for cleaner retrieval
- **Similarity search**: ChromaDB vector store with cosine similarity ranking
- **Match scoring**: 0-100% confidence scores with evidence citations
- **Guardrails**: LLM Guard for prompt injection detection (input + ingestion), out-of-scope query rejection, output quality validation
- **QA retry loop**: Validator checks answer quality; if rejected, Responder retries with feedback (max 3 attempts)
- **Dashboard**: Ingestion stats, query performance, evaluation results
- **Synthetic data generator**: 20 diverse personas across 8 role categories
- **Bring-your-own API key**: Enter an OpenRouter key in the frontend sidebar; it's sent per-request as a header and never stored server-side - lets you run a public deployment without baking your own key into it

## Architecture

```
User -> Streamlit Frontend (app.py) -> FastAPI Backend (src/main.py)
                                            |
                                            v
                                    Agentic Pipeline (src/agents/orchestrator.py)
                                            |
                                            v
                              +--------------+--------------+
                              |              |              |
                              v              v              v
                           Planner       Retriever      Validator
                       (classify +    (ChromaDB       (quality
                        extract)       similarity)     check)
                              |              |              |
                              v              v              v
                          LLM Client    VectorStore    LLM Client
                       (OpenRouter)   (all-MiniLM)   (OpenRouter)
                                             |
                                             v
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

## API Key Handling

All OpenRouter calls (Planner, Responder, Validator) accept a per-request key:

1. The frontend sends the key entered in the sidebar as an `X-OpenRouter-Key` header on every request.
2. The backend uses that header's key if present; otherwise it falls back to `OPENROUTER_API_KEY` from `.env` / `config.json` (convenient for local dev).
3. **Neither the frontend nor the backend ever writes the key to disk or logs it.**
4. `GET /api/key/validate` (sidebar "Validate Key" button) checks the key against OpenRouter's own `/api/v1/auth/key` endpoint - free, doesn't consume completion credits, and reports usage/limit if valid.

For a public deployment where you don't want your own key exposed or spent by strangers, leave `OPENROUTER_API_KEY` unset in the deployment environment - every visitor must then supply their own key to use Query or Evaluation (document upload doesn't need a key at all; ingestion never calls an LLM).

## Deployment

A single `Dockerfile` packages both the FastAPI backend and the Streamlit frontend into one container (`docker-entrypoint.sh` starts the backend, waits for it to become healthy, then starts the frontend, which talks to the backend over `localhost`).

```bash
docker build -t agentic-rag-cv .

docker run -p 8501:8501 -p 8000:8000 \
  -v cvmatcher_data:/app/.data \
  agentic-rag-cv
```

- Open the app at `http://localhost:8501`.
- The `-v cvmatcher_data:/app/.data` volume persists the ChromaDB knowledge base, uploads, evaluation history, and logs across container restarts - omit it for a fully ephemeral deployment.
- Do **not** set `OPENROUTER_API_KEY` in the container's environment for a public deployment - see [API Key Handling](#api-key-handling) above.


### Where to host it

This app still needs two long-running processes (not short-lived serverless functions) and local disk for ChromaDB persistence. That combination rules out **Vercel** - it's built for serverless functions and static/Next.js sites with short execution limits and no persistent local disk, not long-lived Docker services with WebSocket connections (which is how Streamlit works).

Better fits for this Dockerfile:
- **Railway / Render / Fly.io** - straightforward Docker deploys with persistent volumes and generous-enough free/hobby tiers. Probably the least friction.
- **A plain VM** (DigitalOcean, Hetzner, EC2) running `docker run` directly - full control, disk persists by default, no serverless constraints.
- **Google Cloud Run** - works, but only exposes one public port per service and has an ephemeral filesystem by default. Since only the Streamlit frontend needs to be internet-facing (it calls FastAPI over `localhost` inside the same container), you can deploy the container as-is exposing port 8501 only; add a mounted volume (or accept that the knowledge base resets on redeploy/scale-to-zero) for persistence.

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
# Edit config.json to change model defaults or application settings.
# Create a .env file and set your OpenRouter API key secret:
echo "OPENROUTER_API_KEY=your_key_here" > .env

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
2. Go to **Home** -> Upload CV files (PDF/TXT/CSV/Excel)
3. Go to **Query** -> Ask questions like "Find a Java developer with Spring Boot experience"
4. View match scores, citations, and evidence

### CLI - Synthetic Data Generator

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

# Ask a question (X-OpenRouter-Key is optional if OPENROUTER_API_KEY is set server-side)
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: sk-or-..." \
  -d '{"question": "Find a Java developer"}'

# Check whether a key is valid
curl http://localhost:8000/api/key/validate -H "X-OpenRouter-Key: sk-or-..."

# List documents
curl http://localhost:8000/api/documents/

# Dashboard stats
curl http://localhost:8000/api/dashboard/stats

# Start evaluation (runs in the background; poll /progress for status)
curl -X POST http://localhost:8000/api/evaluation/start
curl http://localhost:8000/api/evaluation/progress
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
|-- app.py                      # Streamlit frontend (5 pages)
|-- requirements.txt            # Python dependencies (added pyyaml)
|-- config.json                 # Global configuration defaults
|-- .env.example                # Template for OPENROUTER_API_KEY secret
|-- README.md                   # This file
|-- run.sh                      # Convenience run script (local dev)
|-- Dockerfile                  # Backend + frontend packaged into one container
|-- docker-entrypoint.sh        # Starts backend, waits for health, then starts frontend
|-- .dockerignore
|-- src/
|   |-- config.py               # Settings loader (merges config.json + .env)
|   |-- main.py                 # FastAPI app with route registration
|   |-- api/                    # API route handlers
|   |   |-- deps.py             # Shared dependencies (per-request OpenRouter key extraction)
|   |   |-- apikey.py           # GET /validate - checks a key against OpenRouter, no server-side storage
|   |   |-- documents.py        # Upload, list, remove documents
|   |   |-- query.py            # Query knowledge base
|   |   |-- dashboard.py        # Dashboard statistics
|   |   |-- evaluation.py       # Background evaluation run + progress polling
|   |   +-- synthetic.py        # Generate sample data
|   |-- agents/                 # Agentic RAG pipeline
|   |   |-- context.py          # PipelineContext dataclass
|   |   |-- llm_client.py       # OpenRouter API wrapper
|   |   |-- planner.py          # Query classification + scope detection
|   |   |-- retriever.py        # ChromaDB similarity search + ranking
|   |   |-- responder.py        # Answer generation with citations
|   |   |-- validator.py        # Output quality validation
|   |   +-- orchestrator.py     # Pipeline orchestration + retry loop
|   |-- ingestion/              # Document processing
|   |   |-- extractors.py       # PDF/TXT/CSV/Excel text extraction
|   |   |-- chunker.py          # Section-aware chunking (with localized headers)
|   |   +-- pipeline.py         # Extract -> Chunk -> Scan -> Store
|   |-- vectorstore/
|   |   +-- store.py            # ChromaDB wrapper (add, query, delete, list)
|   |-- guardrails/
|   |   +-- scanner.py          # LLM Guard wrapper
|   |-- data/                   # Synthetic data generator
|   |   |-- resources/
|   |   |   +-- generator_data.yaml # Persona templates, pools, and text generation assets
|   |   |-- personas.py         # Load generator pool & template personas
|   |   |-- generator.py        # PDF/TXT/CV renderer
|   |   +-- cli.py              # CLI entry point
|   +-- utils/
|       |-- query_log.py        # Query statistics tracking
|       |-- json_parser.py      # Robust JSON cleaner and parser
|       |-- json_store.py       # Thread-safe read/append helpers for JSON-file-backed storage (using RLock)
|       +-- filenames.py        # Mojibake filename repair (shared across upload/query/delete)
|-- tests/                      # pytest test suite (56 tests)
|   |-- resources/
|   |   +-- default_questions.json # Evaluation question dataset (test-local resource)
|   |-- test_extractors.py      # Document extraction tests
|   |-- test_chunker.py         # Section-aware chunking tests
|   |-- test_guardrails.py      # Prompt injection detection tests
|   |-- test_pipeline.py        # Agent pipeline integration tests
|   +-- test_api.py             # API route tests
|-- sample_data/                # Generated synthetic CVs
|   |-- pdf/                    # 20 PDF CVs
|   |-- txt/                    # 20 TXT CVs
|   +-- personas.csv            # All personas in CSV
+-- data/                       # Runtime data (gitignored)
    |-- chromadb/               # Vector store persistence
    |-- uploads/                # Uploaded documents
    |-- evaluation/             # Evaluation results (eval_results.json)
    +-- query_log.json          # Query statistics
```

## Limitations and Challenges

1. **Embedding quality**: Uses all-MiniLM-L6-v2 (384-dim) for speed. For production, swap to OpenAI text-embedding-3-small via config change.

2. **Offline classifier precision**: The keyword-based fallback classifier (used when no OpenRouter key is configured) has lower precision than the LLM-powered classifier. It may reject some valid queries or accept some out-of-scope ones.

3. **PDF extraction**: Text-based PDFs only. Scanned/image-based PDFs will produce empty text. OCR was dropped from scope.

4. **Similarity scores**: Scores from all-MiniLM-L6-v2 are relatively low (0.3-0.5 range). The ranking is correct; the absolute values are not comparable across embedding models.

5. **No persistence of query history**: Query log is JSON-file-backed. Although thread-safety locks have been implemented to prevent concurrent write corruptions, it is not suitable for high-volume production use.

6. **Shared knowledge base, no auth**: Each visitor can bring their own OpenRouter key (see API Key Handling), but the uploaded CVs and ChromaDB knowledge base are global - anyone who can reach a public deployment can see, query, and delete the same documents. Fine for a personal demo behind an unlisted URL; not a substitute for real multi-tenancy or access control.

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
| 7 | RAG pipeline | Retrieve -> LLM generates grounded, cited answer |
| 8 | Agent-based reasoning | 4-agent architecture with retry loop |
| 9 | Safety controls | LLM Guard injection defense + scope rejection + output validation |
| 10 | Deploy + document | FastAPI + Streamlit deployment, this README |

