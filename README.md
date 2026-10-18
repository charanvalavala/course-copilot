# Course Copilot

Course Copilot is an evaluation-first document assistant for course material. It combines semantic embeddings and SQLite FTS5 lexical search, merges rankings with reciprocal-rank fusion, returns inspectable evidence, and can generate cited answers with an optional local Ollama model.

The BITS-specific demo is one use case, not an official BITS Pilani service. Use public or personally authorized documents only. The repository does not include private student records or copyrighted lecture notes.

## What it demonstrates

- PDF, TXT, and Markdown ingestion with content hashing, page provenance, bounded uploads, and overlapping chunks.
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2` plus SQLite FTS5 BM25, fused with reciprocal-rank fusion.
- Evidence excerpts with file and page citations; extractive evidence mode remains available when Ollama is not configured.
- Optional local answer generation through Ollama with a grounded-answer prompt and a citation-number check against the retrieved evidence list.
- A manually labeled 67-query JSONL evaluation set reporting Hit@1, Hit@k, MRR, and abstention accuracy, clearly scoped to retrieval relevance.
- A responsive browser interface for uploads, questions, evidence inspection, and evaluation.
- Docker packaging and a read-only deployment switch.

## Run locally

Requirements: Python 3.11 or newer. Install dependencies and start the API:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
uvicorn course_copilot.api:app --reload
```

Open `http://127.0.0.1:8000`. The embedding model downloads on first use. The app works without Ollama by returning the retrieved excerpts instead of presenting a generated answer as if one existed.

For local generation, install Ollama, pull `qwen2.5:3b` or set `OLLAMA_MODEL`, and start the Ollama service. For container usage, build and run the included Dockerfile with a persistent volume mounted at `/app/data/runtime`.

## Evaluation

The demo corpus has 13 curated source notes linked to official BITS Pilani academic documents. The 67 questions in `data/eval/manual-67.jsonl` comprise 52 manually written, source-labeled answerable questions and 15 out-of-scope questions used to check abstention. This is a small, manually authored demonstration set, not an independent user study or a claim about answer correctness.

Each JSONL row has a question and one or more expected source filenames:

```jsonl
{"question":"When does Semester II registration begin?","expected_sources":["semester-two-start.md"]}
```

The endpoint reports Hit@k and mean reciprocal rank. These do not measure answer correctness, whether a cited passage supports every generated claim, or student outcomes. The optional generation path rejects missing or out-of-range citation numbers, but does not independently score semantic faithfulness. Use a held-out, manually checked query set and report the number of queries, corpus version, pinned embedding model revision, and retrieval settings alongside any score.

## API

- `GET /health`
- `GET /api/documents`
- `POST /api/documents` with multipart `file`
- `DELETE /api/documents/{document_id}`
- `POST /api/query` with `{"question":"...","top_k":5}`
- `POST /api/evaluate` with labeled JSON cases

## Public deployment notes

The Docker image defaults to `COURSE_READ_ONLY=true`. It seeds the sample collection on first startup when the configured database is empty, then disables upload, deletion, and the seed endpoint. The app also applies in-process request limits keyed by the client address; these reset on restart and do not replace shared edge rate limits in multi-worker deployments. Mount persistent storage at `COURSE_DATA_DIR`. Do not expose a writable multi-user service without authentication, quotas, and tenant isolation. Keep model endpoints private and put provider credentials in deployment secrets.

Build and run with persistent data storage:

```powershell
docker build -t course-copilot .
docker run --rm -p 8000:8000 -v course-copilot-data:/app/data/runtime course-copilot
```

The embedding model downloads on the first indexing run. Override `COURSE_READ_ONLY=false` only for a trusted local instance where you need to upload or delete documents.

## Limitations

- Scanned PDFs need OCR before upload; complex tables may not preserve their layout.
- Dense search scans the local chunk set and targets a student-sized corpus.
- Source-level labels do not measure semantic correctness of generated answers.
- The optional language model is local and not needed to inspect retrieval results.
