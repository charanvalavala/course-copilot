# Course Copilot

**Live demo:** [Open Course Copilot](https://charan-course-copilot.onrender.com). Hosted on Render free tier; the service sleeps when idle, so the first visit may take about 50 seconds to wake.

Course Copilot is an evaluation-first document assistant for course material. It combines semantic embeddings and SQLite FTS5 lexical search, merges rankings with reciprocal-rank fusion, returns inspectable evidence, and can generate cited answers with an optional local Ollama model.

The BITS-specific demo is one use case, not an official BITS Pilani service. Use public or personally authorized documents only. The repository does not include private student records or copyrighted lecture notes.

## What it demonstrates

- PDF, TXT, and Markdown ingestion with content hashing, page provenance, bounded uploads, and overlapping chunks.
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2` plus SQLite FTS5 BM25, fused with a 2:1 BM25-to-dense reciprocal-rank weight selected in a documented retrieval ablation.
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

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
uvicorn course_copilot.api:app --reload
```

For local generation, install Ollama, pull `qwen2.5:3b` or set `OLLAMA_MODEL`, and start the Ollama service. For container usage, build and run the included Dockerfile with a persistent volume mounted at `/app/data/runtime`.

## Evaluation

The demo corpus has 13 curated source notes linked to official BITS Pilani academic documents. The 67 questions in `data/eval/manual-67.jsonl` comprise 52 manually written, source-labeled answerable questions and 15 out-of-scope questions used to check abstention. This is a small, manually authored demonstration set, not an independent user study or a claim about answer correctness.

Each JSONL row has a question and one or more expected source filenames:

```jsonl
{"question":"When does Semester II registration begin?","expected_sources":["semester-two-start.md"]}
```

After starting the API, reproduce the bundled evaluation with:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/evaluate/demo?top_k=5" | ConvertTo-Json -Depth 6
```

The endpoint reports Hit@k and mean reciprocal rank. These do not measure answer correctness, whether a cited passage supports every generated claim, or student outcomes. The optional generation path rejects missing or out-of-range citation numbers, but does not independently score semantic faithfulness. Use a held-out, manually checked query set and report the number of queries, corpus version, pinned embedding model revision, and retrieval settings alongside any score.

### Retrieval ablation

Reproduce the same-query comparison after the embedding model has downloaded:

```powershell
$env:PYTHONPATH = "src"
python scripts/ablation.py
```

```bash
PYTHONPATH=src python scripts/ablation.py
```

The current 13-note corpus results use `top_k=5` and the pinned MiniLM revision. Retrieval metrics are computed over the 52 answerable questions; abstention accuracy is computed over the 15 out-of-scope questions.

| Retriever | Hit@1 | Hit@5 | MRR | Abstention accuracy |
| --- | ---: | ---: | ---: | ---: |
| BM25 only | 0.904 | 1.000 | 0.946 | 0.933 |
| Dense only | 0.712 | 0.904 | 0.808 | 1.000 |
| Equal-weight RRF | 0.846 | 1.000 | 0.923 | 0.933 |
| Weighted RRF, BM25:dense = 2:1 | 0.923 | 1.000 | 0.962 | 0.933 |
| Weighted RRF, 2:1 with a dense gate | 0.865 | 0.923 | 0.894 | 1.000 |
| Weighted RRF, BM25:dense = 3:1 | 0.904 | 1.000 | 0.952 | 0.933 |

The default uses the 2:1 weighting: it ranked the expected source first on 48 of 52 answerable queries, compared with 44 for equal-weight RRF. The dense gate removed the only false positive on the 15 out-of-scope prompts, but also lowered Hit@5 from 1.000 to 0.923, so it was not selected. The remaining false positive is “How many seats are available in this elective now?”; the demo notes discuss course offerings but do not provide live seat counts.

Against BM25 alone, the selected hybrid setting moved Hit@1 from 47/52 to 48/52 and MRR from 0.946 to 0.962 on this same development set. That is a one-query top-rank gain, not evidence that hybrid retrieval will outperform BM25 on a larger or independent corpus.

The four Hit@1 misses all placed the expected file at rank 2: whether a bulletin guarantees a course will run; when students can see the semester's course list; where the binding enrollment rules are; and which document lists courses actually running this term. This explains the observed MRR of 0.962 on this set.

The benchmark is small and self-authored. The weighting was selected on this same development set, and relevance labels identify source files rather than exact supporting passages. The report at `reports/retrieval-ablation.json` includes per-query retrieved sources and the model revision. It is a development comparison, not an independent test score.

Run the focused tests with:

```powershell
$env:PYTHONPATH = "src"
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest -q
```

## API

- `GET /health`
- `GET /api/documents`
- `POST /api/documents` with multipart `file`
- `DELETE /api/documents/{document_id}`
- `POST /api/query` with `{"question":"...","top_k":5}`
- `POST /api/evaluate` with labeled JSON cases
- `GET /api/evaluate/demo?top_k=5` with the included 67-query set

## Public deployment notes

The Docker image defaults to `COURSE_READ_ONLY=true`. It seeds the sample collection on first startup when the configured database is empty, then disables upload, deletion, and the seed endpoint. The app also applies in-process request limits keyed by the client address; these reset on restart and do not replace shared edge rate limits in multi-worker deployments. Mount persistent storage at `COURSE_DATA_DIR`. Do not expose a writable multi-user service without authentication, quotas, and tenant isolation. Keep model endpoints private and put provider credentials in deployment secrets.

Build and run with persistent data storage:

```powershell
docker build -t course-copilot .
docker run --rm -p 8000:8000 -v course-copilot-data:/app/data/runtime course-copilot
```

The embedding model downloads on the first indexing run. Override `COURSE_READ_ONLY=false` only for a trusted local instance where you need to upload or delete documents.

The repository includes a Render Blueprint at `render.yaml` for a read-only API and browser demo. Create the service from the repository's Blueprint in Render; its temporary demo data and model cache are rebuilt after an instance restart.

## Limitations

- Scanned PDFs need OCR before upload; complex tables may not preserve their layout.
- Dense search scans the local chunk set and targets a student-sized corpus.
- Source-level labels do not measure semantic correctness of generated answers.
- The optional language model is local and not needed to inspect retrieval results.
