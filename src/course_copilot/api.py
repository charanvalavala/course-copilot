from __future__ import annotations

from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from course_copilot import database
from course_copilot.config import DEMO_DIR, EVAL_PATH, MAX_UPLOAD_BYTES, READ_ONLY, STATIC_DIR
from course_copilot.evaluation import evaluate
from course_copilot.generation import answer
from course_copilot.ingestion import ingest, parse_file
from course_copilot.limits import enforce_limit
from course_copilot.retrieval import retrieve


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize()
    if READ_ONLY and not database.list_documents():
        seed_demo_documents()
    yield


app = FastAPI(title="Course Copilot", version="1.0.0", description="Hybrid retrieval, evidence-grounded document Q&A, and retrieval evaluation.", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class Case(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    expected_sources: list[str] = Field(default_factory=list)
    should_abstain: bool = False


class Evaluation(BaseModel):
    cases: list[Case] = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(database.list_documents())}


@app.get("/api/config")
def app_config():
    return {"read_only": READ_ONLY}


@app.get("/api/documents")
def documents():
    return database.list_documents()


def seed_demo_documents():
    import hashlib

    seeded = []
    for path in sorted(DEMO_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        first, _, body = content.partition("\n")
        source_url = first.removeprefix("SOURCE_URL: ").strip()
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if database.has_document_hash(digest):
            seeded.append({"name": path.name, "duplicate": True})
            continue
        chunks = parse_file(path.name, body.encode("utf-8"))
        result = database.add_document(path.name, source_url, digest, chunks)
        seeded.append({"name": path.name, **result})
    return {"documents": seeded}


@app.post("/api/demo/seed")
def seed_demo(request: Request):
    if READ_ONLY:
        raise HTTPException(status_code=403, detail="Demo seeding is disabled in this read-only deployment.")
    enforce_limit(request, "seed", 3)
    return seed_demo_documents()


@app.post("/api/documents")
async def upload(request: Request, file: UploadFile = File(...)):
    if READ_ONLY:
        raise HTTPException(status_code=403, detail="Uploads are disabled in this read-only demo.")
    enforce_limit(request, "upload", 5)
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file exceeds the 8 MB upload limit.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = ingest(file.filename or "upload.txt", content)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int):
    if READ_ONLY:
        raise HTTPException(status_code=403, detail="Deletion is disabled in this read-only demo.")
    if not database.remove_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True}


@app.post("/api/query")
def query(http_request: Request, request: Query):
    enforce_limit(http_request, "query", 20)
    try:
        results = retrieve(request.question, request.top_k)
        generated = answer(request.question, results)
        return {**generated, "evidence": results}
    except Exception as error:
        raise HTTPException(status_code=503, detail="The retrieval or generation service is unavailable.") from error


@app.post("/api/evaluate")
def run_evaluation(http_request: Request, request: Evaluation):
    enforce_limit(http_request, "evaluate", 3)
    try:
        return evaluate([case.model_dump() for case in request.cases], retrieve, request.top_k)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/evaluate/demo")
def evaluate_demo(request: Request, top_k: int = 5):
    enforce_limit(request, "evaluate-demo", 3)
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 10.")
    cases = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        return evaluate(cases, retrieve, top_k)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
