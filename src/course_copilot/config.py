from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("COURSE_DATA_DIR", str(ROOT / "data" / "runtime")))
STATIC_DIR = ROOT / "static"
DEMO_DIR = ROOT / "data" / "demo"
EVAL_PATH = ROOT / "data" / "eval" / "manual-67.jsonl"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
os.environ.setdefault("HF_HOME", str(DATA_DIR / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_DENSE_SIMILARITY = float(os.getenv("MIN_DENSE_SIMILARITY", "0.45"))
READ_ONLY = os.getenv("COURSE_READ_ONLY", "false").lower() == "true"
