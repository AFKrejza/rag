"""Central configuration. Values can be overridden with environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = Path(os.getenv("RAG_PDF_DIR", ROOT / "data" / "pdfs"))
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "compiler_docs")

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("RAG_LLM_EFFORT", "low")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Also pass the chunks just before/after each hit (same page) to the LLM.
INCLUDE_NEIGHBOURS = os.getenv("RAG_INCLUDE_NEIGHBOURS", "1") == "1"
# Cosine distance (0 = identical, 2 = opposite). Chunks farther than this are
# treated as irrelevant; if none survive, the LLM is not called at all.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.75"))

IDK_ANSWER = "I don't know based on the provided documents."
